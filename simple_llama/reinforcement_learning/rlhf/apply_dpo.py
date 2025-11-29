import os
import time
import random
import math
import inspect
import numpy as np
import torch
from tokenizers import Tokenizer, decoders

from simple_llama.reinforcement_learning.rlhf.dpo_config import DPOConfig
from simple_llama.reinforcement_learning.rlhf.rl_dataset_loader import RLDatasetLoader
from simple_llama.reinforcement_learning.rlhf.utils import align_and_pad_data, eval_model, rl_prompts
from simple_llama.reinforcement_learning.rlhf.preference_loss import PreferenceLoss
from simple_llama.finetune.format_llm_prompt import format_inference_prompt
from simple_llama.pretraining.llama_transformer import LLaMaTransformer
from simple_llama.pretraining.config import TrainingConfig
from simple_llama.pretraining.lr_scheduler import Scheduler
from simple_llama.pretraining.utils import check_log_file_existence


# torch.serialization.add_safe_globals({TrainingConfig})


# Manual seeding for reproducibility testings
seed = 89
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# Setting to 'high' uses TF32 rather than FP32, which makes the training process faster (varies on machines)
# Can set to 'medium' for even faster training, though will be loss in performance
# Check out the documentations https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision("high")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Currently using {device=}")


# Get the config
dpo_config = DPOConfig()


# Hyperparameters
# ==================================
ckpt_dir = dpo_config.ckpt_dir
log_file = dpo_config.log_file

grad_accum_steps = dpo_config.grad_accum_steps
eval_interval = dpo_config.eval_interval * grad_accum_steps

beta = dpo_config.beta
enable_compilation = dpo_config.enable_compilation

dynamic_padding = dpo_config.dynamic_padding
# ==================================

log_file = check_log_file_existence(log_file=log_file, ddp=False)


with open(log_file, "a") as f:
    columns = ["step", "progress (%)", "Training Loss", "Validation Loss",
               "Learning Rate", "L2 Norm", "Time Per Evaluation"]
    f.write(",".join(columns))
    f.write("\n")


os.makedirs(ckpt_dir, exist_ok=True)


print("\nTraining Configurations:")
print("=" * 30)
for field_name in dpo_config.__dataclass_fields__:
    value = getattr(dpo_config, field_name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        print(f"{field_name}: {value:_}")
    else:
        print(f"{field_name}: {value}")
print("=" * 20)
print("\n")



tokenizer = Tokenizer.from_file(dpo_config.tokenize_path)
tokenizer.model.unk_token = "<UNK>"
tokenizer.decoder = decoders.ByteLevel()


ckpt = torch.load(dpo_config.model_path, map_location=device)

training_config: TrainingConfig = ckpt["config"]

# Update configs as needed
training_config.dropout = dpo_config.dropout
training_config.use_flash_attention = dpo_config.use_flash_attention


dataset_loader = RLDatasetLoader(json_filepath=dpo_config.rlhf_dataset_path,
                                 tokenizer=tokenizer,
                                 batch_size=dpo_config.batch_size,
                                 max_seq_len=training_config.max_seq_len,
                                 train_split=dpo_config.train_split,
                                 device=device,
                                 )

train_iterations = (len(dataset_loader.train_dataset)//dpo_config.batch_size) * dpo_config.epochs
optimization_steps = train_iterations // grad_accum_steps  # Number of times to step the optimizer

print(f"{train_iterations=}")
print(f"{optimization_steps=}")


# Load currently LLM, one undergoing rlhf
model = LLaMaTransformer(training_config, tokenizer, device)
model.to(device)
model.train()
model.load_state_dict(ckpt["model_state_dict"], strict=True)

# Load the ref model and freeze gradients
ref_model = LLaMaTransformer(training_config, tokenizer, device)
ref_model.to(device)
ref_model.eval()
ref_model.load_state_dict(ckpt["model_state_dict"], strict=True)

for param in ref_model.parameters():
    param.requires_grad = False


n_params = sum([p.numel() for p in model.parameters()])
print(f"There is {n_params / 1e6:.1f}M parameters in the model")
print("\n")


fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device == "cuda"
extra_args = dict(fused=True) if use_fused else dict()

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=dpo_config.max_lr,
                              betas=(dpo_config.beta1, dpo_config.beta2),
                              weight_decay=dpo_config.weight_decay,
                              **extra_args
                              )
print(f"Using fused optimizer: {use_fused}\n")

# Init custom preference loss obj
criterion = PreferenceLoss(beta=beta)

pad_token = "<PAD>"
pad_id = tokenizer.encode(pad_token).ids
assert len(pad_id) == 1, f"{pad_token=} should be a special token with a single value!"
pad_id = pad_id[0]


scheduler = Scheduler(torch_optimizer=optimizer,
                      schedule="cosine",
                      training_steps=optimization_steps,
                      warmup_steps=dpo_config.warmup_iterations,
                      max_lr=dpo_config.max_lr,
                      min_lr=dpo_config.min_lr)


if enable_compilation:
    compiled_model = torch.compile(model)


# Training Loop
# ------------------------
start = time.time()
all_losses = []
save_ckpt = {}
norm = float("inf")
step = 1
max_seq_len = training_config.max_seq_len


use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print(f"Using auto mixed precision: {use_amp}")

for epoch in range(dpo_config.epochs):
    current_val_epoch = dataset_loader.val_epoch
    current_train_epoch = dataset_loader.train_epoch

    print("\n" + "=" * 20)
    print(f"Current Epoch: {epoch+1}")
    print("=" * 20 + "\n")

    while dataset_loader.train_epoch == current_train_epoch:
        # Each training iteration processes one batch of DPO examples.
        #
        # The dataset loader returns a list of tuples:
        #   (accepted_chat, accepted_suffix, rejected_chat, rejected_suffix)
        # where:
        #   - accepted_chat / rejected_chat are the full tokenized conversations
        #     (system + user + assistant response),
        #   - accepted_suffix / rejected_suffix are the tokenized final assistant
        #     responses used to compute log-probabilities for preference loss.
        #
        # align_and_pad_data() converts this batch into two tensors:
        #   x : concatenated chat inputs (accepted/rejected interleaved)
        #   y : left-padded response targets aligned to x
        # Both have shape [2 * batch_size, seq_len], with even indices being
        # accepted samples and odd indices being rejected ones.
        #
        # The model and frozen reference model each compute logits for x;
        # PreferenceLoss then compares their log-probabilities over y to compute the DPO objective.
        batch = dataset_loader.get_batch(train=True)

        # Encode and pad them accordingly, returns a tensor of shape (batch_size, seq_len)
        # seq_len == len of longest sequence in batch, if dynamic_padding, else seq_len == training_config.max_seq_len
        x, y = align_and_pad_data(batch=batch,
                                  pad_id=pad_id,
                                  max_seq_len=max_seq_len,
                                  dynamic=dynamic_padding,
                                  device=device
                                  )

        with torch.autocast(device_type=device, dtype=torch.bfloat16 if use_amp else torch.float32):
            theta_pred = model(x) if not enable_compilation else compiled_model(x)
            ref_pred = ref_model(x)

        loss = criterion.calculate_loss(theta_logits=theta_pred,
                                        ref_logits=ref_pred,
                                        target_ids=y,
                                        pad_token=pad_id,
                                        device=device)

        train_loss_value = loss.item()
        loss /= grad_accum_steps
        loss.backward()

        all_losses.append(train_loss_value)

        if step % grad_accum_steps == 0:
            scheduler.step(step // grad_accum_steps)  # Set the lr first
            # Clip gradients, step optimizer, and set to None
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevents unstable learning
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % eval_interval == 0:
            single_val_loss = eval_model(model=model, ref_model=ref_model, criterion=criterion,
                                         dataset_loader=dataset_loader, eval_num_samples=dpo_config.eval_num_samples,
                                         use_amp=use_amp, full_eval=False, pad_id=pad_id, max_seq_len=max_seq_len,


                                         dynamic=dynamic_padding, device=device)
            if torch.cuda.is_available() and device == "cuda":
                torch.cuda.synchronize()  # More accurate measurement for `elapsed` since cuda kernels are async

            elapsed = time.time() - start
            current_lr = optimizer.param_groups[0]["lr"]

            with open(log_file, "a") as f:
                write_data = [step,  # Current step
                              round((step / train_iterations) * 100, 2),  # Progress
                              round(train_loss_value, 4),  # Training loss value
                              round(single_val_loss, 4),  # Validation Loss (Small Batch)
                              round(current_lr, 4),  # Learning Rate
                              round(norm.item(), 4),  # L2 norm of the gradients
                              int(elapsed),  # Time taken
                              ]

                write_data = [str(wd) for wd in write_data]
                f.write(",".join(write_data))
                f.write("\n")

            print("----------------")
            print(f"Step: {step} steps   |   "
                  f"Training Progress: {(step / train_iterations) * 100:.2f}%   |   "
                  f"Training Loss: {train_loss_value:.4f}   |   "
                  f"Validation Loss (Interval Subset): {single_val_loss:.4f}   |   "
                  f"Learning Rate: {current_lr:.5f}   |   "
                  f"Norm: {norm.item():.4f}   |   "
                  f"Time: {int(elapsed)}s")

            start = time.time()

            # Setting max_new_tokens to a lower value does increase overall training speed by a noticeable amount (in my case)
            # Exponential generation interval logic
            if 'next_gen_step' not in locals():
                next_gen_step = step  # initialize on first use

            if step >= next_gen_step:
                print("\n")
                # EOS token here is technically a misnomer, but that's fine. (Should rename it to stop_token?)
                rand_prompt = random.choice(rl_prompts)
                formatted_prompt = format_inference_prompt(user=rand_prompt["User"], assistant=[], template="CUSTOM")
                print(model.generate(formatted_prompt, 1024, 1.0, 0.8, eos_token=tokenizer.encode("<EOA>").ids[0]))
                next_gen_step = int(next_gen_step * dpo_config.model_gen_multiplier)
                print("\n")
                print(f"Sampled generation at {step=}, next at {next_gen_step=}")

            print("----------------")

        step += 1

    # Save Checkpoint/Final Weights
    if epoch % dpo_config.ckpt_epochs == 0 or epoch == dpo_config.epochs-1:
        save_ckpt["model_state_dict"] = model.state_dict()
        save_ckpt["config"] = training_config

        n = 1000  # Use avg of last x losses
        avg_loss = int((sum(all_losses[-n:]) / len(all_losses[-n:])) * 1000)
        torch.save(save_ckpt, f"{ckpt_dir}/rlhf_{epoch+1}E_{avg_loss}L_{max_seq_len}MSQ.pth")

    full_val_loss = eval_model(model=model, ref_model=ref_model, criterion=criterion,
                               dataset_loader=dataset_loader, eval_num_samples=dpo_config.eval_num_samples,
                               use_amp=use_amp, full_eval=True, pad_id=pad_id, max_seq_len=max_seq_len,
                               dynamic=dynamic_padding, device=device)

    print("\n" + "=" * 20)
    print(f"Validation Loss: {full_val_loss:.4f}")
    print("=" * 20 + "\n")






