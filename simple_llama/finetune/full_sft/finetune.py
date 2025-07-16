import os
import time
import random
import math
import torch
import inspect
import numpy as np
from tokenizers import Tokenizer, decoders

from simple_llama.pretraining.llama_transformer import LLaMaTransformer, LoRAInjection
from simple_llama.pretraining.lr_scheduler import Scheduler
from simple_llama.pretraining.config import TrainingConfig
from simple_llama.pretraining.utils import check_log_file_existence
from simple_llama.finetune.full_sft.sft_config import SFTConfigs
from simple_llama.finetune.json_dataset_loader import JSONDatasetLoader
from simple_llama.finetune.utils import tokenize_and_pad_data, eval_model, sft_prompts
from simple_llama.finetune.format_llm_prompt import format_inference_prompt


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
print(f"Currently using {device=}\n")


# Get both configs and assert they match, for crucial hyperparameters (E.g. n_embd, n_layers, etc.)
sft_configs = SFTConfigs()


# Hyperparameters
# --------------------------------------
ft_json_path = sft_configs.ft_json_path

enable_compilation = sft_configs.enable_compilation

batch_size = sft_configs.batch_size

eval_interval = sft_configs.eval_interval

warmup_iterations = sft_configs.warmup_iterations
max_lr = sft_configs.max_lr
min_lr = sft_configs.min_lr
beta1 = sft_configs.beta1
beta2 = sft_configs.beta2
weight_decay = sft_configs.weight_decay
train_split = sft_configs.train_split

grad_accum_steps = sft_configs.grad_accum_steps
epochs = sft_configs.epochs
ckpt_epochs = sft_configs.ckpt_epochs

log_file = sft_configs.log_file
model_gen_multiplier = sft_configs.model_gen_multiplier

dynamic_padding = sft_configs.dynamic_padding
eval_interval *= grad_accum_steps  # So evaluate model after eval_interval number of gradient updates
# --------------------------------------


log_file = check_log_file_existence(log_file=log_file, ddp=False)


with open(log_file, "a") as f:
    columns = ["step", "progress (%)", "Training Loss", "Perplexity (Train)", "Validation Loss", "Perplexity (Val)",
               "Learning Rate", "L2 Norm", "Time Per Evaluation"]
    f.write(",".join(columns))
    f.write("\n")

ckpt_dir = sft_configs.ckpt_dir
os.makedirs(ckpt_dir, exist_ok=True)


print("\nTraining Configurations:")
print("=" * 30)
for field_name in sft_configs.__dataclass_fields__:
    value = getattr(sft_configs, field_name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        print(f"{field_name}: {value:_}")
    else:
        print(f"{field_name}: {value}")
print("=" * 20)
print("\n")


# Initializing/Loading objects
# ------------------------

# Instantiate dataset_loader obj
dataset_loader = JSONDatasetLoader(json_filepath=ft_json_path, batch_size=batch_size, train_split=train_split)

# Load in pretrained tokenizer
tokenizer = Tokenizer.from_file(sft_configs.tokenizer_path)
tokenizer.model.unk_token = "<UNK>"  # Set unknown token to <UNK>
tokenizer.decoder = decoders.ByteLevel()  # For byte-level decoding


train_iterations = (len(dataset_loader.train_dataset)//batch_size) * epochs
optimization_steps = train_iterations // grad_accum_steps  # Number of times to step the optimizer

print(f"{train_iterations=}")
print(f"{optimization_steps=}")


ckpt = torch.load(sft_configs.model_path, map_location=device)
training_config: TrainingConfig = ckpt["config"]

# Update/Load configs accordingly
# ---------------------------------------------
max_seq_len = training_config.max_seq_len

training_config.dropout = sft_configs.dropout
training_config.use_flash_attention = sft_configs.use_flash_attention
# training_config.enable_compilation = sft_configs.enable_compilation
# ---------------------------------------------

model = LLaMaTransformer(config=training_config, tokenizer=tokenizer, device=device).to(device)
model.train()

model.load_state_dict(ckpt["model_state_dict"], strict=True)  # Should strictly match


n_params = sum([p.numel() for p in model.parameters()])
print(f"There is {n_params / 1e6:.1f}M parameters in the model")
print("\n")



fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device == "cuda"
extra_args = dict(fused=True) if use_fused else dict()

optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(beta1, beta2), weight_decay=weight_decay, **extra_args)
print(f"Using fused optimizer: {use_fused}\n")


pad_token = "<PAD>"
pad_id = tokenizer.encode(pad_token).ids
assert len(pad_id) == 1, f"{pad_token=} should be a special token with a single value!"
pad_id = pad_id[0]


criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

scheduler = Scheduler(torch_optimizer=optimizer,
                      schedule="cosine",
                      training_steps=optimization_steps,
                      warmup_steps=warmup_iterations,
                      max_lr=max_lr,
                      min_lr=min_lr)

# Compiling the model via torch.compile reduces the training time
# Though may not be compatible with certain GPUs. If so, turn "compile_model" in config to False
if enable_compilation:
    compiled_model = torch.compile(model)


# Training Loop
# ------------------------
start = time.time()
all_losses = []  # Keeping track of all losses
save_ckpt = {}  # Used to save model checkpoint (Holds all state_dicts, hyperparameters, etc.)
norm = float("inf")  # A temp place holder for actual norm
step = 1  # Step here is an approximate since this is now epoch-based rather than token-based iterations

# This autocasts certain parts of the layers (mostly matmul portion) within the model to bf16 for faster training
use_amp = torch.cuda.is_available() and device == "cuda" and torch.cuda.is_bf16_supported()
print(f"Using auto mixed precision: {use_amp}")

for epoch in range(epochs):
    current_val_epoch = dataset_loader.val_epoch
    current_train_epoch = dataset_loader.train_epoch

    print("\n" + "=" * 20)
    print(f"Current Epoch: {epoch+1}")
    print("=" * 20 + "\n")

    while dataset_loader.train_epoch == current_train_epoch:
        # batch is a list of tuples, each with 2 strings being (x, y) respectively
        # y is guaranteed by the 'format_training_prompt' function in 'format_llm_prompt.py' to be the ending suffix of x
        # E.g.
        # x:  "<SOT>Be concise.<EOT>\n\n<SOU>What's 2+2?<EOU>\n<SOA>4<EOA>",
        # y:  "4<EOA>"
        batch = dataset_loader.get_batch(train=True)

        # Encode and pad the x-y strings
        x, y = tokenize_and_pad_data(batch=batch,
                                     tokenizer=tokenizer,
                                     pad_id=pad_id,
                                     max_seq_len=max_seq_len,
                                     dynamic=dynamic_padding,
                                     device=device
                                     )

        with torch.autocast(device_type=device, dtype=torch.bfloat16 if use_amp else torch.float32):
            if enable_compilation:
                pred = compiled_model(x)
            else:
                pred = model(x)

            B, T, C = pred.shape
            loss = criterion(pred.reshape(B * T, C), y.reshape(B * T))


        train_loss_value = loss.item()  # Before normalization for backprop
        loss /= grad_accum_steps  # Normalize loss
        loss.backward()

        all_losses.append(train_loss_value)  # Add the loss

        if step % grad_accum_steps == 0:
            scheduler.step(step // grad_accum_steps)  # Set the lr first
            # Clip gradients, step optimizer, and set to None
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevents unstable learning
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


        # Save/print metrics while evaluating model prompt generation ability
        # No need for val loss, due to how dataset is created
        if step % eval_interval == 0:
            # Eval model, use max_seq_len for first evaluation
            single_val_loss = eval_model(model=model, criterion=criterion, tokenizer=tokenizer,
                                         dataset_loader=dataset_loader, use_amp=use_amp, full_eval=False, pad_id=pad_id,
                                         max_seq_len=max_seq_len, dynamic=dynamic_padding, device=device)

            if torch.cuda.is_available() and device == "cuda":
                torch.cuda.synchronize()  # More accurate measurement for `elapsed` since cuda kernels are async

            elapsed = time.time() - start
            current_lr = optimizer.param_groups[0]["lr"]

            with open(log_file, "a") as f:
                write_data = [step,  # Current step
                              round((step / train_iterations) * 100, 2),  # Progress
                              round(train_loss_value, 4),  # Training loss value
                              round(math.e ** train_loss_value, 2),  # Train Perplexity
                              round(single_val_loss, 4),  # Validation Loss (Single Batch)
                              round(math.e ** single_val_loss, 2),  # Val Perplexity
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
                  f"Perplexity (Training): {math.e ** train_loss_value:.2f}   |   "
                  f"Validation Loss (Single Example): {single_val_loss:.4f}   |   "
                  f"Perplexity (Validation): {math.e ** single_val_loss:.2f}   |   "
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
                rand_prompt = random.choice(sft_prompts)
                formatted_prompt = format_inference_prompt(user=rand_prompt["User"], assistant=rand_prompt["Assistant"], template=rand_prompt["Template"][0])
                print(model.generate(formatted_prompt, 128, 1.0, 0.8, eos_token=tokenizer.encode("<EOA>").ids))
                next_gen_step = int(next_gen_step * model_gen_multiplier)  # exponential growth in generation spacing
                print("\n")
                print(f"Sampled generation at {step=}, next at {next_gen_step=}")

            print("----------------")

        step += 1

    # Save Checkpoint/Final Weights
    if epoch % ckpt_epochs == 0 or epoch == epochs-1:
        # Since LoRA is designed to be extremely cheap, I don't think there's a need to checkpoint (In terms of resuming training)
        # This is only done to eval model performance over certain epochs

        save_ckpt["model_state_dict"] = model.state_dict()
        save_ckpt["config"] = training_config  # This is mostly used for model model creation when loading in state dict

        # Save and name based on tokens processed and loss
        n = 50  # Use avg of last x losses
        avg_loss = int((sum(all_losses[-n:]) / len(all_losses[-n:])) * 1000)
        torch.save(save_ckpt, f"{ckpt_dir}/sft_{epoch+1}E_{avg_loss}L_{max_seq_len}MSQ.pth")

    # Evaluate the full validation epoch
    full_val_loss = eval_model(model=model, criterion=criterion, tokenizer=tokenizer,
                               dataset_loader=dataset_loader, use_amp=use_amp, full_eval=True, pad_id=pad_id,
                               max_seq_len=max_seq_len, dynamic=dynamic_padding, device=device)

    print("\n" + "=" * 20)
    print(f"Validation Loss: {full_val_loss:.4f}")
    print("=" * 20 + "\n")


