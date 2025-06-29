import os
import time
import random
import math
import torch
import inspect
import numpy as np
from tokenizers import Tokenizer, decoders

from simple_llama.pretraining.llama_transformer import LLaMaTransformer
from simple_llama.pretraining.dataset_loader import DatasetLoader
from simple_llama.pretraining.lr_scheduler import Scheduler
from simple_llama.pretraining.utils import load_checkpoint, few_shot_prompts
from simple_llama.pretraining.config import TrainingConfig




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


# Hyperparameters
# --------------------------------------
config = TrainingConfig()

# Unpack values from config for convenience
enable_compilation = config.enable_compilation

batch_size = config.batch_size
max_seq_len = config.max_seq_len

eval_interval = config.eval_interval
training_tokens = config.training_tokens

warmup_iterations = config.warmup_iterations
max_lr = config.max_lr
min_lr = config.min_lr
beta1 = config.beta1
beta2 = config.beta2
weight_decay = config.weight_decay

grad_accum_steps = config.grad_accum_steps
load_ckpt = config.load_ckpt
token_ckpt = config.token_ckpt
use_prev_scheduler = config.use_prev_scheduler

log_file = config.log_file
model_gen_multiplier = config.model_gen_multiplier

eval_interval *= grad_accum_steps  # So evaluate model after eval_interval number of gradient updates
# --------------------------------------


if os.path.exists(log_file):
    print(f"Log file '{log_file}' already exists.")
    print("Options:")
    print("1. Delete the existing log file")
    print("2. Use a new log file name")
    print("3. Exit")

    user = input("Enter choice (1/2/3): ").strip()

    if user == "1":
        confirm = input("Confirm deletion of file? [Y/N]: ")
        if confirm.lower() != "y":
            print("Now exiting")
            exit()
        os.remove(log_file)
        print("Deleted original log file.")
    elif user == "2":
        new_name = input("Enter new log file name: ").strip()
        if not new_name:
            print("Invalid filename. Exiting.")
            exit()
        else:
            log_file = new_name + ".txt" if not new_name.endswith(".txt") else new_name
            print(f"Now using new log file: {log_file}")
    else:
        print("Exiting") if user == "3" else print("Invalid response. Now exiting.")
        exit()


    print("\n\n")


with open(log_file, "a") as f:
    columns = ["step", "progress (%)", "Training Loss", "Perplexity", "Learning Rate", "L2 Norm",
               "Tokens Processed (Current- In Millions)", "Tokens Processed (Total- In Millions)",
               "Tokens Per Second", "Time Per Evaluation"]
    f.write(",".join(columns))
    f.write("\n")


tokens_per_step = batch_size * max_seq_len
tokens_per_opt_step = tokens_per_step * grad_accum_steps   # How many tokens to process before optimization step
train_iterations = int(training_tokens // tokens_per_step)
optimization_steps = train_iterations // grad_accum_steps  # Number of times to step the optimizer

ckpt_dir = config.ckpt_dir
os.makedirs(ckpt_dir, exist_ok=True)


print(f"Model will be trained on training_tokens={training_tokens / 1e9:.2f} Billion Tokens")

print("\nTraining Configurations:")
print("=" * 30)
for field_name in config.__dataclass_fields__:
    value = getattr(config, field_name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        print(f"{field_name}: {value:_}")
    else:
        print(f"{field_name}: {value}")
print("=" * 20)
print("")
print(f"{tokens_per_step=}")
print(f"{tokens_per_opt_step=:_}")
print(f"{train_iterations=:_}")
print("\n")


# Initializing/Loading objects
# ------------------------

# Instantiate dataset_loader obj
bytes_per_token = 2  # 2 byte per token (Assuming using uint16)
dataset_loader = DatasetLoader(batch=batch_size, seq_len=max_seq_len, dataset_dir=config.dataset_dir, device=device)
dataset_loader.print_ds_info(bytes_per_token=bytes_per_token)
print("\n\n")


# Load in pretrained tokenizer
tokenizer = Tokenizer.from_file(config.tokenizer_path)
tokenizer.model.unk_token = "<UNK>"  # Set unknown token to <UNK>
tokenizer.decoder = decoders.ByteLevel()  # For byte-level decoding


# Create model
model = LLaMaTransformer(
    config=config,
    tokenizer=tokenizer,
    device=device,
).to(device)
model.train()


# Betas, weight decay, and scheduler follows the LLaMa paper, with the exception of the learning rate
# Used fused operations if available, from Dr. Karpathy's video
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device == "cuda"
extra_args = dict(fused=True) if use_fused else dict()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(beta1, beta2), weight_decay=weight_decay, **extra_args)
print(f"Using fused optimizer: {use_fused}\n")


# Instantiating CE Loss and scheduler
# T_max may have a slight deviation if using warmup_steps, but overall it's fine as long as T_max >> warmup_steps
criterion = torch.nn.CrossEntropyLoss()
# scheduler = CosineAnnealingLR(optimizer, T_max=train_iterations, eta_min=min_lr)
scheduler = Scheduler(torch_optimizer=optimizer,
                      schedule="cosine",
                      training_steps=optimization_steps,
                      warmup_steps=warmup_iterations,
                      max_lr=max_lr,
                      min_lr=min_lr)

# prev_tok_trained would be how many tokens the model has already been trained (for loading in models, if applicable)
prev_tok_trained = 0

print(f"{dataset_loader.file_idx=}")
print(f"{dataset_loader.tok_idx=}")
# Loading in checkpoint to resume training if needed
if load_ckpt:
    ckpt_dict, prev_tok_trained = load_checkpoint(config=config, ckpt_dir=ckpt_dir)

    model.load_state_dict(ckpt_dict["model_state_dict"])
    optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])

    # Manually check the scheduler here, T_max and eta_min should match, if not, can lead to undefined behaviors
    # Since it's a bit static, there's a boolean toggle in config.py, useful if scheduler needs to be changed.
    if use_prev_scheduler:
        assert ckpt_dict["max_lr"] == max_lr
        assert ckpt_dict["min_lr"] == min_lr
        assert ckpt_dict["train_iterations"] == train_iterations
        scheduler.load_state_dict(ckpt_dict["scheduler_state_dict"])

    dataset_loader.file_idx = ckpt_dict["file_idx"]
    dataset_loader.tok_idx = ckpt_dict["tok_idx"]
    dataset_loader.file_data = np.load(dataset_loader.filepaths[dataset_loader.file_idx])

    print(f"{dataset_loader.file_idx=}")
    print(f"{dataset_loader.tok_idx=}")


# Compiling the model via torch.compile reduces the training time
# Though may not be compatible with certain GPUs. If so, turn "compile_model" in config to False
if enable_compilation:
    compiled_model = torch.compile(model)


print("\n")
n_params = sum([p.numel() for p in model.parameters()])
print(f"There is {n_params / 1e6:.1f}M parameters in the model")


# Training Loop
# ------------------------

total_tok_trained = 0  # Keeping track of total current tokens that has been processed
# Used to check if to save model.
# Ex. At the start, token_ckpt is 50M. So when total_tok_train--TTT-- >= 50M, save model/optimizer state dict
# Then next_token_ckpt += token_ckpt, which would now be 100M. Once TTT >= next_token_ckpt, save and increment again
next_token_ckpt = token_ckpt


eos_token = tokenizer.encode("<EOS>").ids
start = time.time()
all_losses = []  # Keeping track of all losses
save_ckpt = {}  # Used to save model checkpoint (Holds all state_dicts, hyperparameters, etc.)
norm = float("inf")  # A temp place holder for actual norm

# This autocasts certain parts of the layers (mostly matmul portion) within the model to bf16 for faster training
use_amp = torch.cuda.is_available() and device == "cuda" and torch.cuda.is_bf16_supported()
print(f"Using auto mixed precision: {use_amp}")

for step in range(1, train_iterations+1):
    x, y = dataset_loader.get_batch()

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

    # Increment number of tokens processed
    total_tok_trained += tokens_per_step

    all_losses.append(train_loss_value)  # Add the loss

    if step % grad_accum_steps == 0:
        scheduler.step(step // grad_accum_steps)  # Set the lr first
        # Clip gradients, step optimizer, and set to None
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevents unstable learning
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


    # Save checkpoint
    if total_tok_trained > next_token_ckpt or step == train_iterations:
        next_token_ckpt += token_ckpt

        # Save checkpoints
        save_ckpt["config"] = config

        save_ckpt["model_state_dict"] = model.state_dict()
        save_ckpt["optimizer_state_dict"] = optimizer.state_dict()
        save_ckpt["scheduler_state_dict"] = scheduler.state_dict()

        save_ckpt["max_lr"] = max_lr
        save_ckpt["min_lr"] = min_lr
        save_ckpt["train_iterations"] = train_iterations
        save_ckpt["total_tok_trained"] = total_tok_trained + prev_tok_trained

        save_ckpt["file_idx"] = dataset_loader.file_idx
        save_ckpt["tok_idx"] = dataset_loader.tok_idx

        # Save and name based on tokens processed and loss
        n = 50  # Use avg of last x losses
        avg_loss = int((sum(all_losses[-n:]) / len(all_losses[-n:])) * 1000)
        combined_tokens = total_tok_trained + prev_tok_trained
        if combined_tokens < 1e10:
            torch.save(save_ckpt, f"{ckpt_dir}/model_{int(combined_tokens / 1e6)}M_{avg_loss}L_{max_seq_len}MSQ.pth")
        else:
            torch.save(save_ckpt, f"{ckpt_dir}/model_{int(combined_tokens / 1e9)}B_{avg_loss}L_{max_seq_len}MSQ.pth")


    # Save/print metrics while evaluating model prompt generation ability
    # No need for val loss, due to how dataset is created
    if step % eval_interval == 0:
        if torch.cuda.is_available() and device == "cuda":
            torch.cuda.synchronize()  # More accurate measurement for `elapsed` since cuda kernels are async

        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]
        tokens_processed = int(total_tok_trained // 1e6)

        with open(log_file, "a") as f:
            write_data = [step,  # Current step
                          round((step / train_iterations) * 100, 2),  # Progress
                          round(train_loss_value, 4),  # Training loss value
                          round(math.e ** train_loss_value, 2),  # Perplexity
                          round(current_lr, 4),  # Learning Rate
                          round(norm.item(), 4),  # L2 norm of the gradients
                          tokens_processed,  # Tokens processed, current run
                          int(prev_tok_trained // 1e6) + tokens_processed,  # Tokens processed, total
                          int((eval_interval * tokens_per_step) // elapsed),  # Tokens per second
                          int(elapsed),  # Time taken
                          ]

            write_data = [str(wd) for wd in write_data]
            f.write(",".join(write_data))
            f.write("\n")

        print("----------------")
        print(f"Step: {step} steps   |   "
              f"Training Progress: {(step / train_iterations) * 100:.2f}%   |   "
              f"Training Loss: {train_loss_value:.4f}   |   "
              f"Perplexity: {math.e ** train_loss_value:.2f}   |   "
              f"Learning Rate: {current_lr:.5f}   |   "
              f"Norm: {norm.item():.4f}   |   "
              f"Tokens Processed: {tokens_processed}M ({int(prev_tok_trained // 1e6) + tokens_processed}M)   |   "
              f"tok/s: {int((eval_interval * tokens_per_step) // elapsed)}   |   "
              f"Time: {int(elapsed)}s")

        start = time.time()

        # Setting max_new_tokens to a lower value does increase overall training speed by a noticeable amount (in my case)
        # Exponential generation interval logic
        if 'next_gen_step' not in locals():
            next_gen_step = step  # initialize on first use

        if step >= next_gen_step:
            print("\n")
            print(model.generate(random.choice(few_shot_prompts), 64, 1.0, 0.8, eos_token=eos_token))
            next_gen_step = int(next_gen_step * model_gen_multiplier)  # exponential growth in generation spacing
            print("\n")
            print(f"Sampled generation at {step=}, next at {next_gen_step=}")

        print("----------------")

