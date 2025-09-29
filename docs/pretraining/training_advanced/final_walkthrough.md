# Training Script Walkthrough

This final piece of documentation in the training guide section provides a sequential walkthrough of the LLM training script, explaining each major section and how everything connects.

---

## Initial Setup and Imports

```python
import os
import time
import random
import math
import inspect
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tokenizers import Tokenizer, decoders

from simple_llama.pretraining.llama_transformer import LLaMaTransformer
from simple_llama.pretraining.dataset_loader import DatasetLoader
from simple_llama.pretraining.lr_scheduler import Scheduler
from simple_llama.pretraining.utils import load_checkpoint, few_shot_prompts, check_log_file_existence
from simple_llama.pretraining.config import TrainingConfig
```

**Key imports:**

- `torch.distributed`: For multi-GPU training support  
- `tokenizers`: Hugging Face tokenizer for text processing  
- Custom modules: Model architecture, data loading, and utilities  

---

## Distributed Training Setup

```python
# To run, use `torchrun --standalone --nproc_per_node=8 train.py`
# Set global variables for DDP
ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
if ddp:
    assert torch.cuda.is_available(), "Should have cuda available if using DDP!"
    init_process_group(backend="nccl")  # Initialize the distributed communication backend
    ddp_rank = int(os.environ["RANK"])
    # Assuming this is single-node multi-GPU setup, so I'm not using local_rank
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:  # Non-Distributed setup. Either CPU or single GPU
    ddp_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Currently using {device=}
")
```

**What this does:**

- Checks if we're running in a distributed environment by looking for RANK and WORLD_SIZE environment variables (Automatically set by `torchrun` when used)
- Initializes the process group with NCCL backend for GPU communication  
- Sets device to the appropriate GPU for each process  
- Designates rank 0 as the `master_process` for logging and checkpointing  

---
## Reproducibility and Performance Settings

```python
# Manual seeding for reproducibility testings
seed = 89
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Setting to 'high' uses TF32 rather than FP32, which makes the training process faster (varies on machines)
# Can set to 'medium' for even faster training, though will be loss in performance
torch.set_float32_matmul_precision("high")
```

Using the same random seeds ensure training is reproducible across runs and TF32 precision provides speedup on NVIDIA Ampere+ GPUs while maintaining accuracy  

---

## Configuration Loading

```python
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
```

**Key configuration values for our 1.3B model:**

- `batch_size = 4` sequences per GPU  
- `max_seq_len = 2048` tokens per sequence  
- `training_tokens = 45,000,000,000` (45B tokens)  
- `grad_accum_steps = 64` (for effective batch size of 524,288 tokens)  

---

## Distributed Training Adjustments

```python
# Need to make sure gradient accumulation step is evenly divisible by # GPUs
assert grad_accum_steps % ddp_world_size == 0, (f"{grad_accum_steps=} % {ddp_world_size=} != 0\n"
                                                f"Please adjust 'tokens_per_update' in config file accordingly!")

grad_accum_steps = grad_accum_steps // ddp_world_size

# Do the same for eval interval
assert eval_interval % ddp_world_size == 0, (f"{eval_interval=} % {ddp_world_size=} != 0\n"
                                             f"Please adjust 'eval_interval' in config file accordingly!")

eval_interval = eval_interval // ddp_world_size
```


These adjustments are needed in DDP because since each GPU accumulates gradients independently, we need to ensure all GPUs perform the same number of accumulation steps and evaluation intervals must be synchronized across processes.  

Note the `grad_accum_steps` update. 
If `ddp_world_size` is 1, meaning single GPU training, then `grad_accum_steps` remains the same. However, if `ddp_world_size` is 8, meaning training is being parallized across 8 GPUs, then `grad_accum_steps` would be reduced by 1/8.  

For the remainder of this walkthrough, we'll assume `ddp_world_size=8` and `grad_accum_steps=8`

---

## Logging Setup

```python
if master_process:  # Check if log_file already exists and deal with it accordingly
    log_file = check_log_file_existence(log_file, ddp)

if master_process:
    with open(log_file, "a") as f:
        columns = ["step", "progress (%)", "Training Loss", "Perplexity", "Learning Rate", "L2 Norm",
                   "Tokens Processed (Current- In Millions)", "Tokens Processed (Total- In Millions)",
                   "Tokens Per Second", "Time Per Evaluation"]
        f.write(",".join(columns))
        f.write("\n")
```

**Logging strategy:**

- Only the master process handles file I/O to avoid conflicts  
- CSV format for easy analysis and plotting  
- Various metrics to monitor training progress  

---

## Training Calculations

```python
tokens_per_step = batch_size * max_seq_len * ddp_world_size
tokens_per_opt_step = tokens_per_step * grad_accum_steps   # How many tokens to process before optimization step
train_iterations = int(training_tokens // tokens_per_step)
optimization_steps = train_iterations // grad_accum_steps  # Number of times to step the optimizer

ckpt_dir = config.ckpt_dir
os.makedirs(ckpt_dir, exist_ok=True)
```

**Example calculations for 8 GPUs:**

- `tokens_per_step = 4 × 2048 × 8 = 65,536 tokens/step`  
- `tokens_per_opt_step = 65,536 × 8 = 524,288 tokens/optimizer_step`  
- `train_iterations = 45,000,000,000 ÷ 65,536 ≈ 686,645 steps`  
- `optimization_steps = 686,645 ÷ 8 = 85,830 optimizer steps`  

Optimization steps is divided by `grad_accum_steps` because we only step the optimizer (update parameter) after each round of gradient accumulations. 

---

## Model and Data Initialization

```python
# Instantiate dataset_loader obj
bytes_per_token = 2  # 2 byte per token (Assuming using uint16)
dataset_loader = DatasetLoader(batch=batch_size, seq_len=max_seq_len, process_rank=ddp_rank,
                               num_processes=ddp_world_size, dataset_dir=config.dataset_dir, device=device)
if master_process:
    dataset_loader.print_ds_info(bytes_per_token=bytes_per_token)
    print(f"{dataset_loader.file_idx=}")
    print(f"{dataset_loader.tok_idx=}")


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

```

The dataset loader handles sharding across multiple GPUs and streams data from disk to handle large datasets. 

The 1.3B param model is initialized primarily with: 

- 24 transformer layers  
- 2048 embedding dimension  
- 32 attention heads (64-dim each)  
- RoPE positional embeddings  
- SwiGLU activation functions  

---

## Optimizer and Scheduler Setup

```python
# Betas, weight decay, and scheduler follows the LLaMa paper, with the exception of the learning rate
# Used fused operations if available, from Dr. Karpathy's video
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and (device == "cuda" or ddp)
extra_args = dict(fused=True) if use_fused else dict()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(beta1, beta2), weight_decay=weight_decay, **extra_args)
if master_process:
    print(f"Using fused optimizer: {use_fused}\n")

# Instantiating CE Loss and scheduler
criterion = torch.nn.CrossEntropyLoss()
scheduler = Scheduler(torch_optimizer=optimizer,
                      schedule="cosine",
                      training_steps=optimization_steps,
                      warmup_steps=warmup_iterations,
                      max_lr=max_lr,
                      min_lr=min_lr)
```

**In this section:**

- Check if AdamW supports fused kernels for better performance before instantiation
- Create the Criterion (using CrossEntropyLoss which is typical when training LLMs)
- Create a (custom) scheduler based on cosine learning rate schedule with warmup  

---
## Checkpoint Loading

```python
# prev_tok_trained would be how many tokens the model has already been trained (for loading in models, if applicable)
prev_tok_trained = 0

# Loading in checkpoint to resume training if needed
if load_ckpt:
    ckpt_dict, prev_tok_trained = load_checkpoint(config=config, ckpt_dir=ckpt_dir, ddp=ddp, master_process=master_process)

    model.load_state_dict(ckpt_dict["model_state_dict"])
    optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])

    # Manually check the scheduler here, T_max and eta_min should match, if not, can lead to undefined behaviors
    if use_prev_scheduler:
        assert ckpt_dict["max_lr"] == max_lr
        assert ckpt_dict["min_lr"] == min_lr
        assert ckpt_dict["train_iterations"] == train_iterations
        scheduler.load_state_dict(ckpt_dict["scheduler_state_dict"])

        dataset_loader.file_idx = ckpt_dict["file_idx"]
        dataset_loader.tok_idx = ckpt_dict["tok_idx"]
        dataset_loader.file_data = np.load(dataset_loader.filepaths[dataset_loader.file_idx])
```

For checkpoint restoration, this would need to load in the model and optimizer state dicts, and if desired, continue exactly from where the previous run left off at. 

---

## Model Compilation and DDP Wrapping

```python
# Compiling the model via torch.compile reduces the training time
# Though may not be compatible with certain GPUs. If so, turn "compile_model" in config to False
if enable_compilation and ddp:
    # Interestingly enough, DDP docs recommends applying ddp wrapper before compiling
    # Karpathy's implementation is the other way around, compile -> ddp wrapper
    model_handle = torch.compile(DDP(model, device_ids=[ddp_rank]))
elif enable_compilation and not ddp:
    model_handle = torch.compile(model)
elif ddp:
    model_handle = DDP(model, device_ids=[ddp_rank])
else:
    model_handle = model  # Plain case, not recommended for actual usage
```

Notice that no matter if we compile, apply DDP, do both or do none, the resulting model will be called `model_handle`. 
That's because when we need to checkpoint the model, we need the underlying model itself, not the wrapped DDP/Compiled version, and so this deals with separation of concerns.

**Important note about compilation order:**

- Current code uses `torch.compile(DDP(model))` which follows DDP documentation  
- Some implementations use `DDP(torch.compile(model))` — both have tradeoffs  

---

## Training Loop Initialization

```python
total_tok_trained = 0  # Keeping track of total current tokens that has been processed
next_token_ckpt = token_ckpt

eos_token = tokenizer.encode("<EOS>").ids[0]
start = time.time()
all_losses = []  # Keeping track of all losses
save_ckpt = {}  # Used to save model checkpoint (Holds all state_dicts, hyperparameters, etc.)
norm = float("inf")  # A temp placeholder for actual norm

# This autocasts certain parts of the layers (mostly matmul portion) within the model to bf16 for faster training
use_amp = torch.cuda.is_available() and (device == "cuda" or ddp) and torch.cuda.is_bf16_supported()
if master_process:
    print(f"Using auto mixed precision: {use_amp}")
```

**Tracking variables:**

- `total_tok_trained`: Counts tokens processed in current run  
- `next_token_ckpt`: Token count for next checkpoint save  
- `all_losses`: History for checkpoint naming and analysis  

---

## Main Training Loop

```python
for step in range(1, train_iterations+1):
    x, y = dataset_loader.get_batch()

    with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16 if use_amp else torch.float32):
        pred = model_handle(x)
        B, T, C = pred.shape
        loss = criterion(pred.reshape(B * T, C), y.reshape(B * T))
```

Each iteration begins by fetching a batch of input and target sequences, here shaped `(4, 2048)`, based on the configuration. 
The forward pass is run inside a `torch.autocast` context, which enables mixed-precision execution (BF16 where available) to improve speed and memory efficiency. 
The model outputs predictions of shape `(B, T, C)`, which are then compared against the targets using cross-entropy loss. This loss measures how well the model’s predicted distributions align with the true next tokens across all sequence positions.

---

## Gradient Accumulation and Backward Pass

```python
    train_loss_value = loss.item()
    loss /= grad_accum_steps

    if ddp:
        model_handle.require_backward_grad_sync = (step % grad_accum_steps == 0)
    loss.backward()

    total_tok_trained += tokens_per_step
    all_losses.append(train_loss_value)
```

The computed loss is divided by the number of accumulation steps so that gradients average correctly across multiple smaller batches. 
In distributed setups, gradient synchronization is deferred until the end of an accumulation cycle (`step % grad_accum_steps == 0`) to reduce communication overhead. 
The backward pass then contributes gradients to parameters, while counters track total tokens processed and log raw loss values. This strategy allows training with effectively large batch sizes even on limited GPU memory, while keeping updates consistent across devices.

---

## Optimizer Step

```python
    if step % grad_accum_steps == 0:
        scheduler.step(step // grad_accum_steps)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

**Optimizer step details:**

- Scheduler steps on optimizer steps, not training steps  
- Gradient clipping at 1.0 prevents explosion  
- `set_to_none=True` is more memory efficient than zeroing  

---

## Checkpoint Saving

```python
    if (total_tok_trained > next_token_ckpt or step == train_iterations) and master_process:
        next_token_ckpt += token_ckpt

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

        n = 2500
        avg_loss = int((sum(all_losses[-n:]) / len(all_losses[-n:])) * 1000)
        combined_tokens = total_tok_trained + prev_tok_trained
        if combined_tokens < 1e10:
            torch.save(save_ckpt, f"{ckpt_dir}/model_{int(combined_tokens / 1e6)}M_{avg_loss}L_{max_seq_len}MSQ.pth")
        else:
            torch.save(save_ckpt, f"{ckpt_dir}/model_{int(combined_tokens / 1e9)}B_{avg_loss}L_{max_seq_len}MSQ.pth")
```

At every `token_ckpt` token interval (or at the very last step of the training run), save a copy of the state at that point. 
Then calculate the average loss in the past `n` steps which would be used to name the checkpoint file in conjunction with training tokens.


---

## Evaluation and Logging

```python
    if step % eval_interval == 0 and master_process:
        if torch.cuda.is_available() and device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]
        tokens_processed = int(total_tok_trained // 1e6)

        with open(log_file, "a") as f:
            write_data = [step, round((step / train_iterations) * 100, 2), round(train_loss_value, 4), 
                         round(math.e ** train_loss_value, 2), round(current_lr, 4), round(norm.item(), 4),
                         tokens_processed, int(prev_tok_trained // 1e6) + tokens_processed,
                         int((eval_interval * tokens_per_step) // elapsed), int(elapsed)]
            f.write(",".join([str(wd) for wd in write_data]))
            f.write("\n")

        print("----------------")
        print(f"Step: {step} steps   |   Training Progress: {(step / train_iterations) * 100:.2f}%   |   "
              f"Training Loss: {train_loss_value:.4f}   |   Perplexity: {math.e ** train_loss_value:.2f}   |   "
              f"Learning Rate: {current_lr:.5f}   |   Norm: {norm.item():.4f}   |   "
              f"Tokens Processed: {tokens_processed}M ({int(prev_tok_trained // 1e6) + tokens_processed}M)   |   "
              f"tok/s: {int((eval_interval * tokens_per_step) // elapsed)}   |   Time: {int(elapsed)}s")

        start = time.time()
```


At regular intervals, the training script logs key metrics to both console and file. 
These include training progress, loss, and perplexity (computed as `exp(loss)` for easier interpretation), along with learning rate, gradient norm, and tokens processed. 
Token throughput (`tok/s`) is also tracked to measure efficiency. 
Synchronizing CUDA before timing ensures accurate elapsed measurements, making these logs a reliable snapshot of both training stability and performance.

---

## Text Generation Samples

```python
        if 'next_gen_step' not in locals():
            next_gen_step = step

        if step >= next_gen_step:
            print("\n")
            print(model.generate(random.choice(few_shot_prompts), 64, 1.0, 0.8, eos_token=eos_token))
            next_gen_step = int(step * model_gen_multiplier)
            print("\n")
            print(f"Sampled generation at {step=}, next at {next_gen_step=}")

        print("----------------")
```


In addition to numeric metrics, the model is periodically prompted to generate text from a random few-shot example. 
The interval between generations grows exponentially and the check helps confirm that the model is learning to produce structured, human-like outputs.

---

## Cleanup

```python
if ddp:
    destroy_process_group()
```

**Final step:**

- Properly shuts down distributed process group  
- Ensures clean exit and resource release  
