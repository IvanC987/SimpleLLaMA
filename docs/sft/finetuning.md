# Supervised Fine-Tuning Script (`finetune.py`)

## Overview

Supervised Fine-Tuning (SFT) is the first post-pretraining stage in the instruction-tuning pipeline. Its purpose is to teach a pretrained model to follow instructions, adopt roles or personas, and provide structured answers in a format expected by users.

This script implements a **full fine-tuning loop** for this stage, using a JSON dataset and pre-trained weights as the base. In the following sections we'll walk through the `finetune.py` script located in `simple_llama\finetune\full_sft\finetune.py` step by step, covering it in depth.

---

## 1. Configuration

(Most of the setup is very similar to the pretraining script, which was covered in the previous section, and will only be briefly covered here.)

At the start of the script, key imports and configurations are handled:

```python
import os
import time
import random
import math
import torch
import inspect
import numpy as np
from tokenizers import Tokenizer, decoders

from simple_llama.pretraining.llama_transformer import LLaMaTransformer
from simple_llama.pretraining.lr_scheduler import Scheduler
from simple_llama.pretraining.config import TrainingConfig
from simple_llama.finetune.full_sft.sft_config import SFTConfigs
from simple_llama.finetune.json_dataset_loader import JSONDatasetLoader
from simple_llama.finetune.utils import tokenize_and_pad_data, eval_model, sft_prompts
from simple_llama.finetune.format_llm_prompt import format_inference_prompt
```

Next, set a manual seed for **reproducibility**:

```python
seed = 89
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
```

Then specify PyTorch matmul precision and device:

```python
torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"
```

All hyperparameters for SFT are loaded from the `SFTConfigs` dataclass, which mirrors `TrainingConfig` but is adjusted for finetuning:

```python
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
...
```

This includes batch size, gradient accumulation, learning rates, dropout, epochs, and logging/checkpoint paths.

Finally, set up a log file and print out configuration, similar to how pretraining script does it.

---

## 2. Initialization

A `JSONDatasetLoader` object is instantiated to manage train/validation splits:

```python
dataset_loader = JSONDatasetLoader(json_filepath=ft_json_path, batch_size=batch_size, train_split=train_split)
```

This loader is responsible for retrieving `(x, y)` pairs and manages batching and epoch updates.

That's then followed by loading the tokenizer and saved checkpoint:

```python
tokenizer = Tokenizer.from_file(sft_configs.tokenizer_path)
tokenizer.model.unk_token = "<UNK>"  # Set unknown token to <UNK>
tokenizer.decoder = decoders.ByteLevel()  # For byte-level decoding

ckpt = torch.load(sft_configs.model_path, map_location=device)
```

The pad token `<PAD>` is set to maintain consistent masking:

```python
pad_token = "<PAD>"
pad_id = tokenizer.encode(pad_token).ids[0]
criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
```

After that, initialize a model instance with the loaded configuration:

```python
model = LLaMaTransformer(config=training_config, tokenizer=tokenizer, device=device).to(device)
model.train()

model.load_state_dict(ckpt["model_state_dict"], strict=True)

n_params = sum([p.numel() for p in model.parameters()])
print(f"There is {n_params / 1e6:.1f}M parameters in the model")
```

The pretrained weights are loaded directly from the checkpoint, and the total parameter count is displayed for verification.

Next, configure the optimizer and check for fused kernel support (a small speed optimization for GPUs):

```python
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device == "cuda"
extra_args = dict(fused=True) if use_fused else dict()

optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(beta1, beta2), weight_decay=weight_decay, **extra_args)
print(f"Using fused optimizer: {use_fused}\n")

pad_token = "<PAD>"
pad_id = tokenizer.encode(pad_token).ids
assert len(pad_id) == 1, f"{pad_token=} should be a special token with a single value!"
pad_id = pad_id[0]
```

Then, the scheduler and optional model compilation step are set up:

```python
criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

scheduler = Scheduler(torch_optimizer=optimizer,
                      schedule="cosine",
                      training_steps=optimization_steps,
                      warmup_steps=warmup_iterations,
                      max_lr=max_lr,
                      min_lr=min_lr)

if enable_compilation:
    compiled_model = torch.compile(model)
```

The loss function (`CrossEntropyLoss`) is initialized with `ignore_index=pad_id` so that padding tokens do not affect training. The cosine scheduler controls learning rate decay over time. Model compilation (via `torch.compile`) can further accelerate training on supported GPUs.

## 3. Training Loop

Right before entering the training loop, 

```python
start = time.time()
all_losses = []  # Keeping track of all losses
save_ckpt = {}  # Used to save model checkpoint (Holds all state_dicts, hyperparameters, etc.)
norm = float("inf")  # A temp place holder for actual norm
step = 1  # Step here is an approximate since this is now epoch-based rather than token-based iterations

# This autocasts certain parts of the layers (mostly matmul portion) within the model to bf16 for faster training
use_amp = torch.cuda.is_available() and device == "cuda" and torch.cuda.is_bf16_supported()
print(f"Using auto mixed precision: {use_amp}")
```

Here’s what is initialized:

- **Timer (`start`)** — Tracks elapsed training time.  
- **`all_losses`** — Stores raw loss values per iteration for later averaging/plotting.  
- **`save_ckpt`** — A dictionary to collect all model states and optimizer/scheduler info for checkpointing.  
- **`norm`** — Placeholder for gradient norm, updated later each optimization step for stability checks.  
- **`step`** — Tracks global step count, which approximates iteration progress across epochs.  
- **`use_amp`** — Checks if the device supports **automatic mixed precision** (AMP), which speeds up training by casting certain matrix operations to `bfloat16`.


After that setup, the true training loop begins. 

First, 

```python
for epoch in range(epochs):
    current_val_epoch = dataset_loader.val_epoch
    current_train_epoch = dataset_loader.train_epoch

    print("\n" + "=" * 20)
    print(f"Current Epoch: {epoch+1}")
    print("=" * 20 + "\n")
```

We loop `epochs` times, as set by the sft config. Save the current validation and training epoch, and print the current progress.

Next, enter a while loop. While we are within the same training epoch, keep sampling batches from the training set until complete. 

In the first part of the while loop: 

```python
    while dataset_loader.train_epoch == current_train_epoch:
        batch = dataset_loader.get_batch(train=True)

        # Encode and pad the x-y strings
        x, y = tokenize_and_pad_data(batch=batch,
                                     tokenizer=tokenizer,
                                     pad_id=pad_id,
                                     max_seq_len=max_seq_len,
                                     dynamic=dynamic_padding,
                                     device=device
                                     )
```

Grab a batch from the dataset loader, use the `tokenize_and_pad_data` function that was discussed previously to convert the raw strings into x-y tensors.

```python
        with torch.autocast(device_type=device, dtype=torch.bfloat16 if use_amp else torch.float32):
            if enable_compilation:
                pred = compiled_model(x)
            else:
                pred = model(x)

            B, T, C = pred.shape
            loss = criterion(pred.reshape(B * T, C), y.reshape(B * T))
```

Using the `torch.autocast` context manager, pass the input tensor `x` into the model, and compute the loss accordingly. Mixed precision training can significantly speed up computation and reduce memory usage on modern GPUs.

```python
        train_loss_value = loss.item()  # Before normalization for backprop
        loss /= grad_accum_steps  # Normalize loss
        loss.backward()

        all_losses.append(train_loss_value)  # Add the loss

        if step % grad_accum_steps == 0:
            scheduler.step(step // grad_accum_steps)  # Set the lr first
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevents unstable learning
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
```

Save the training loss, normalize it based on `grad_accum_steps` for gradient accumulation, then backpropagate the loss. If we have accumulated gradients across `grad_accum_steps` steps, update the learning rate, clip gradient norm if needed, step the optimizer, and then reset all gradients.

The next section after this is just about intermediate evaluation and checkpoint saving, which is pretty much the same as the part in pretraining with just very minor changes, I will not go over it again here. 

Finally, at the end of each epoch, the model evaluates across the full validation set:

```python
    full_val_loss = eval_model(model=model, criterion=criterion, tokenizer=tokenizer,
                               dataset_loader=dataset_loader, use_amp=use_amp, full_eval=True, pad_id=pad_id,
                               max_seq_len=max_seq_len, dynamic=dynamic_padding, device=device)

    print("\n" + "=" * 20)
    print(f"Validation Loss: {full_val_loss:.4f}")
    print("=" * 20 + "\n")
```

Once the training loop finishes an epoch, we calculate validation loss across all validation examples, print it out, and proceed to the next epoch. This ensures that each epoch ends with a full model performance check rather than just a partial metric.

---

## Summary

The `finetune.py` script completes the SFT stage by transforming the pretrained LLaMA model into an instruction-aligned assistant. While smaller than pretraining, it is conceptually critical: it teaches the model to respond *as* an assistant rather than merely continue text.

In later iterations, **LoRA fine-tuning** can be integrated to reduce compute and memory costs. This version, however, performs full fine-tuning to maximize learning fidelity and alignment clarity for educational demonstration.

**Final Notes:**  
Because SFT is much lighter than pretraining (on smaller scale training), DDP is not used, as the fine-tuning process is fast enough to run efficiently on a single GPU.
