# Distributed Data Parallel

## Introduction

Training modern large language models requires immense computational resources. Even for a small LLM like the one trained in this project, with 1.3B parameters, training on 50B tokens would take many tens of days on a single GPU. Scaling upwards on larger model and bigger dataset sizes, it would become infeasible, and so we need to find a way to expand to multiple GPU usage. 

**Distributed Data Parallel (DDP)** is PyTorch's solution for multi-GPU training that allows you to:  

- Distribute the training workload across multiple GPUs
- Scale to much larger batch sizes
- Reduce training time significantly
- Utilize expensive hardware efficiently

In this guide, we'll explore how DDP works and how it's implemented in the LLM training script.

---

## How DDP Works: The Core Concepts

### 1. Process-Based Parallelism

DDP uses a multi-process approach where each GPU runs its own independent process with a complete copy of the model:

```
Process 0 (GPU 0)   Process 1 (GPU 1)   Process 2 (GPU 2)
     ↓                   ↓                   ↓
 Model Copy 0       Model Copy 1         Model Copy 2
     ↓                   ↓                   ↓
 Data Shard 0       Data Shard 1         Data Shard 2
```

Here, each process has it's own model replica. In each forward pass, they would get their own subset of data, compute gradients independently, and synchronize gradients across all processes.

### 2. The Synchronization Process

The key to DDP is gradient synchronization. Here's what happens each training step:

1. **Forward Pass**: Each GPU processes its mini-batch independently  
2. **Backward Pass**: Each GPU computes gradients for its portion  
3. **All-Reduce**: Gradients are averaged across all GPUs  
4. **Optimizer Step**: Each GPU updates its model weights identically  

This ensures all model replicas stay synchronized throughout training.

---

## DDP Implementation in Our Training Code

### Initialization and Setup

Let's examine how DDP is initialized in the script:

```python
# DDP Initialization
ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
if ddp:
    assert torch.cuda.is_available(), "Should have cuda available if using DDP!"
    init_process_group(backend="nccl")  # Initialize the distributed communication backend
    ddp_rank = int(os.environ["RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:  # Non-Distributed setup
    ddp_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
```

**Key Components:**  

- **RANK**: Unique identifier for each process (0, 1, 2, ...)  
- **WORLD_SIZE**: Total number of processes (equal to number of GPUs)  
- **backend="nccl"**: NVIDIA Collective Communications Library - optimized for GPU-to-GPU communication  
- **Master Process**: This is the Rank 0 process that handles logging, checkpointing, and other major functionalities

Small note about Master Process: 
When training on a single GPU, that process is the only process running, which is the master process. However when dealing with multiple GPUs, we would need to choose a single process to serve as the 'master process'. Otherwise, when we save model checkpoints, loggings, and such later on, it will be duplicated many times. 
By convention, master process is the process with a Rank value of 0. 

### Launching DDP Training

DDP requires a specific launch command that sets up the environment variables:

```bash
# Launch with 8 GPUs
torchrun --standalone --nproc_per_node=8 train.py

# Alternative older syntax
python -m torch.distributed.launch --nproc_per_node=8 train.py
```

The `torchrun` command automatically sets:  

- **RANK** - process rank (0 to 7)  
- **WORLD_SIZE** - total processes (8)  
- **LOCAL_RANK** - local GPU index  

### Data Distribution Strategy

**Batch Distribution**  
In DDP, we distribute various batches to different GPUs. 
Recall the example in the previous section about gradient accumulation. 
Assuming we have `batch_size=4`, `seq_len=2048`, `tokens_per_update=2**19`, then we would need 64 forward passes before we take a single optimizer step.  

However if we now parallize this operation across 4 GPUs, in each forward pass, we would process 4 subsets at once, reducing the total number of forward passes by a factor of `num_gpus`, in this case, from 64 forwards passes per optimizer step to 16 forward passes per optimizer step (since each forward pass now is equivalent to 4 forward pass on a single GPU)

Our dataset loader handles this distribution:

```python
dataset_loader = DatasetLoader(
    batch=batch_size, 
    seq_len=max_seq_len, 
    process_rank=ddp_rank,
    num_processes=ddp_world_size, 
    dataset_dir=config.dataset_dir, 
    device=device
)
```

Each process gets a different shard of data, ensuring no overlap between GPUs.

### Gradient Accumulation with DDP

Gradient accumulation requires special handling in DDP. We need to ensure the accumulation steps are properly synchronized:

```python
# Need to make sure gradient accumulation step is evenly divisible by # GPUs
assert grad_accum_steps % ddp_world_size == 0, (
    f"{grad_accum_steps=} % {ddp_world_size=} != 0\n"
    f"Please adjust 'tokens_per_update' in config file accordingly!"
)

# Adjust accumulation steps per process
grad_accum_steps = grad_accum_steps // ddp_world_size
```

This matters because each process accumulates gradients independently. We need to ensure the total accumulation across all processes matches our desired effective batch size.

### Model Wrapping and Compilation

**DDP Model Setup**  
The model needs to be wrapped with DDP after moving it to the appropriate device:

```python
# Basic DDP wrapping
if ddp:
    model_handle = DDP(model, device_ids=[ddp_rank])
```

**Advanced: DDP with Model Compilation**  
Our code handles the complex interaction between DDP and `torch.compile`:

```python
# Compiling the model via torch.compile reduces the training time
# Though may not be compatible with certain GPUs. If so, turn "compile_model" in config to False
if enable_compilation and ddp:
    # model_handle = DDP(torch.compile(model), device_ids=[ddp_rank]) REMOVE

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

**Important:** The order matters! As mentioned in the comments, the DDP documentations from [PyTorch](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) said that the recommended order is apply DDP first, then compile the model. However others have also done it the other way around. Seems like people are split between the two? Here, I just follow the docs recommendation. 

(Further details about model compilation will be detailed in the `Throughput Optimizations` page)

### Gradient Synchronization Mechanics

**The All-Reduce Operation**  
DDP uses an all-reduce operation to synchronize gradients. Here's what happens:
During each backward pass, all GPUs compute their local gradients based on the given batch of data.  
When we are about to update the parameter values in the mode, applying All-reduce averages gradients across GPUs  
The result is that every GPU has identical averaged gradients and the model is kept in sync.

**Efficient Synchronization with Gradient Accumulation**  
For gradient accumulation, we need to control when synchronization happens:

```python
# Only synchronize if at the step right before stepping optimizer
if ddp:
    model_handle.require_backward_grad_sync = (step % grad_accum_steps == 0)

loss.backward()

if step % grad_accum_steps == 0:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

This ensures we only perform the expensive all-reduce operation when we're actually ready to update weights.

---

### Theoretical Speedup

The ideal speedup with DDP is nearly linear:  

- 2 GPUs: ~1.9x speedup  
- 4 GPUs: ~3.8x speedup  
- 8 GPUs: ~7.5x speedup  

Not fully linear due to additional All-Reduce operations and various overheads, though not too much.

---

### Integration with Other Parallelism Strategies

DDP can be combined with other parallelism methods:

- **Pipeline Parallelism**  
  Split model layers across GPUs  
  DDP handles data parallelism within each stage  

- **Tensor Parallelism**  
  Split individual layers across GPUs  
  Often used with DDP for extreme scaling  

Our Current Approach: We're using pure data parallelism, which is sufficient for models that fit on a single GPU. As models grow larger, you might need to combine DDP with these other strategies.

---

## Conclusion

DDP is a powerful tool that makes multi-GPU training remarkably straightforward. This implementation demonstrates:

- Proper initialization with environment variables  
- Model wrapping and compilation order  
- Gradient accumulation with controlled synchronization  
- Master-process coordinations

The key insight is that DDP allows us to think about training in terms of global batch sizes while automatically handling the distribution across multiple GPUs. This abstraction makes it possible to scale training without rewriting the entire training loop.

As models continue to scale, understanding DDP will be essential for efficient resource utilization and achieving state-of-the-art results.
