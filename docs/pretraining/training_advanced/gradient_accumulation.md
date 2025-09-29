# Gradient Accumulation

Modern large language models need to be trained with **very large effective batch sizes** — often hundreds of thousands of tokens per update minimum. Large batches provide a better statistical approximation of the "true gradient" (the gradient computed over the entire dataset), which leads to **more stabilized training**. This is particularly important for LLM training where the signal-to-noise ratio in gradients can be very low with small batches.

However, there's a fundamental hardware constraint: **GPU memory limitations**.

---

## The Memory Bottleneck: Why We Can't Just Increase Batch Size

When training neural networks, GPU memory is consumed by several components:

1. **Model Parameters:** The weights and biases of the model (e.g., 1.3B parameters ≈ 2.6GB in FP16)
2. **Optimizer States:** Additional values like momentum buffers (e.g., Adam adds ~2x model size)
3. **Gradients:** Gradients for each parameter (same size as parameters)
4. **Activations:** Intermediate results from forward pass needed for backward pass

The critical insight is that **activations scale linearly with batch size and sequence length**. When you double the batch size, you approximately double the memory needed for activations during the forward and backward passes.

Let's examine a concrete example:

- Sequence length (`seq_len`) = 2048 tokens  
- Per-GPU batch size = 4 sequences  
- Hidden dimension = 2048 (value used in this project)

Each forward/backward pass processes:
$$
\text{tokens per batch} = 4 \times 2048 = 8,192 \text{ tokens}
$$

The memory required for the model/optimizer parameters and gradients alone is almost 10.5GB, not to mention the activations which dominates the total memory usage. If we tried to increase this to the desired effective batch size of ~524,288 tokens, we'd need:

$$
\text{Required multiplier} = \frac{524,288}{8,192} = 64\times \text{ more tokens}
$$

Attempting to push to 64× larger effective batch size would imply an enormous jump in memory demand. In practice, activations already dominate memory usage at modest batch sizes, often consuming several times more memory than parameters and optimizer states combined. Scaling this up directly would push requirements well beyond what any single GPU can handle — potentially into the hundreds of gigabytes range. This is why gradient accumulation (and activation checkpointing, which will not be covered here) is essential, used to reduce memory requirements.

---

## Gradient Accumulation: Simulating Large Batches with Limited Memory

Gradient accumulation solves this problem by breaking the large batch into **micro-batches** and accumulating gradients across multiple forward/backward passes before performing a single optimizer update.

The procedure works as follows:

1. **Forward Pass:** Process a small micro-batch through the model
2. **Backward Pass:** Compute gradients for that micro-batch
3. **Accumulate:** Add these gradients to a running total (instead of updating weights)
4. **Repeat:** Process `grad_accum_steps` micro-batches
5. **Update:** Perform a single optimizer step using the accumulated gradients
6. **Reset:** Clear the gradient buffer and repeat

Mathematically, this is equivalent to training with the large batch all at once because:  

- The gradient of a sum equals the sum of gradients
- By properly averaging, we get the same update as if we processed all data simultaneously

---

## Gradient Normalization

Since PyTorch's autograd system accumulates gradients by **summation**, we need to ensure that after accumulating across multiple micro-batches, the gradients represent the **average** rather than the sum.

The most common approach is to scale the loss accordingly:  

```python
loss /= grad_accum_steps  # Normalize the loss
loss.backward()  # Each backward adds gradient/accum_steps
```
After `grad_accum_steps`, the gradients will sum to the correct average.

---

## Implementation in Training Code

Here's how gradient accumulation is typically implemented in a training loop:

```python
for step in range(1, train_iterations+1):
    x, y = dataset_loader.get_batch()

    with torch.autocast(device_type="cuda" if "cuda" in device else "cpu",
                        dtype=torch.bfloat16 if use_amp else torch.float32):
        pred = model_handle(x)
        B, T, C = pred.shape
        loss = criterion(pred.reshape(B * T, C), y.reshape(B * T))

    train_loss_value = loss.item()  # Log before normalization
    loss /= grad_accum_steps        # Normalize loss for accumulation
    loss.backward()

    if step % grad_accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

Let's break down what happens at each step:

- **Micro-batch Processing:** Each call to `get_batch()` returns a small batch (e.g., 4 sequences × 2048 tokens = 8,192 tokens)
- **Loss Computation:** The model computes loss on this micro-batch
- **Loss Scaling:** We divide the loss by `grad_accum_steps` to ensure proper averaging
- **Gradient Accumulation:** `loss.backward()` adds the normalized gradients to the buffer
- **Conditional Update:** Only every `grad_accum_steps` iterations do we actually update the weights and reset gradients

---

## Calculating the Accumulation Steps

The number of gradient accumulation steps is determined by your target effective batch size:

$$
\text{grad_accum_steps} = \frac{\text{target tokens per update}}{\text{tokens per micro-batch}}
$$

Where:

- `tokens per micro-batch =` (batch size × sequence length)
- `target tokens per update =` your desired effective batch size

For example:

- `B = 4`
- `T = 2048`
- `tokens_per_batch = 8192`
- `Desired tokens_per_update = 524,288`

The calculation would be:

$$
\text{grad_accum_steps} = \frac{524,288}{8,192} = 64
$$

This means we accumulate gradients over 64 micro-batches before each optimizer step, effectively training with 524,288 tokens per update while only ever storing 8,192 tokens in memory at once.

---

## Practical Considerations

### Memory vs. Time Trade-off
Gradient accumulation allows you to simulate large batches with limited memory, but it comes with a time cost:

- Without accumulation: 1 forward/backward = 1 optimizer step
- With accumulation: N forward/backward = 1 optimizer step

You're effectively trading memory for computation time.

### Gradient Behavior
With gradient accumulation, the optimizer sees less frequent but higher-quality gradient estimates. This often allows for:

- Higher stable learning rates
- Smoother convergence curves
- Better final model performance

---

## Summary

Gradient accumulation is an essential technique for LLM training that enables:

- Large effective batch sizes despite GPU memory constraints
- Stable training through better gradient estimates
- Flexible scaling to match the batch sizes required by modern scaling laws

The implementation requires just two simple modifications to the training loop: **loss normalization** and **conditional optimizer steps**, but these few lines of code are critical for successful large-scale model training.
