# Throughput Optimizations

Achieving high training throughput is crucial for practical LLM development, and this section outlines key optimizations in the training script that deliver noticeable speedups.

---

## 1. BF16 vs FP32/FP16: The Precision Trade-off

### Understanding Numerical Precision

- **FP32**: 32-bit floating point (standard float)  

    - Range: ±1.7×10³⁸, Precision: ~7 decimal digits  
    - High precision, large memory footprint  

- **FP16**: 16-bit floating point (half precision)  

    - Range: ±65,504, Precision: ~4 decimal digits  
    - Small memory, but limited range can cause overflow/underflow  

- **BF16**: Brain Floating Point (Google's solution)  

    - Range: ±3.4×10³⁸ (same exponent as FP32, much larger than FP16)  
    - Precision: ~3 decimal digits (7 mantissa bits, less than FP16)
    - **Ideal for deep learning**: preserves range, sacrifices precision  

### Why BF16 Wins for LLM Training

```python
# BF16 has the dynamic range of FP32 with the memory footprint of FP16
fp32_tensor = torch.randn(1000, 1000, dtype=torch.float32)  # 4MB
bf16_tensor = torch.randn(1000, 1000, dtype=torch.bfloat16)  # 2MB

print(f"FP32 range: ~10^38, BF16 range: ~10^38 (same!)")
print(f"FP16 range: ~10^4 (much smaller - risk of overflow)")
```

The key is that LLM training is more sensitive to range than precision. Gradient values can span many orders of magnitude, making BF16's preserved range crucial.

Do note that although this would reduce memory usage substantially, but would not directly halve the memory usage, a few layers/activations in the model are kept in FP32 due to precision sensitivity, along with optimizer state dicts which stays in FP32. 

---

## 2. Automatic Mixed Precision (AMP) with Autocast

As mentioned, not all operations benefit from lower precision. Some need FP32 for numerical stability:

- **BF16-friendly:** Heavy matrix multiplications, elementwise operations, gradients
- **FP32-required:** Reductions, loss computation, certain activations that requires numeric precision  


PyTorch handles this by providing `torch.autocast` to automatically manage precision:

```python
use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

with torch.autocast(device_type="cuda" if "cuda" in device else "cpu",
                   dtype=torch.bfloat16 if use_amp else torch.float32):
    # Everything in this context uses mixed precision
    pred = model_handle(x)  # Most ops in BF16
    B, T, C = pred.shape
    loss = criterion(pred.reshape(B * T, C), y.reshape(B * T))  # Loss in FP32
```

### How Autocast Works

- **Operator-Level Precision Rules:** PyTorch has built-in rules for each operation type  
- **Automatic Casting:** Inputs are cast to appropriate precision  
- **Output Casting:** Results are cast back to expected types  

Overall, this results not only in memory usage reduction, operations in BF16 is much faster compared to in FP32 on modern tensor cores. 

---

## 3. Torch.Compile: Graph-Level Optimizations

### From Eager Mode to Graph Mode

PyTorch normally runs in eager mode: each operation executes immediately. This is flexible but inefficient due to Python overhead.

`torch.compile` converts your model to a graph that can be optimized:

```python
# Before compilation (eager mode)
if enable_compilation:
    model_handle = torch.compile(model)
    
# After compilation - same API, much faster execution
pred = model_handle(x)  # Now runs as optimized graph
```

### What Compilation Optimizes
- Kernel Fusion: Combine multiple operations into single kernels  
- Memory Layout Optimization: Optimize tensor memory access patterns  
- Dead Code Elimination: Remove unnecessary operations  
- Constant Propagation: Precompute constant expressions  

### Real-World Example: Attention Mechanism

```python
# Eager mode: many small operations
def attention_eager(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1))  # Separate kernel
    scores = scores / math.sqrt(Q.size(-1))        # Separate kernel  
    attn = torch.softmax(scores, dim=-1)           # Separate kernel
    output = torch.matmul(attn, V)                 # Separate kernel
    return output
```

Compiled: potentially fused into fewer kernels.  

The compiler can merge these operations for better memory locality.

---

## 4. Fused Optimizer Kernels

Similar to how `torch.compile` offers kernel fusion operation in the model, optimizers can also provide fusion operations to more efficiently compute and update model parameters. 
 
In a more naive implementation pytorch may launch a lot of small kernels which would be relatively inefficient with overhead costs, coming from many small GPU operations.

Here we can inspect if the optimizer supports fused operations, and pass that kwarg during optimizer instantiation to combine multiple operations into single kernels:

```python
# Check for fused optimizer support
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and (device == "cuda" or ddp)
extra_args = dict(fused=True) if use_fused else dict()

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=max_lr, 
    betas=(beta1, beta2), 
    weight_decay=weight_decay, 
    **extra_args  # Uses fused kernels if available
)
```

In general, there would be some speed up optimizations, though that would be dependent upon the optimizer and operations within the model. 
It reduces memory bandwidth usage and is especially beneficial for models with many small parameters  . 
The speedup varies by hardware and model size, but it’s typically a “free” optimization worth enabling when available.

---

## 5. FlashAttention

One of the biggest computational and memory bottlenecks in Transformers comes from the **attention mechanism**.  
Standard attention computes a large score matrix of shape `(B, n_heads, T, T)`. 

Imagine we train a language model using a sequence length, `T`, with a moderate value like 32k, batch size of 4, and 32 heads. A single matrix of that shape using BF16 would require over 250GB of memory alone!
Granted, that's the most naive implementation, but for long sequences, this quadratic cost in both time and memory quickly dominates training, due to it's quadratic nature.

**FlashAttention** is a memory-efficient attention algorithm that reorders the way attention is computed so that the large intermediate matrix never needs to be materialized in GPU memory. Instead, it streams blocks of the computation through high-bandwidth GPU SRAM, drastically reducing memory usage and speeding up execution. Overall computation requirement remains quadratic, however the memory requirement is now linear with respect to sequence length. 

### Standard Attention
1. Compute scores: `QK^T / sqrt(d_head)` → shape `(B, n_heads, T, T)`  
2. Apply softmax → still `(B, n_heads, T, T)`  
3. Multiply by `V`  

Both the score matrix and the softmax output is stored for backpropagation, which can be quite large, depending on the sequence length.

### FlashAttention Approach
- The key trick is to compute attention **block by block** directly in GPU SRAM, never materializing the full `(T, T)` matrix.  
- Uses a numerically stable online softmax to handle blocks sequentially.  
- Backward pass recomputes pieces instead of storing them, further saving memory.

The benefits is that this approach reduces activation memory from O(T²) to O(T), allowing much longer context lengths without blowing up VRAM and often results in 2–4× faster computation for attention at long sequences.  

It's important to recognize that this is just a very high level overview of what it does. Flash Attention is quite complex and explaining the details would fall outside the scope of this project. 
If interested, here's the link to read more about this: [Flash Attention Paper](https://arxiv.org/pdf/2307.08691)

### Example Usage

In this project, Flash Attention is applied as: 


```python
# (batch, seq_len, n_heads, h_dim) -> (batch, n_heads, seq_len, h_dim)
q = q.permute(0, 2, 1, 3)
k = k.permute(0, 2, 1, 3)
v = v.permute(0, 2, 1, 3)

if self.use_flash_attention:
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
else:
    # (batch, n_heads, seq_len, h_dim) @ (batch, n_heads, h_dim, seq_len) -> (batch, n_heads, seq_len, seq_len)
    y = q @ k.transpose(-2, -1) / (self.h_dim ** 0.5)
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device))
    y = y.masked_fill(~mask, float("-inf"))
    y = F.softmax(y, dim=-1)

    # (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, h_dim) -> (batch, n_heads, seq_len, h_dim)
    y = y @ v
```

Given the Q, K, V matrices of shape `(batch, n_heads, seq_len, h_dim)`, we pass it to `nn.functional.scaled_dot_product_attention`, telling it that this uses causal masking, and would use flash attention to compute the result. 

Whether or not it's used (depending if compatible with torch version/gpu device) is set in the configuration, which is then passed to the model during instantiation. 

---

## Summary

Throughput optimizations transform LLM training from impractical to feasible. This implementation demonstrates common approaches used in development:

- Layered optimizations that compound benefits  
- Automatic precision management with AMP  
- Graph-level optimizations via torch.compile  
- Hardware-aware kernels for maximum performance  

While each provides individual benefits, these combination of optimizations enables the scale of modern LLM training.
