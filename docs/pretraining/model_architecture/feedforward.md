# Feedforward Networks (FFN) in Transformers

When people first learn about Transformers, the attention mechanism usually takes the spotlight.  
But the **feedforward network (FFN)** is equally important — in fact, it often accounts for the majority of parameters in the model.  

---

## Why Do We Need Feedforward Layers?

Attention layers are powerful, but they are still fundamentally **linear operations** (matrix multiplications, weighted sums).  
A stack of only linear layers would remain a linear model, which cannot approximate complex, nonlinear functions.  

The feedforward network adds **nonlinearity** and **capacity to transform information**.  
It allows the model to map representations into a higher-dimensional space, apply nonlinear activation, and then project back down.  

In practice, every Transformer block has the structure:  

```
Input → [Attention] → [Feedforward] → Output
```

Both attention and FFN are wrapped with normalization and residual connections.

---

## Vanilla Transformer FFN

In the original Transformer paper (Vaswani et al., 2017), the FFN was defined as:  

```
FFN(x) = max(0, xW1 + b1)W2 + b2
```

- Two linear layers with a ReLU in between.  
- The hidden dimension is usually set to **4× the embedding dimension**, then projected back down.  

For example, if embedding size = 1024, the FFN hidden size = 4096.  

This “expand and contract” pattern gives the model strong nonlinear mixing power, because the network has a wide layer to mix features, then projects it back to the model’s working dimension.  

### Implementation Example

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.layer1 = nn.Linear(n_embd, 4 * n_embd)
        self.layer2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor):
        return self.layer2(torch.nn.functional.relu(self.layer1(x)))
```

Let’s walk through this carefully:  
1. Input tensor `x` has shape `(batch, seq_len, n_embd)`.  
2. First layer projects it into `(batch, seq_len, 4 * n_embd)`.  
3. Apply ReLU to introduce non-linearity (important because without it, stacking linear layers would still just be linear).  
4. Second layer projects it back down to `(batch, seq_len, n_embd)`.  

So while the shape going into and out of the FFN is the same, the hidden computation in between allows the network to express far richer functions.

---

## LLaMA-Style FFN (SwiGLU)

The LLaMA architecture introduced a key modification: instead of the plain ReLU-based FFN, it uses a **SwiGLU activation** (SiLU-Gated Linear Unit).  

Here’s how it looks in code:  

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd: int, multiple_of: int, dropout: float):
        super().__init__()

        hidden_dim = int(4 * n_embd * (2 / 3))  # Authors of LLaMa used 2/3 of 4*n_embd
        # Round hidden_dim up to a nicer multiple for efficient GPU utilization
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        h = F.silu(self.w1(x)) * self.w3(x)  # SwiGLU block
        return self.dropout(self.w2(h))
```

### Breaking It Down Step by Step

- In the vanilla FFN we had **two linear layers**.  
- In LLaMA’s FFN we now have **three linear layers**. Why? To introduce a **gating mechanism**.  

The formula looks like this:  

\[
h = \text{SiLU}(W_1x) \odot (W_3x)
\]

where `⊙` is elementwise (Hadamard) multiplication.  

- `W1x`: main transformation path.  
- Apply SiLU activation (smooth, like ReLU but differentiable everywhere).  
- `W3x`: produces a second transformed version of `x`, used as a **gate**.  
- Multiply them elementwise: the gate decides how much of each hidden unit passes through.  

This means:  
- If a value in `W3x` is near 0 → the signal from `W1x` gets suppressed.  
- If large → it amplifies the signal.  
- If negative → it can flip the sign of the signal.  

So the gating function lets the model regulate the flow of information more flexibly than a simple ReLU cutoff.

### Parameter Balancing

But wait — adding a third projection layer means more parameters, right?  
Yes, but the authors balanced this by shrinking the hidden dimension.  

- Vanilla FFN parameters: about `8 * n_embd²`.  
  (from `(n_embd × 4n_embd) + (4n_embd × n_embd)`).  
- LLaMA FFN: uses hidden_dim = `int(4 * n_embd * 2/3)`.  
  With three projections (`w1`, `w2`, `w3`), total ≈ `8 * n_embd²`.  

So both end up with roughly the same parameter budget, but LLaMA gets more expressive power via gating.  

### Shapes in Action

- Input: `(batch, seq_len, n_embd)`  
- After `w1` and `w3`: `(batch, seq_len, hidden_dim)`  
- After SiLU + elementwise product: `(batch, seq_len, hidden_dim)`  
- After `w2`: `(batch, seq_len, n_embd)`  

Input and output shapes match the original — only the internal transformation is richer.  

### Multiple-of Trick

The `multiple_of` hyperparameter ensures `hidden_dim` is divisible by a convenient number (like 64, 128, 256).  
This is purely for GPU efficiency: matrix multiplications run faster on dimensions aligned to powers of two.  

For example:  
- Without adjustment: `n_embd=1024` → hidden_dim = 2730.  
- With `multiple_of=128`: hidden_dim becomes 2816, which is far more efficient for hardware.  

### Dropout

At the end, dropout is applied. This isn’t heavily used during large-scale pretraining, but it becomes important in fine-tuning (SFT, RLHF) to regularize and prevent overfitting when datasets are smaller.  

---

## Summary

- Feedforward layers provide the **nonlinear expressiveness** that attention alone cannot.  
- Vanilla Transformer FFN: **Linear → ReLU → Linear**, with a 4× expansion.  
- LLaMA FFN: **SwiGLU-style** (SiLU + gating with a third linear layer).  
- Gating allows richer feature modulation (suppress, amplify, flip).  
- Parameter count is balanced to stay ~`8 * n_embd²`.  
- Input/output shapes stay the same, but the internal computation is more expressive.  
- Most of a Transformer’s parameters live in these FFNs — they are just as crucial as attention.  
