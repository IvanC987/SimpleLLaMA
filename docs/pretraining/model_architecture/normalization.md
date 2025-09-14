# Normalization

When training deep neural networks, one of the recurring problems is that activations can either **explode** (grow without bound) or **vanish** (shrink toward zero) as they pass through many layers. This makes optimization unstable and slows down learning.  

Another issue is **internal covariate shift** — the idea that the distribution of activations keeps changing as each layer is updated during training, which makes it harder for later layers to adapt.  

Internal Covariate Shift can be thought of as follows: 

Imagine you’re a student preparing for math exams.  

- On the first day, you get an **Algebra I exam**. You do your best, turn it in, get feedback from the teacher, and feel ready to improve for the next one.  
- Based on the feedback, you adjust your study strategy and expect another Algebra test.  
- But the next exam you receive is suddenly **Calculus** — a totally different level of difficulty. All the preparation you just did no longer matches what you’re being tested on.  

This is what happens in deep neural networks without normalization.  
Each hidden layer is like a student preparing for its next “exam” (the next round of inputs). After every weight update, the **distribution of outputs from the previous layer shifts**. That means the “exam” the next layer sees can suddenly look very different than what it was trained on.  

As the network gets deeper, these unexpected shifts **compound**, making training unstable. Layers spend more time re-adapting to constantly changing input distributions rather than actually learning useful features.

**Normalization layers** (like BatchNorm, LayerNorm, etc.) fix this problem.  
They act like a teacher who ensures that every new exam stays at the same Algebra level — same general difficulty, same type of questions — just slightly adjusted each time. This consistency allows each layer to steadily improve rather than getting thrown off by wild distribution shifts.

In short:  
- *Without normalization*: “I prepared for Algebra, but got Calculus.”  
- *With normalization*: “I keep getting Algebra, just with different numbers.”  



Normalization layers stabilize the distribution of activations, keep gradients more predictable, and generally allow networks to train deeper and faster.

---

## Layer Normalization (LayerNorm)

In Transformers (like the original paper “Attention Is All You Need”), the normalization method of choice was **Layer Normalization (LayerNorm)**.

How it works:  
- Given a tensor `x` of shape `(batch, seq_len, n_embd)`, LayerNorm normalizes **across the embedding dimension** for each token.  
- For each token vector, it computes the mean and variance across its `n_embd` values.  
- The normalized vector is then scaled and shifted by learnable parameters (`gamma` and `beta`).  

Mathematically:

```
LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
```

Where:  
- `mean(x)` and `var(x)` are computed over the last dimension (`hidden_dim`).  
- `gamma` and `beta` are learnable parameters that allow the model to "undo" normalization if needed.  
- `eps` is a small constant to prevent division by zero.  

Effectively, LayerNorm **re-centers** (subtracts the mean) and **re-scales** (divides by standard deviation). This ensures each token’s hidden vector has roughly zero mean and unit variance before being rescaled.

LayerNorm is still widely used in most Transformer implementations, but it comes with a computational cost: subtracting means, computing variances, and performing square roots for every vector.

---

## Root Mean Square Normalization (RMSNorm)

LLaMA and some later models (including this implementation) instead use **RMSNorm**, a simpler but surprisingly effective alternative.

The key idea:  
Research showed that *re-centering* (subtracting the mean) is less important than *re-scaling* (fixing the variance). In other words, what really stabilizes activations is making sure their magnitude (energy) doesn’t blow up, not whether they’re mean-centered.

So RMSNorm skips the mean subtraction entirely.

Mathematically:

```
RMSNorm(x) = (x / RMS(x)) * weight
```

Where:  
- `RMS(x) = sqrt(mean(x^2))`  
- `weight` is a learnable scaling vector (similar to `gamma` in LayerNorm).  
- No `beta`, since there’s no re-centering.  

This implementation of `RMSNorm` shows this clearly:

```python
class RMSNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)
```

Step by step:  
1. Square all elements (`x.pow(2)`).  
2. Take the mean across the last dimension (`mean(-1, keepdim=True)`).  
3. Add a tiny epsilon for numerical stability.  
4. Take the reciprocal square root (`rsqrt`).  
5. Multiply back with the original `x` → now the activations are normalized to unit RMS.  
6. Finally, scale by a learnable weight vector (one parameter per hidden dimension).  

This achieves the stabilization effect of LayerNorm but with slightly fewer operations.

---

## Why RMSNorm?

So why is RMSNorm used in models like LLaMA (and this project)?

1. **Efficiency**  
   RMSNorm is simpler than LayerNorm. It removes the mean subtraction, which slightly reduces compute and memory usage — especially important when running at very large scales.

2. **Empirical stability**  
   Experiments ([see the RMSNorm paper](https://arxiv.org/pdf/1910.07467)) showed that mean-centering didn’t improve stability much. The key factor was scaling by the variance (or root mean square).

3. **Better fit for Transformers**  
   Since Transformers already have residual connections and other stabilizing tricks, skipping the mean step doesn’t hurt — and in fact, models trained with RMSNorm often match or exceed performance of LayerNorm.

---

## Putting It Together

In this project, RMSNorm appears inside the **Decoder Block** in two places:

```python
h = x + self.attention(self.norm1(x), freqs_complex)
out = h + self.ffwd(self.norm2(h))
```

Here, `norm1` and `norm2` are both `RMSNorm` layers. They normalize activations before passing them into attention and feedforward sublayers. This is known as a **Pre-Norm Transformer** design (normalize before the sublayer, then add the residual). Pre-Norm improves gradient flow compared to the original Post-Norm design.

---

## Summary

- Normalization stabilizes deep networks, preventing exploding/vanishing activations.  
- Transformers originally used **LayerNorm**, which re-centers and re-scales each hidden vector.  
- **RMSNorm** drops the mean subtraction, keeping only the re-scaling step.  
- Despite being simpler, RMSNorm works just as well (sometimes better) and is used in LLaMA and your implementation.  
- In SimpleLLaMA, RMSNorm ensures stable training across all decoder blocks while keeping the implementation lightweight.  
