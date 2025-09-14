# Transformer Decoder Block

The **decoder block** is the fundamental building unit of the transformer.  
Each block combines **attention**, **feedforward networks**, **normalization**, and **residual connections** into a repeatable structure.

---

## Structure of a Decoder Block

A decoder block has two main parts:

1. **Multi-Head Self-Attention (MHA)** → lets tokens exchange information.  
2. **Feedforward Network (FFN)** → transforms the attended features into richer representations.  

Surrounding these are:  
- **RMSNorm** → stabilizes training by normalizing activations.  
- **Residual Connections** → ensure information from earlier layers isn’t lost.  

The primary block flow is:

```
Input → Norm → Attention → Residual → Norm → Feedforward → Residual → Output
```

This **“pre-norm” setup** (normalize before each sub-layer) is known to improve stability in deep transformers.

---

## Example Walkthrough

Let’s step through what happens inside one decoder block.  
Suppose we have an input tensor `x` of shape `(batch, seq_len, n_embd)`.

### 1. First Normalization
```python
h = self.norm1(x)
```
- RMSNorm is applied to `x`.  
- This ensures the activations are scaled to a stable range before entering attention.  
- Unlike LayerNorm, RMSNorm does not recenter the mean — it only rescales variance.  

### 2. Multi-Head Self-Attention
```python
attn_out = self.attention(h, freqs_complex)
```
- Each token produces **query, key, and value** vectors.  
- Rotary Position Embeddings (RoPE) are applied to Q and K to inject positional info.  
- Attention computes how strongly each token attends to others in the sequence.  
- The output has the same shape as the input: `(batch, seq_len, n_embd)`.

### 3. First Residual Connection
```python
h = x + attn_out
```
- Here we add the original input `x` back to the attention output.  
- This is called a **residual connection** (or skip connection).  

Why is this important?  
- Imagine stacking dozens of layers. Without skip connections, the network could "forget" the original signal after being transformed multiple times.  
- By adding `x` back, we preserve the original information while also giving the model access to the new transformed features from attention.  
- During backpropagation, residuals also help gradients flow more smoothly, preventing vanishing or exploding gradients.  
- In practice, you can think of it as: *the model learns adjustments (deltas) on top of the original input, instead of rewriting it from scratch every time.*  

### 4. Second Normalization
```python
h_norm = self.norm2(h)
```
- Again, normalize before the next sublayer.  
- This keeps the values stable before passing into the FFN.  

### 5. Feedforward Network
```python
ffn_out = self.ffwd(h_norm)
```
- Input passes through a SwiGLU feedforward network (as described in detail in the FFN doc).  
- Adds nonlinearity and transformation capacity.  
- Output shape: `(batch, seq_len, n_embd)`.

### 6. Second Residual Connection
```python
out = h + ffn_out
```
- Again, the skip connection ensures that the block doesn’t overwrite the information coming from the attention stage.  
- Instead, it layers on additional transformations from the FFN.  
- By the time you stack many decoder blocks, each one is contributing refinements while keeping the original context intact.  
- This makes the network much more robust and trainable.  

Final output shape: `(batch, seq_len, n_embd)`.

---

## In This Project

- **Attention type**: defaults to standard multi-head self-attention, with optional MLA for efficiency.  
- **Normalization**: RMSNorm used everywhere (simpler than LayerNorm, but empirically stable).  
- **Activation**: SiLU-based feedforward (SwiGLU).  
- **Dropout**: applied after projections, mainly used during fine-tuning (SFT/RLHF).  
- **Residuals**: used after both the attention and FFN sublayers.  

Together, these form the repeating backbone of the SimpleLLaMA model.  
By stacking many of these blocks, the network can build increasingly complex representations of text sequences.
