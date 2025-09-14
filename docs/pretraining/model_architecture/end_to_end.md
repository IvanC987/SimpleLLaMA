# End-to-End Walkthrough

Now at this point, most of the aspects of the architecture and pipeline have been covered in detail.  
This final page will be used to **tie everything together** and give a more thorough, step-by-step overview of how all the parts interact to form a working large language model.  

---

## 1. From Raw Text to Tokens

We first start out with a **massive corpus of text** — think in the scale of hundreds of gigabytes upwards, containing billions or even trillions of tokens.  
This corpus is gathered from sources like books, Wikipedia, academic papers, code repositories, and curated parts of the internet.  

But raw text isn’t useful to the model. The model only works with numbers, so the very first step is to **tokenize** this text using a pretrained tokenizer.  

For example, suppose we have the sentence:  

```
"The quick brown fox jumps over the lazy dog."
```

The tokenizer will break this into smaller units (subwords or characters depending on the algorithm) and then convert each into an integer ID.  

That means something like:  

```
["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", "."]
→ [1202, 850, 149, 4211, 769, 1839, 3521, 4879, 2035, 1209]
```

Now the sentence is represented as a sequence of integers.  
This is the form that the neural network can actually process.  

---

## 2. Batching and Shaping the Data

Instead of feeding one sentence at a time, training uses **mini-batches** to process many sequences in parallel.  
This is crucial for efficiency on GPUs/TPUs.  

Suppose we have a long stream of tokens like:  

```
[1202, 850, 149, 4211, 769, 1839, 3521, 4879, 2035, 1209, 954, 4461, 3546, 206, 4401, ...]
```

If we set:

- `batch_size = 2`  
- `seq_len = 6`  

We would take the first `batch_size * seq_len = 12` tokens and reshape them into a `(batch, seq_len)` tensor:  

```python
batch_size = 2
seq_len = 6
tokens = [1202, 850, 149, 4211, 769, 1839, 3521, 4879, 2035, 1209, 954, 4461, 3546, 206, 4401, ...]

input_tokens = torch.tensor(tokens[:batch_size * seq_len]).reshape(batch_size, seq_len)

# When printed, input_tokens would look like:
# tensor([[1202,  850,  149, 4211,  769, 1839],
#         [3521, 4879, 2035, 1209,  954, 4461]])
```

This reshaped tensor is now ready for the model.  

In realistic training runs, values are much larger, e.g.:  
- `batch_size = 32` (process 32 sequences in parallel)  
- `seq_len = 2048` (each sequence is 2048 tokens long)  

So the model processes a tensor of shape `(32, 2048)` in one forward pass.  

---

## 3. Passing Through the Transformer Model

Next, this tensor of token IDs is passed into the **transformer model**.  
The model is composed of the architecture we previously touched upon in detail: embeddings, attention, feedforward networks, normalization, and residual connections, all stacked together into many decoder blocks.  

Here is the structure of the `LLaMaTransformer` class that uses all the previous building blocks:  

```python
class LLaMaTransformer(nn.Module):
    def __init__(self,
                 config: any,
                 tokenizer: tokenizers.Tokenizer,
                 device: str):

        super().__init__()

        # === Unpack config ===
        max_seq_len = config.max_seq_len
        n_embd = config.n_embd
        n_heads = config.n_heads
        n_layers = config.n_layers
        multiple_of = config.multiple_of
        # And many more, will omit for documentation

        assert n_embd % n_heads == 0, f"n_embd ({n_embd}) % n_heads ({n_heads}) must equal 0!"
        assert (n_embd // n_heads) % 2 == 0 and qk_rope_head_dim % 2 == 0, "head_dim must be even for RoPE!"


        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device
        self.embeddings = nn.Embedding(tokenizer.get_vocab_size(), n_embd)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(n_embd=n_embd, n_heads=n_heads, multiple_of=multiple_of, use_mla=use_mla, ...)
             for _ in range(n_layers)])
        self.norm = RMSNorm(n_embd, eps)
        self.final_linear = nn.Linear(n_embd, tokenizer.get_vocab_size(), bias=False)

        h_dim = qk_rope_head_dim if use_mla else n_embd // n_heads
        self.freq_complex = precompute_theta_pos_frequencies(max_seq_len, h_dim, theta, device)

    def forward(self, x):
        batch, seq_len = x.shape

        # (batch, seq_len) -> (batch, seq_len, n_embd)
        h = self.embeddings(x)  # Each token ID mapped to vector
        freqs_complex = self.freq_complex[:seq_len]

        # Pass through all Decoder Blocks
        for dec_block in self.decoder_blocks:
            h = dec_block(h, freqs_complex)

        h = self.norm(h)
        return self.final_linear(h)  # (batch, seq_len, n_embd) -> (batch, seq_len, vocab_size)
```

---

## 4. Step-by-Step Inside the Model

### 4.1 Embedding Layer
- Input: `(batch, seq_len)` of integers.  
- Each integer token is mapped into a dense vector of length `n_embd`.  
- Output: `(batch, seq_len, n_embd)`  

This is the **semantic representation** of tokens: instead of just IDs, we now have vectors that carry meaning and relationships.  

### 4.2 Positional Information
We then fetch `freqs_complex`, which stores precomputed values for **Rotary Position Embeddings (RoPE)**.  
RoPE encodes the positions of tokens into the attention mechanism, so the model knows order (e.g., difference between “dog bites man” and “man bites dog”).  

### 4.3 Decoder Blocks
The embedded tensor is then passed into the first **Decoder Block**.  
Each block applies:  
- **RMSNorm** → normalizes values for stability.  
- **Multi-Head Attention** → lets each token attend to others in the sequence.  
- **Feedforward Network (SwiGLU)** → adds nonlinear transformation capacity.  
- **Residual Connections** → add the original input back to preserve information.  

The internal computations change the **contents** of the tensor but not its shape: it remains `(batch, seq_len, n_embd)`.  

The output of block 1 is passed into block 2, then block 3, and so on, until it passes through all `n_layers`.  

### 4.4 Final Normalization
After the last decoder block, the tensor goes through one final **RMSNorm layer**.  
This ensures the distribution of activations is stable before the final projection.  

### 4.5 Final Linear Layer
Finally, we project each hidden vector of size `n_embd` into a vector the size of the vocabulary.  
- Shape: `(batch, seq_len, n_embd) → (batch, seq_len, vocab_size)`  

This gives **logits** — raw scores for each token in the vocabulary.  

---

## 5. From Logits to Predictions

The logits are not yet probabilities. To turn them into probabilities, we apply **softmax** across the vocabulary dimension:  

```
probabilities = softmax(logits, dim=-1)
```

Now, for each position in the sequence, we get a probability distribution over the vocabulary.  
For example:  

```
At position 6, "fox" might have 0.72 probability, "dog" 0.05, "cat" 0.02, ...
```

During training, we compare these probabilities against the actual next token using **cross-entropy loss**.  
This loss guides backpropagation, which updates the model weights via gradient descent.  

---

## 6. Training Loop Connection

To summarize the **training loop** connection:  

1. Start with batch of token sequences.  
2. Forward pass through embeddings → decoder blocks → normalization → linear layer.  
3. Produce logits `(batch, seq_len, vocab_size)`.  
4. Apply softmax to get probabilities.  
5. Compute loss vs. ground truth next tokens.  
6. Backpropagate gradients.  
7. Update weights with optimizer (AdamW, etc.).  
8. Repeat across billions of tokens until the model converges.  

Over time, the model gradually learns grammar, facts, and semantics purely by predicting the next token.  

---

## Key Takeaway

This end-to-end flow shows how everything connects:  
- **Tokenization** converts raw text into IDs.  
- **Embeddings + RoPE** give meaning and order.  
- **Decoder Blocks** repeatedly transform and refine the representations.  
- **Final Linear Layer + Softmax** produce predictions over the vocabulary.  
- **Loss and Optimization** allow the model to learn from its mistakes.  

By stacking these stages together, and scaling up with billions of tokens and parameters, we arrive at a large language model capable of generating coherent and context-aware text.  
