# Attention

Attention is the heart of the transformer. It’s the mechanism that lets the model decide:  
**“Which other tokens in the sequence are relevant for predicting the next one?”**

---

## Vanilla Self-Attention

In the vanilla self-attention, we first start out with a given tensor, `x`, of shape `(batch, seq_len, n_embd)` as mentioned in the previous `embeddings` section.

Given that tensor, we compute three tensors, Each of them is a linear projection from the input tensor `x`  
- **Query (Q)**: what each token is looking for.  
- **Key (K)**: what each token offers.  
- **Value (V)**: the information carried by each token.  


This is done by either creating 3 separate linear layers in the constructor for the self-attention class: 

```python
self.q_proj = nn.Linear(n_embd, n_embd)
self.k_proj = nn.Linear(n_embd, n_embd)
self.v_proj = nn.Linear(n_embd, n_embd)

self.o_proj = nn.Linear(n_embd, n_embd)  # This will be used and touched upon later on
```

Where it can later be invoked to create the Q, K, and V tensors that will be used in attention computation:

```python
# Takes an input tensor x of shape (batch, seq_len, n_embd) and linearly project it into another tensor, retaining the shape
q = self.q_proj(x)  
k = self.k_proj(x)
v = self.v_proj(x)
```

However a more common method is the merge all of those linear layers into a single one, for more efficient computation:
```python
self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)

self.o_proj = nn.Linear(n_embd, n_embd)
```

Then later on, use that to get: 

```python
# Input shape: (batch, seq_len, n_embd), output shape: (batch, seq_len, 3 * n_embd)
qkv = self.qkv_proj(x)

# Split the tensor along the last dimension 
q, k, v = qkv.chunk(3, dim=-1)
```

Both of those will produce the same result. 

At this point, given an input tensor `x`, we now have 3 separate tensors `q`, `k`, and `v`, all of the shape `(batch, seq_len, n_embd)`  


Next, we 'expand' the tensor from 3d to 4d, using the `n_heads` hyperparameter. 
`n_heads` defines how many 'heads' we want in attention. Further explanation as to what it does will be given below. 

We would use the last dimension, `n_embd` and divide that into `(n_heads, head_dim)`, based on the formula `head_dim = n_embd // n_heads`
For example, given `n_embd=1024`, `n_heads=16`, then `head_dim=1024//16=64`, meaning we transform our 1024 embedding dimensions into 16 heads, each head have 64 dimension to work with. 
It is crucial to add an assertion that `n_embd % n_heads == 0` to make sure it evenly divides. 

Given the hyperparameter `n_heads`, calculate `head_dim`, then view/reshape the tensor accordingly as follows: 
```python
# (batch, seq_len, n_embd) -> (batch, seq_len, n_heads, head_dim)
q = q.view(batch, seq_len, n_heads, head_dim)
k = k.view(batch, seq_len, n_heads, head_dim)
v = v.view(batch, seq_len, n_heads, head_dim)
```

Finally, we swap the dimension for `seq_len` and `n_heads`

```python
# (batch, seq_len, n_heads, head_dim) -> (batch, n_heads, seq_len, head_dim)
q = q.permute(0, 2, 1, 3)
k = k.permute(0, 2, 1, 3)
v = v.permute(0, 2, 1, 3)
```


Now that our Q, K, V matrices are all of shape `(batch, n_heads, seq_len, head_dim)`, we apply the self attention formula: 

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + Mask \right) V
$$


The first step is to compute $$QK^\top$$
Looking at the shape of the tensors, we will get an output:
`(batch, n_heads, seq_len, head_dim) @ (batch, n_heads, head_dim, seq_len) = (batch, n_heads, seq_len, seq_len)`  
which we can denote as `attn_scores = QK^T`  

Note the `(seq_len, seq_len)` ending dimensions. That's the reason why people say that attention computation scales quadratically w.r.t the seq_length of the model

Next, we apply element-wise division on the computed matrix, where the divisor is the sqrt of `d_k`, which from the paper refers to the dimensions of each head (`head_dim`)

`attn_scores = attn_scores / math.sqrt(head_dim)`

The attention scores is normalized by head-dim primarily because of the softmax operation that will immediately take place next.  
In short, typically the Q, K, V matrices have unit gaussian distribution, when we apply matrix multiplication, the variance scales by `head_dim`
However with how softmax works, values that are more than, say, 3 units less than the maximum in the dimension of interest will be heavily squashed to approximately 0. 
In our example scenario, if `head_dim=64`, that means the std increased from 1 to 8, which would compress the tensor into something similar to a one-hot vector.  

Moving along, after we normalize the attention scores, which doesn't change the shape of the tensor, we would need to apply a triangular mask to `attn_scores`.  

Typically, LLMs would add in these mask to prevent the model from 'cheating' by looking at future tokens.  
Sort of like:  
If someone gives you a sentence, 'I love my dog' and asks, 'What is the word after 'my'?'
It's trivial. The answer is 'dog'. However masking prevents that by, as the name suggests, masking out the next token. 
In that same example, the other person would give 'I love my _' and then ask, 'What is the word after 'my'?'

Now in this example, let's use the sentence: "That is a blue dog" tokenized into `[That, is, a, blue, dog]` (Since there are only 5 tokens, that means `seq_len=5` in this example)
After going through the embedding layer and above steps, we will reach a tensor of shape `(batch, n_heads, 5, 5)`
Grabbing one of the (5, 5) matrices arbitrarily might look something like: 



```text
"That"   "is"     "a"      "blue"   "dog"
----------------------------------------------
[ 1.93    1.49     0.90    -2.11     0.68 ]   ← "That"
[-1.23   -0.04    -1.60    -0.75    -0.69 ]   ← "is"
[-0.49    0.24    -1.11     0.09    -2.32 ]   ← "a"
[-0.22   -1.38    -0.40     0.80    -0.62 ]   ← "blue"
[-0.59   -0.06    -0.83     0.33    -1.56 ]   ← "dog"
```


That tells us how much attention each token pays to each other. 
Now these are un-normalized values, so it would be hard to interpret, at least for now. 

We then apply the triangular mask that prevents tokens to look ahead. It would be something like: 

```text
[  0   -∞   -∞   -∞   -∞ ]
[  0    0   -∞   -∞   -∞ ]
[  0    0    0   -∞   -∞ ]
[  0    0    0    0   -∞ ]
[  0    0    0    0    0 ]
```


Applying via element-wise addition:
`attn_scores = attn_scores + mask` 

The result would now look like: 
```text
"That"   "is"     "a"      "blue"   "dog"
----------------------------------------------
[  1.93    -∞      -∞      -∞      -∞  ]   ← "That"
[-1.23   -0.04     -∞      -∞      -∞  ]   ← "is"
[-0.49    0.24   -1.11     -∞      -∞  ]   ← "a"
[-0.22   -1.38   -0.40     0.80     -∞  ]   ← "blue"
[-0.59   -0.06   -0.83     0.33   -1.56 ]   ← "dog"
```


we pass it through the softmax function to transform it into something like a probability distribution. 

```python
attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
```

```text
"That"   "is"     "a"      "blue"   "dog"
----------------------------------------------
[1.0000   0.0000   0.0000   0.0000   0.0000 ]   ← "That"
[0.2330   0.7670   0.0000   0.0000   0.0000 ]   ← "is"
[0.2759   0.5753   0.1488   0.0000   0.0000 ]   ← "a"
[0.2032   0.0632   0.1699   0.5637   0.0000 ]   ← "blue"
[0.1566   0.2658   0.1236   0.3942   0.0596 ]   ← "dog"
```

As you can see, starting out with the token corresponding to 'That', it can only pay attention to itself, since it's the first token in the sequence
Next is the word 'is', which splits the attention between 'That' and itself
So on and so forth. This tells the model how much each token attends to each other. 

At this point, the `attn_weights` tensor is still of shape `(batch, n_heads, seq_len, seq_len)` since both normalization and softmax doesn't change the tensor shape  

Now we process the final steps: 


```python
# (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, head_dim) = (batch, n_heads, seq_len, head_dim)
attn_output = attn_weights @ v

# (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, n_heads, head_dim) -> (batch, seq_len, n_embd)
attn_output = attn_output.permute(0, 2, 1, 3).view(batch, seq_len, n_embd)

return self.o_proj(attn_output)
```

The final step here is to first matrix multiply with the `v` tensor to get the tensor of shape `(batch, n_heads, seq_len, head_dim)`
We revert the permutation and viewing to get back the original input shape `(batch, seq_len, n_embd)`
Finally, apply the output projection matrix to `attn_output` before returning. 

So what's the purpose of output matrix? 

One can think of it as a way to combine the information that each head learned. 
Recall that when we apply attention, we use multiple heads. Each head process their own 'chunk' of embedding dimensions compeltely separately from each other. 
It's beneficial to allow them to learn their own information, however at the end, we merely concatenate them together. 

The final output projection matrix allows the information to get 'aggregated' and combined. 

---

## In This Project: Implementation

In `MHSelfAttention`, queries, keys, and values are created together, where it's first viewed then chunked along the last dimension:

```python
qkv = self.qkv_linear(x)  
q, k, v = qkv.view(batch, seq_len, n_heads, 3 * h_dim).chunk(3, dim=-1)
```

- Each input token embedding is linearly projected into Q, K, V vectors.  
- Shape after splitting: `(batch, seq_len, n_heads, h_dim)`.  

Then, before computing attention, **Rotary Position Embeddings (RoPE)** are applied to Q and K:

```python
q = apply_rotary_embeddings(q, freqs_complex)
k = apply_rotary_embeddings(k, freqs_complex)
```

Why? Token embeddings alone tell the model what a word is, but not where it appears. Without positional info,  

- “The cat sat on the mat.”  
- “The mat sat on the cat.”  

would look identical.

Instead of adding positional vectors (like the original Transformer’s sinusoidal method), RoPE rotates Q and K in the complex plane by an amount proportional to their position. This makes attention directly sensitive to relative distances between tokens.

For our purposes, you can think of RoPE as: “a lightweight operation on Q and K that encodes order, without changing tensor shapes.”

(If you want to dive deeper, check the RoPE paper on arxiv.)


Next, permute the tensors then apply scaled dot-product attention:

```python
q = q.permute(0, 2, 1, 3)
k = k.permute(0, 2, 1, 3)
v = v.permute(0, 2, 1, 3)
        
scores = q @ k.transpose(-2, -1) / sqrt(h_dim)
mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device))
scores = scores.masked_fill(~mask, float('-inf'))
weights = F.softmax(scores, dim=-1)
out = weights @ v
```

- The causal mask ensures each token can only attend to past tokens (left-to-right).  
- Softmax converts similarity scores into weights.  
- Weighted sum with V produces the final attended representation.

Finally, results from all heads are concatenated and passed through a linear projection to mix information.

---

## Notes

1. **Why not Multi-Query Attention?**  
   The original LLaMA-2 paper uses *multi-query attention* (MQA), where all heads share the same K and V but have separate Q.  
   This greatly reduces KV-cache memory usage, which is important for scaling to very large models and efficient inference.  
   For this project, memory pressure from KV-cache isn’t a bottleneck, so standard multi-head attention is simpler and sufficient.

2. **What about DeepSeek’s MLA?**  
   This project includes an optional implementation of **Multi-Head Latent Attention (MLA)**, which is a refinement that reduces KV-cache memory even further while keeping multiple latent spaces.  
   It’s more efficient than MQA, but again — KV-cache isn’t the limiting factor here.  
   Since the focus is educational clarity, SimpleLLaMA sticks with classic multi-head attention.
