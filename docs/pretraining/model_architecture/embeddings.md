# Embeddings

Embeddings are the first step in turning discrete tokens (integers from tokenization) into continuous vectors for a neural network to process.

---

## Token Embeddings

After tokenization, each word or subword is represented by an integer ID. But LLMs don’t work directly with these integers. Instead, we map each token ID into a **dense vector of fixed size** (the embedding dimension).

In PyTorch, this is done with an `nn.Embedding` layer. In the `LLaMaTransformer` class from this project, you’ll see:

```python
self.embeddings = nn.Embedding(tokenizer.get_vocab_size(), n_embd)
```

(Often times the embedding dimensionality of the model, `n_embd` in this case, is referred to under different names. Common ones include `embedding_dim`, `d_model`, and `hidden_size`)

Here, an embedding layer is created, which serves as a lookup table for the model.  
It takes in two primary values, `vocab_size` and `n_embd` and create a matrix of shape `(vocab_size, n_embd)`  
Each row corresponds to a token ID and is a trainable vector. For example:

- Token "I" → 73 → [0.12, -0.44, 1.05, ...]
- Token "an" → 256 → [1.33, 0.05, -0.72, ...]

Both will be map to unique vectors, all of which will be of length `n_embd`
At initialization, these vectors are random. During training, the model adjusts them so that it learns semantic relationship between tokens.  

`n_embd` is a crucial hyperparameter when creating a LLM. It essentially gives the model the flexibility of how much 'semantics' each token can hold.   

For example, say the word `man` and `woman` can be represented by a single token, `1098` and `1290` respectively. 
Passing those through the embedding layer, the model will grab the vector at row index of `1098` to represent that as man, and row index `1290` for woman

Their vectors will differ, but both have shape `(n_embd,)`.
You can think of each dimension in this vector space as encoding some abstract feature the model has learned. For example, one combination of dimensions might capture gender-like differences (man vs. woman), while another might capture whether something is an animate being or an object.  
However this is just a simplified way of explanation. In reality, these dimension are polysemantic and is much more complex.
(Should include explanation that each value is a dimension as well?)

Once we convert our list of tokens into a list of vectors, we can proceed with passing that to the Decoder Block.

---

## Embeddings in This Project

- **Embedding dimension** (`n_embd`) is configurable (e.g., 768, 1024, or higher).  
- **RoPE** is used for positional encoding by default, following LLaMA.  
- **Initialization**: embeddings start random and are updated through backpropagation.  
- **Tied weights**: this project experimented with tying embeddings to the final output projection layer (a trick used in some models). But in practice, training became unstable, so it was disabled.


---

## Walkthrough Example

Let’s walk through a toy example with the sentence:  

**“I love dogs”**  

**Step 1. Tokenization**  

Using an arbitrary *word-level* tokenizer, each word is mapped to an integer ID:  

- `"I"` → 73  
- `"love"` → 786  
- `"dogs"` → 2934  

So the input sequence is:  

```
[73, 786, 2934]
```

---

**Step 2. Embedding Matrix**  

When we create an `nn.Embedding(vocab_size, embedding_dim)` layer, it internally builds a big **lookup table** (a matrix).  

- Shape = `(vocab_size, embedding_dim)`  
- Each row index corresponds to a token ID.  
- Each row contains a vector of length `embedding_dim`.  

For this example, let’s set `embedding_dim = 8`. That means every token ID will be mapped to an **8-dimensional vector**.  

A (very small) portion of the embedding matrix might look like this at initialization (values are random floats):  

```
0:     [ 0.11, -0.07,  0.45,  0.02, -0.33,  0.19, -0.48,  0.05]
1:     [-0.21,  0.34, -0.11, -0.08,  0.27, -0.39,  0.17, -0.43]
2:     [ 0.09,  0.13,  0.28, -0.47, -0.36,  0.22,  0.41, -0.18]
3:     [-0.15,  0.54,  0.28,  0.12, -0.41, -0.41, -0.53,  0.44]
4:     [ 0.12,  0.25, -0.58,  0.56,  0.4,  -0.35, -0.38, -0.38]
5:     [-0.23,  0.03, -0.08, -0.25,  0.13, -0.43, -0.25, -0.16]
...
73:    [ 0.22, -0.51,  0.36,  0.08, -0.44,  0.19, -0.09,  0.27]
...
786:   [-0.13,  0.42,  0.07, -0.36,  0.55, -0.22,  0.18,  0.04]
...
2934:  [ 0.31, -0.14, -0.25,  0.49, -0.07,  0.61, -0.12, -0.33]
...
```

---

**Step 3. Lookup**  

Now, to embed our sentence `[73, 786, 2934]`, the embedding layer simply **selects the rows** at those indices:  

- Token ID **73 (“I”)** → `[ 0.22, -0.51,  0.36,  0.08, -0.44,  0.19, -0.09,  0.27 ]`  
- Token ID **786 (“love”)** → `[ -0.13,  0.42,  0.07, -0.36,  0.55, -0.22,  0.18,  0.04 ]`  
- Token ID **2934 (“dogs”)** → `[ 0.31, -0.14, -0.25,  0.49, -0.07,  0.61, -0.12, -0.33 ]`  

---

**Step 4. Output Tensor**  

Stacking them together, the embedding layer outputs a tensor:  

```
[
  [ 0.22, -0.51,  0.36,  0.08, -0.44,  0.19, -0.09,  0.27 ],   # "I"
  [ -0.13,  0.42,  0.07, -0.36,  0.55, -0.22,  0.18,  0.04 ], # "love"
  [ 0.31, -0.14, -0.25,  0.49, -0.07,  0.61, -0.12, -0.33 ]   # "dogs"
]
```

Shape = `(3, 8)` → 3 tokens, each represented by an 8-dimensional vector.

Essentially, given an input 1d tensor of tokens, which the number of tokens is often referred to as `(seq_len,)`,
we transform it into a tensor of shape `(seq_len, n_embd)` 

In this example, it is `(3, 8)`

This is the format that gets passed on to the Decoder Block. 



A very important note is that there's almost always a third dimension, called a Batch dimension. 
This allows parallel processing, which makes training much faster. 
Batch dimension is always the very first dimension, so the shape of output tensor is `(batch, seq_len, n_embd)`
In this case, since we only have a single example sentence, batch dimension value would be 1, which is

`(1, 3, 8)`




