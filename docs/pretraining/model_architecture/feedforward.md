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

In the original Transformer paper (Vaswani et al., 2017), the FFN was simply:  

```
FFN(x) = max(0, xW1 + b1)W2 + b2
```

- Two linear layers with a ReLU in between.  
- The hidden dimension is usually set to **4× the embedding dimension**, then projected back down.  

If embedding size = 1024, the FFN hidden size = 4096.  

This “expand and contract” pattern gives the model lots of nonlinear mixing power.  


Looking at a more concrete implementation, the feedforward class might be defined as:

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.layer1 = nn.Linear(n_embd, 4 * n_embd),
        self.layer2 = nn.Linear(4 * n_embd, n_embd),

    def forward(self, x: torch.tensor):
        return self.layer2(torch.nn.functional.relu(self.layer1(x)))
```

Recall that we pass an input tensor `x` into the attention sublayer, after applying the attention mechanism, the output tensor remains the same shape, `(batch, seq_len, n_embd)`  
That will then get passed into feedforward sublayer (after prenorm, but shape remains the same)

So in the constructor, we define the two linear layers. 
The first one takes in a tensor, where the last dimension is of shape `n_embd`, projects it into `4 * n_embd` space. 
The second layer takes in a tensor, expecting the last dimension to be of shape `4 * n_embd` and projects it back down into `n_embd` 

In the forward function, we can see that given the input tensor x, we apply those two layers, with a `relu` function sandwiched in between to introduce non-linearity. 

---

## LLaMA-Style FFN (SwiGLU)

The LLaMA architecture made one key modification: instead of ReLU, it uses a **SwiGLU activation**.  

In this project, this is implemented in the `FeedForward` class:

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd: int, multiple_of: int, dropout: float):
        super().__init__()

        hidden_dim = int(4 * n_embd * (2 / 3))  # Authors of LLaMa used 2/3 of 4*n_embd (To distribute num params)
        # Rather than being 1024*4 * (2/3) = 2730, using multiple_of with a value base 2 functions better (e.g. 64, 128, or 256)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        # Element-wise multiplication between two matrices of shape (batch, seq_len, hidden_dim)
        # along with silu instead of relu activation function
        h = F.silu(self.w1(x)) * self.w3(x)
        return self.dropout(self.w2(h))
```

Breaking this down:  

The authors created a variation where they wanted to introduce a gating function, hence the usage of 3 linear layers compared to the traditional 2 layers in FFN. 

This formulation is known as a **SwiGLU activation** (SiLU-Gated Linear Unit).  
In other words:

\[
h = \text{SiLU}(W_1x) \odot W_3x
\]

is the SwiGLU block, which replaces the ReLU in the vanilla Transformer FFN.  

However simply adding in a third linear layer for gating would just increase the number of parameters in the model. To keep the model size approximately the same for comparison, they split the distribution of parameters evenly among the three layers. 

Recall that in the original FFN implmentation, there are two layers. 
They have weights of shape `(n_embd, 4 * n_embd)` and `(4 * n_embd, n_embd)`
Combined, the total number of parameters they hold is `8 * n_embd**2`

By calculating a hidden dimension variable using `hidden_dim = int(4 * n_embd * (2 / 3))`
That evenly splits the distribution into 3 parts. 
Then initialize the layers where both `w1` and `w3` are of shape `(n_embd, hidden_dim)` and `w2` of shape `(hidden_dim, n_embd)`

Overall parameter count would be `3 * n_embd * hidden_dim`
Substituting in hidden_dim from the above formula would yield `8 * n_embd**2` parameters, which is the same as the original.


Moving on, let's take a look at what happens when a tensor x, of shape `(batch, seq_len, n_embd)` is passed to this FFN block

First, it will pass through `self.w1` layer, transforming it into `(batch, seq_len, hidden_dim)`
Then apply a SiLU activation function to that, which squashes the range of possible values from [-inf, inf] to approximately [-0.3, inf], followed by a hadamard product with `self.w3(x)`
This is a gating function where essentially the `self.w3` layer determines how much 'signal' can pass by.

Imagine for a certain value in `self.w3(x)` is close to 0, that mean the corresponding value in `self.w1(x)` gets suppressed to near 0 as well, whereas if the value from w3 is a large value, it will amplify correspondingly. 
As it also allow negative values, it can also flip value as needed, allowing rich representations. 

Finally, the resulting tensor gets passed through `self.w2(x)` which transforms the tensor from shape `(batch, seq_len, hidden_dim)` back to `(batch, seq_len, n_embd)`


This trick balances parameter count and computational efficiency without hurting performance much.  


Note that there's an intermediate step of calculating the hidden dimension using the input variable `multiple_of`
This is used to round-up the `hidden_dim` value up to the nearest multiple of `multiple_of` value. 

For example, say we have `n_embd=1024`
Using the formula we would get `hidden_dim = int(4 * n_embd * (2 / 3)) = 2730`
Which has a pitiful amount of powers of two, in fact, it only has a single factor of 2 contained within. 

By setting `multiple_of` hyperparameter to a power of 2 like 64, 128, 256, etc., the hidden dimension value would be a 'nicer' number per-say
The reason why these 'nicer' numbers are better is because it allows more efficient GPU utilization. 

Take the previous example, if we set `multiple_of` to get 128, then hidden dimension would turn into `2816`, which is a much nicer number with high numbers of powers of two contained within. 


Finally, there's an additional dropout layer at the end, which is only really used in the SFT/RLHF step, not of particular importance here.  
It prevents overfitting and improves generalization, especially useful in smaller-scale dataset when training. 

---

## Summary

- The feedforward block provides **nonlinear transformation capacity**, complementing attention.  
- Vanilla Transformers used Linear → ReLU → Linear.  
- LLaMA use **SwiGLU-style FFN** for better performance.  
- Most of the parameters in the model live in these FFNs.  
- Without them, the Transformer would collapse into a linear model and lose expressive power.
