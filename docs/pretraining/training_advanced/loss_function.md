# Loss Function in Language Modeling

The loss function is the core of how we train language models. It tells the model *how wrong it was* during training, and provides the signal for how to adjust its parameters. In the case of autoregressive language models (like GPT or LLaMA), the standard loss is the **negative log-likelihood (NLL)**, which is equivalent to the **cross-entropy loss**.

---

## The Goal: Modeling the Entire Sequence

When training a language model, we don’t just want it to guess one token correctly when given the entire sentence. That would make it closer to a conditional classifier. Instead, the goal is to train it to **generate full sequences**, step by step, in an autoregressive manner.

Formally, the probability of a sequence of tokens \\(y_1, y_2, \dots, y_N\\) is defined as the product of conditional probabilities:

$$
P(y_1, y_2, \dots, y_N) = \prod_{t=1}^N p(y_t \mid x_{<t})
$$

Here:  
- \\(N\\) = number of tokens in the sequence.  
- \\(y_t\\) = the true token at timestep \\(t\\).  
- \\(x_{<t}\\) = all the tokens before timestep \\(t\\).  
- \\(p(y_t \mid x_{<t})\\) = the probability that the model assigns to the correct token given the context.  

This factorization says: to get the joint probability of the full sequence, the model predicts the first token, then the second token conditioned on the first, then the third conditioned on the first two, and so on. This is why the product shows up: it captures the idea of generating the *entire sequence* autoregressively.

---

## Why Logs?

The per-token negative log-likelihood is:

$$
\mathcal{L}_t = -\log p(y_t \mid x_{<t})
$$

This tells us how much the model is “penalized” for its prediction at timestep \\(t\\). But why take logs in the first place? There are **two main reasons**:

### 1. Turning Products into Sums
The probability of the whole sequence is a product of many terms (each between 0 and 1). For example:

$$
P = p(y_1)p(y_2 \mid y_1)p(y_3 \mid y_1,y_2)...
$$

This product can become *extremely small*. Even if each predicted correct token has a very high probability like ~0.8, for a mere 250 tokens, \\(0.8^{100} \\approx 6 * 10^{-25}\\). That’s practically a value of zero.  

Taking the log turns the product into a sum:

$$
\log P(y_1, \dots, y_N) = \sum_{t=1}^N \log p(y_t \mid x_{<t})
$$

Overall, that's much easier to compute, and sums don’t collapse to near zero the way products do.
Since computers can’t represent extremely tiny numbers well. By taking logs, we keep values in a reasonable numeric range. Instead of multiplying a thousand small decimals, we just add up logs, which are often small negative numbers (like -0.9, -1.2).  

### 2. Better Gradients
The derivative of \\(\\log f(x)\\) is \\(\\frac{1}{f(x)} f'(x)\\). This plays especially nicely with softmax outputs, leading to a simple, stable gradient for training. In fact, the widely used “softmax + cross-entropy loss” is implemented in PyTorch as a single efficient function (`CrossEntropyLoss`) because of this mathematical property.

---

## Example Walkthrough

Take the sentence:  

**“The cat sat on the mat.”**  

Suppose the model’s vocabulary includes tokens like “cat”, “dog”, “mat”, etc. When predicting the final word “mat”, the model outputs probabilities:

- \\(p(\\text{cat} \mid \text{The cat sat on the}) = 0.01\\)  
- \\(p(\\text{dog} \mid …) = 0.03\\)  
- \\(p(\\text{mat} \mid …) = 0.30\\)  

Here, the correct token is “mat,” so the probability we care about is 0.30. The loss for this timestep is:

$$
-\log(0.30) \approx 1.20
$$

If the model had been less confident, say \\(p(\\text{mat})=0.15\\), then the loss would be:

$$
-\log(0.15) \approx 1.90
$$

So the model is assigned a higher loss when it assigns low probability to the correct answer.

---

## Averaging Over Timesteps

We compute this log-loss for **every token in the sequence** and average:

$$
\mathcal{L} = -\frac{1}{N}\sum_{t=1}^N \log p(y_t \mid x_{<t})
$$

This way, the loss reflects how well the model predicts the *entire sequence*, not just one token. Averaging keeps the loss value on a consistent scale, regardless of sequence length.

---

## Connecting to the Training Code

It’s useful to see how the theoretical loss we’ve described translates into the actual training loop. Consider the following code snippet:

```python
x, y = dataset_loader.get_batch()

with torch.autocast(device_type="cuda" if "cuda" in device else "cpu",
                    dtype=torch.bfloat16 if use_amp else torch.float32):
    pred = model_handle(x)         # Forward pass
    B, T, C = pred.shape           # (Batch, Time, Vocabulary size)
    loss = criterion(pred.reshape(B * T, C), y.reshape(B * T))
```

Let’s break this down:

1. **Batch Data**  

    - `x`: input tokens of shape `(B, T)` where `B` = batch size and `T` = sequence length.  
    - `y`: target tokens of shape `(B, T)` — essentially the same sequence but shifted by one token (next-token prediction).  

2. **Model Output**  

    - `pred = model_handle(x)` gives the raw logits (unnormalized scores) for every token in the vocabulary at each position in the sequence.  
    - Shape: `(B, T, C)` where `C` = vocabulary size.  

    Example: if `B=2`, `T=4`, `C=50,000`, then `pred` has predictions for *every position in both sequences*, across the full vocabulary.

3. **Reshaping for the Loss**  

    PyTorch’s `CrossEntropyLoss` expects inputs of shape `(N, C)` and targets of shape `(N,)`, where `N` is the number of training examples.  

    - `pred.reshape(B*T, C)` flattens the batch and sequence dimensions, so we now have `N = B*T` predictions, each over the vocabulary.  
    - `y.reshape(B*T)` flattens the targets into a single vector of token IDs.  

    In other words, each token in the batch is treated as a separate training example.

4. **Loss Calculation**  
    - `criterion` (referring to `torch.nn.CrossEntropyLoss` here) internally applies a log-softmax to `pred` and computes:  

     $$
     \mathcal{L} = - \frac{1}{B \cdot T}\sum_{i=1}^{B \cdot T} \log p(y_i \mid x_{<i})
     $$

    - This is exactly the negative log-likelihood we’ve been discussing, but implemented efficiently.  

---

## Summary

- We want to train models to generate *entire sequences*, not just single tokens. This leads to the product of conditional probabilities.  
- Taking logs converts the product into a sum, avoids numerical underflow, and gives clean gradients.  
- The negative average log-likelihood (cross-entropy loss) is the standard loss function for training LLMs.  

