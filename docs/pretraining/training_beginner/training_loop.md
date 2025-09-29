
Training a large language model (LLM), even a small‑scale one like in this project, comes down to a repeated cycle: take a batch of data, run it through the model, calculate how wrong the predictions are, push the error backwards to update the weights, and repeat. This cycle is what we call the **training loop**. This section will discuss in detail through the core parts of the loop. 

---

### Instantiation

Before we can train the model, we need to set up all the core building blocks. Once everything is in place, the training loop itself becomes a straightforward repetition of forward pass, loss calculation, backward pass, and optimization.

---

**1. Configuration Object**

The first thing we need is a configuration object that stores all of our hyperparameters. Instead of scattering values like batch size, learning rate, and number of layers across different files, it’s cleaner to place them in a single file/class. This makes the code easier to manage, debug, and extend.  

In this project, it will be the `TrainingConfig` class, located within the `simple_llama/pretraining/config.py` file

```python
@dataclass
class TrainingConfig:
    # === Paths and Dataset ===
    dataset_dir: str = root_path("simple_llama", "dataset", "short")       # Path to tokenized training data
    tokenizer_path: str = root_path("simple_llama", "dataset", "bpe_8k.json")          # Path to tokenizer model
    ckpt_dir: str = root_path("simple_llama", "pretraining", "checkpoints")   # Directory to store checkpoints
    log_file: str = root_path("simple_llama", "pretraining", "training_progress.txt")  # File to log training progress

    # === Batch & Sequence ===
    batch_size: int = 4             # Minibatch size
    max_seq_len: int = 2048         # Maximum sequence length per sample
    tokens_per_update: int = 2**19  # ~512K tokens per optimizer update

    # === Model Architecture ===
    n_embd: int = 2048               # Embedding dimension
    n_heads: int = 32                # Number of attention heads
    n_layers: int = 24               # Number of transformer layers
    multiple_of: int = 256           # Feedforward dim multiple for efficient matmul
    eps: float = 1e-5                # Epsilon value to prevent div-by-zero in normalization layers
    theta: int = 10_000              # Theta for RoPE rotation frequency
    dropout: float = 0.0             # Dropout rate; typically 0.0 for pretraining

    ...  # And many more

config = TrainingConfig()
```

This way, if we want to adjust hyperparameters like `n_heads` or experiment with a different `max_lr`, it’s a single line change.

---

**2. Dataset Loader**

Next, instantiate a dataset loader object that is defined, passing in hyperparameters as needed, extracted from the configuration object:

```python
from simple_llama.pretraining.dataset_loader import DatasetLoader

dataset_loader = DatasetLoader(batch=batch_size, seq_len=max_seq_len, 
                               process_rank=ddp_rank, num_processes=ddp_world_size, 
                               dataset_dir=config.dataset_dir, device=device)
```

---

**3. The Model**

Now comes the centerpiece: the transformer model itself. In this project, we’ve implemented `LLaMaTransformer`, which includes embeddings, attention blocks, feedforward layers, normalization, and output projection.

```python
model = LLaMaTransformer(config, tokenizer, device="cuda")
```

Here:  

- `config` gives the model hyperparameters.  
- `tokenizer` provides the vocabulary size.  
- `device="cuda"` places the model on GPU.

Initially, the model’s parameters are random. Training gradually adjusts them so that token predictions become more accurate.

---

**4. The Loss Function**

Next, we define how the model will be judged. For language modeling, the go-to choice is **cross-entropy loss**:

```python
criterion = torch.nn.CrossEntropyLoss()
```

Cross-entropy measures how “surprised” the model is by the correct next token.  

- If the model assigns high probability → low loss.  
- If it assigns low probability → high loss.

---

**5. The Optimizer**

Finally, we define the optimizer. We use **AdamW**, which is the de facto standard for transformers because it combines Adam’s adaptive gradient updates with weight decay for stability.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(beta1, beta2), weight_decay=weight_decay, **extra_args)
```

This way, every training step will use the optimizer to update the model parameters and the scheduler to adjust the learning rate.

(More about AdamW optimizer will be covered in the Advanced section of the documentation)

---

**6. Learning Rate Scheduler**

Finally, the **learning rate scheduler**. The scheduler controls how the learning rate evolves over time, which is important for training.

We’re using the custom `Scheduler` class implemented earlier, which supports *linear decay*, *cosine decay*, or just a *constant* learning rate.

```python
scheduler = Scheduler(torch_optimizer=optimizer,
                      schedule="cosine",
                      training_steps=optimization_steps,
                      warmup_steps=warmup_iterations,
                      max_lr=max_lr,
                      min_lr=min_lr)
```

---

At this point, we’ve instantiated:  

- Configuration object
- Dataset loader
- Transformer model
- Loss function
- Optimizer
- Learning rate scheduler

All the main components are ready. The next step is to actually run them inside the training loop.

---

### The Model Forward Pass

We begin with a batch of input tokens, grabbed from the DatasetLoader object via the `get_batch()` method. Each integer corresponds to a token ID from our vocabulary.  

Let’s say our batch size is `B = 4`, and the sequence length we train on is `T = 16`. The shape of a batch from the dataset loader would look like:

```
x.shape = (B, T) = (4, 16)
```

So `x` is a 2D tensor of integers. Each row is one training sequence, and each entry is a token ID.  

When we feed this into the model:

```python
logits = model(x)
```

the transformer runs all of its layers: embedding lookup, multiple decoder blocks, attention, feedforward layers, normalization, and finally a linear projection back to vocabulary size.  

The key here is the shape change:  

- Input: `(B, T)` — integers.  
- Output: `(B, T, C)` — floats, where `C` is the vocabulary size.  

Why `(B, T, C)`? Because for every position in every sequence, the model outputs a vector of size `C`, which are the raw unnormalized scores for each possible token in the vocabulary. These are called **logits**.

---

### The Loss Function

Once we have logits, we want to measure how good the predictions are. That is the role of the **loss function**. For language modeling, the standard is **cross entropy loss**.

The goal is simple: the model is asked to predict the next token in the sequence. If the input sequence is `[The, cat, sat, on, the]`, the correct output is `[cat, sat, on, the, mat]`. Each token should map to the next token.  

Cross entropy measures how “surprised” the model is by the correct answer. If the model already places high probability on the true next token, the loss is small. If the model thought another token was much more likely, the loss is large.  

In PyTorch, we use:

```python
criterion = nn.CrossEntropyLoss()
```

However, `CrossEntropyLoss` expects inputs of shape `(N, C)` where `N` is the number of items and `C` is the number of classes, and targets of shape `(N,)`.  

Our logits are `(B, T, C)` and our targets are `(B, T)`. So we flatten them:

```python
loss = criterion(logits.view(-1, C), targets.view(-1))
```

This reshapes:

- `logits.view(-1, C)` → `(B*T, C)`  
- `targets.view(-1)` → `(B*T,)`  

Effectively, we treat the whole batch as one big list of token predictions.

Mathematically, cross entropy loss is:

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \log \big( \text{softmax}(\text{logits})_{i, y_i} \big)
$$


where `y_i` is the true class (the correct next token).  

More details will be covered in the Advanced Training section

---

### The Backward Pass

Now comes the critical part: telling the model how wrong it was. This is done with:

```python
loss.backward()
```

This triggers PyTorch’s **autograd engine**, which walks backwards through the computational graph.  

Every tensor operation in PyTorch (matrix multiplies, nonlinearities, normalizations) records how it was computed. During `.backward()`, PyTorch applies the chain rule of calculus to compute gradients of the loss with respect to every parameter in the model.

So, if our model has parameters θ = {W1, W2, …}, then after `loss.backward()` we now have stored gradients ∂L/∂W for each parameter. These gradients are stored in each parameter tensor within the `.grad` attribute, which is a matrix of gradients the shape as the weight matrix. 

These gradients tell us: “If you nudge this weight slightly, the loss would go up/down this much.” 

Effectively, they are the signals that will guide weight updates.

---

### The Optimizer Step

With gradients calculated, we actually update the weights. This is the job of the optimizer.  

In this project, we use **AdamW**:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
```

AdamW is a variant of stochastic gradient descent that adapts learning rates per parameter and includes proper weight decay. It’s widely used in training transformers.

The update cycle is:

```python
optimizer.zero_grad(set_to_none=True)  # reset gradients

# Between these two steps, perform forward pass, calculate loss, back propagation

optimizer.step()       # update parameters using gradients
```

Zeroing the gradients is crucial because PyTorch accumulates gradients by default. If we didn’t zero them, gradients from multiple steps would pile up, leading to artificially large parameter updates, quickly destabilizing the model.

So the full cycle is:

1. Zero gradients → prepare for next step.
2. Forward pass → compute logits and loss.
3. Loss calculation → use criterion to calculate loss.
4. Backward pass → compute gradients.  
5. Optimizer step → update weights.  

---

### A Minimal Training Loop

Putting everything together:

```python
for step in range(num_steps):
    # Get a batch
    x, y = dataset_loader.get_batch()   # x: (B, T), y: (B, T)

    # Forward pass
    logits = model(x)                   # (B, T, C)

    # Compute loss
    loss = criterion(logits.view(-1, C), y.view(-1))

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Update
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
```

Granted the actual implementation in `simple_llama.pretraining.train` file is much more complex, however this is the backbone of training. Every sophisticated training pipeline — from GPT to LLaMA — reduces to these few lines.  


---

### Evaluation and Monitoring

Training is only half the story. We need to know if the model is improving. The simplest way is to track the **training loss**. Over time, as the model sees more data, loss should decrease, which means the model is getting progressively better at predicting the next token, given an input sequence of tokens.

At the very beginning, before the model has learned anything meaningful, predictions are essentially random. In this case, the expected loss can be approximated by the natural logarithm of the vocabulary size, since each token is equally likely under a uniform distribution.  

For our project, the vocabulary size is 8192. So if the predictions were truly uniform, the expected initial loss would be:

```
ln(8192) ≈ 9.01
```

However, in practice, most parameters in the model (such as linear layers) are initialized from Gaussian distributions using Kaiming or Xavier initialization. This breaks the perfect uniformity and introduces biases. As a result, the observed loss at the very start of training would likely be higher than the theoretical value — for example, around 9.2 or 9.3 instead of exactly 9.01.  

---

**Why the log of vocab size?**

Cross-Entropy Loss (CEL) is essentially a Negative Log Likelihood (NLL) loss. For a dataset of size \(N\) with true labels \(y_i\) and predicted probabilities \(p(y_i)\):

$$
CEL = -\frac{1}{N} \sum_{i=1}^{N} \log p(y_i)
$$

For a single example where the true class is \(c\):

$$
CEL = -\log p(c)
$$

If the model predicts uniformly over all \(V\) classes, then \(p(c) = \frac{1}{V}\). Plugging this in:

$$
CEL = -\log \left(\frac{1}{V}\right) = \log V
$$

So under uniform predictions, the expected loss equals the log of vocabulary size.

**Example:**  

- \(V = 8192\)  
- \(CEL = \log(8192)\)  
- \(CEL \approx 9.01\)  

This is the theoretical baseline for random guessing. In practice, initialization bias may push it to ~9.3 at step 0.

---

**Training Dynamics**

As training continues, the loss should decrease steadily. For instance, a drop from ~9.3 to ~3 means the model is learning meaningful statistical patterns in the data. Lower loss translates directly into the model being less “surprised” when predicting the next token.

Think of it this way:  

- At loss ≈ 9, the model is basically clueless, assigning ~1/8192 probability to every token.  
- At loss ≈ 3, the model assigns ~1/20 probability to the correct token on average.  
- At loss ≈ 1, the model is strongly confident, giving ~1/3 probability to the correct token.

Even at a loss of around 3.0, the model assigns roughly 5% probability to the observed “correct” token on average. That may sound low if one interpret it as "The model only have a 5% chance of choosing the correct token, for a given sequence"
However that is a bit misleading. In English (or just about all languages) there is natural entropy to it. Vast majority of the time, there are multiple valid answers to a given sequence.  

Taking the previous example, we give the model: `[The, cat, sat, on, the]` and want it to predict the next token. Our true label should be the token corresponding to the word `mat` however, in general, there isn't just a single right-wrong answer. 
Words like `floor`, `ground`, `couch` and such are also completely valid. Hence a probability of 1/20 chance choosing the 'correct' token isn't as bad a it may numerically seem to be. 

---

**Validation?**

It’s also common to periodically **evaluate** on a held-out validation set. This prevents overfitting, since training loss always decreases but validation loss may rise if the model memorizes.  

However, in this project, no validation set is used. Why? Because the dataset (50B tokens gathered from FineWebEdu) is mostly unique. Training is done in a single epoch — the model will only see each token sequence once. Under this regime, overfitting is theoretically impossible.  

In fact, if a model with ~1B parameters *were* able to fully overfit on 50B unique tokens, that would be remarkable — it would essentially mean the model is acting as a form of near-lossless compression of the dataset. From that perspective, it might even be considered desirable. But in practice, that's nearly impossible. Here we will only go through one pass using the 50B tokens, simply track training loss as the main signal of progress.

---

**A Tiny Generation Example**

Even early in training, it’s fun (and useful) to test the model by generating text.  

We take a prompt, tokenize it, and call a generate function:

```python
print(model.generate("Once upon a time", max_new_tokens=20))
```

At the start, the output will be nonsense — the model has learned almost nothing. But as loss decreases, generated samples gradually improve. They start forming grammatical sentences, then coherent paragraphs.

This qualitative check is as important as loss curves, because it directly shows what the model is learning.


Here is an example output log print when training a small model:

```text
Step: 256 steps   |   Training Progress: 0.02%   |   Training Loss: 8.0160   |   Perplexity: 3029.09   |   Learning Rate: 0.00008   |   Norm: 1.0915   |   Tokens Processed: 8M (8M)   |   tok/s: 157961   |   Time: 53s
----------------
Step: 512 steps   |   Training Progress: 0.04%   |   Training Loss: 7.0701   |   Perplexity: 1176.23   |   Learning Rate: 0.00015   |   Norm: 0.2549   |   Tokens Processed: 16M (16M)   |   tok/s: 142851   |   Time: 58s
----------------
----------------
Step: 768 steps   |   Training Progress: 0.06%   |   Training Loss: 6.5323   |   Perplexity: 686.96   |   Learning Rate: 0.00023   |   Norm: 0.1649   |   Tokens Processed: 25M (25M)   |   tok/s: 187962   |   Time: 44s
----------------
----------------
Step: 1024 steps   |   Training Progress: 0.07%   |   Training Loss: 5.8950   |   Perplexity: 363.23   |   Learning Rate: 0.00031   |   Norm: 0.2274   |   Tokens Processed: 33M (33M)   |   tok/s: 187884   |   Time: 44s
----------------
----------------
Step: 1280 steps   |   Training Progress: 0.09%   |   Training Loss: 5.6318   |   Perplexity: 279.16   |   Learning Rate: 0.00038   |   Norm: 0.2636   |   Tokens Processed: 41M (41M)   |   tok/s: 187881   |   Time: 44s
----------------
----------------
Step: 1536 steps   |   Training Progress: 0.11%   |   Training Loss: 5.2796   |   Perplexity: 196.28   |   Learning Rate: 0.00046   |   Norm: 0.2596   |   Tokens Processed: 50M (50M)   |   tok/s: 187936   |   Time: 44s
----------------
----------------
Step: 1792 steps   |   Training Progress: 0.13%   |   Training Loss: 5.1112   |   Perplexity: 165.87   |   Learning Rate: 0.00054   |   Norm: 0.2780   |   Tokens Processed: 58M (58M)   |   tok/s: 187930   |   Time: 44s
----------------
----------------
Step: 2048 steps   |   Training Progress: 0.15%   |   Training Loss: 5.0083   |   Perplexity: 149.65   |   Learning Rate: 0.00060   |   Norm: 0.4782   |   Tokens Processed: 67M (67M)   |   tok/s: 188105   |   Time: 44s
----------------
----------------
Step: 2304 steps   |   Training Progress: 0.17%   |   Training Loss: 4.7851   |   Perplexity: 119.71   |   Learning Rate: 0.00060   |   Norm: 0.3195   |   Tokens Processed: 75M (75M)   |   tok/s: 188087   |   Time: 44s
----------------
----------------
Step: 2560 steps   |   Training Progress: 0.19%   |   Training Loss: 4.6413   |   Perplexity: 103.68   |   Learning Rate: 0.00060   |   Norm: 0.4755   |   Tokens Processed: 83M (83M)   |   tok/s: 187940   |   Time: 44s
----------------
----------------
Step: 2816 steps   |   Training Progress: 0.21%   |   Training Loss: 4.6120   |   Perplexity: 100.68   |   Learning Rate: 0.00060   |   Norm: 0.2608   |   Tokens Processed: 92M (92M)   |   tok/s: 187844   |   Time: 44s
----------------
----------------
Step: 3072 steps   |   Training Progress: 0.22%   |   Training Loss: 4.4724   |   Perplexity: 87.57   |   Learning Rate: 0.00060   |   Norm: 0.6141   |   Tokens Processed: 100M (100M)   |   tok/s: 187720   |   Time: 44s
```

Focusing on the training loss curve, we can see that it declines rapidly before slowly starts to plateau a bit. 
Most metrics is quite simple (Should add some explanation about norms/perplexity?)  


Here is an example generation output by the model at the very start, after just 256 steps:

```text
<SOS> Classify the sentiment as "Positive" or "Negative":
Input: I absolutely loved the movie, it was fantastic!
Sentiment: Positive

Input: The service was terrible and I won't come back.
Sentiment: Negative

Input: This book was boring and hard to finish.
Sentiment: Negative

Input: The app keeps crashing and it's really frustrating.
Sentiment: Negative

Input: I'm impressed by how fast and easy this process was.
Sentiment: Positive

Input: The food was delicious and the atmosphere was wonderful.
Sentiment: told preventing capac appoint aboutterfacesstandingomeotheods char AcuseAAes applications governments Theseind energy
be electroesietThese Discussteries contains spendasma critaries treatels 190 facilitaintically majority might 13 calculate honey Colle robot Orony soils Fin dest confirmed7 financialcom. highest Denheastoet rec branch
```


In the above example, the provided text sequence ended at:

```text
Input: The food was delicious and the atmosphere was wonderful.
Sentiment: 
```

Where after the 'Sentiment: ' part, we can see that the output is garbled, as we would expect. 
However, at the next evaluation, the following resulted from the model:

```text
<SOS> Write a polite email reply:
Input: Hi, can you send me the report by tomorrow? Thanks.
Reply: Hi, sure thing! I'll send the report to you by tomorrow. Let me know if you need anything else.

Input: Hi, just checking if you're available for a meeting this Friday.
Reply: Hi, thanks for reaching out. I'm available on Friday - what time works best for you?

Input: Hi, could you help me with the project deadline?
Reply: Hi, of course. Let me know what you need help with, and I'll do my best to assist.

Input: Hi, do you have the updated slides for the presentation?
Reply: Hi, yes - I'll send over the updated slides shortly. Let me know if you'd like me to walk you through them.

Input: Hi, do you have time to meet with the manager later today?
Reply: How do I have no force?
Learn't have an error in good, and I am looking for a call while you should check.
Use students learn about three different environments, including:
Reportness (such as CCM)
You can use a sample is
Click
```


Here, the provided text sequence is different (to provide variations during evaluations)

In the above example, the provided text sequence ended at: 

```text
Input: Hi, do you have time to meet with the manager later today?
Reply: 
```

The model is clearly learning valid words and starts to piece them together in a somewhat structured way.

---

### Bringing It All Together

To summarize, each training step does:

1. Take a batch `(B, T)` of token IDs.  
2. Run through model → get logits `(B, T, C)`.  
3. Compute cross entropy loss with targets `(B, T)`.  
4. Backpropagate loss → compute gradients.  
5. Optimizer updates weights.  
6. Zero gradients.  

This loop runs millions of times. At small scale, it might be just tens of thousands of steps. At large scale (GPT‑3, LLaMA), training can take trillions of tokens.

But the essence is always the same. The beauty of the transformer is that all of this complexity — embeddings, attention, normalization, feedforward layers — reduces down to the training loop you’ve just seen.

---


