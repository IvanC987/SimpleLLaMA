# Utilities for SFT  

## Tokenize and Pad Data  

The next step after formatting the training examples is the transformation of raw `(x, y)` string pairs into properly padded tensors. In this project, the function `tokenize_and_pad_data` is responsible for doing exactly that.  

At first glance, the function may look overwhelming: it includes several layers of tokenization, checks, and padding strategies. But once broken down into its main stages, the logic becomes much clearer.  

Let’s start by looking at the function signature and its purpose:  

```python
def tokenize_and_pad_data(batch: list[tuple], tokenizer: Tokenizer, pad_id: int, max_seq_len: int, dynamic: bool, device: str):
    """
    Tokenizes and pads a batch of (x, y) training pairs for SFT.

    Each element in `batch` is a tuple:
        - x: full prompt including template, user queries, and assistant responses
        - y: only the final assistant response to supervise with loss

    The x sequence is right-truncated to `max_seq_len`.
    The y sequence is left-padded so it aligns with the end of x.
    Left-padded values in y are filled with `pad_id` and are ignored in loss via `ignore_index`.
    """
```

At its core, the function takes in a list of `(x, y)` examples (batch)  where:  

- `x` is the **full formatted training prompt** (system + user + assistant conversation).  
- `y` is the **target assistant response only** (with `<EOA>` at the end).  

It also takes in the tokenizer, to convert the x-y strings into token ids, pad_id corresponds to the token id for a pad token, `max_seq_len`, the hyper parameter that controls the maximum sequence length allowed, dynamic is a bool meaning if to dynamically pad based on longest example or all up to max_seq_len, device is usually cpu or cuda
Don't worry too much about the other inputs for now, just note the most important inputs is the batch, tokenizer, pad_id and max_seq_len
Further details below. 


The goal is to produce two tensors:  

1. `x_tensor`: tokenized, padded input prompts.  
2. `y_tensor`: tokenized, padded targets aligned to `x`, where padding positions are filled with `pad_id` so the loss function ignores them.  

---


### Step 1: Tokenization and Padding

```python
assert len(batch) != 0, "Given batch data should not be empty!"

x_data, y_data = [], []
max_len = 0  # Maximum tensor len
exceed_len = 0
for example in batch:
    x, y = example  # Unpack values

    # Tokenize x-y pair
    x = tokenizer.encode(x).ids
    y = tokenizer.encode(y).ids

    if len(x) > max_seq_len:  # max_seq_len is inclusive of context, so x shouldn't be >= that
        exceed_len += 1
        continue

    max_len = max(max_len, len(x))
    x_data.append(torch.tensor(x, dtype=torch.long, device=device))

    y = torch.tensor(y, dtype=torch.long, device=device)
    num_left_pad = len(x) - len(y) - 1  # Need an additional right pad later on for left-shift

    if num_left_pad < 0:
        warnings.warn(f"Target response longer than input window. Skipping.")
        continue

    y_left_pad = torch.full((num_left_pad,), pad_id, device=device)
    y_data.append(torch.cat((y_left_pad, y), dim=-1))
```

First, ensure that the given batch is not empty.  
Initialize two empty lists, `x_data` and `y_data`, which will later hold the two resulting x-y tensors.  

Next, the two variables `max_len` and `exceed_len` are initialized. `max_len` keeps track of the length of the longest sequence of tokens seen in the current batch, while `exceed_len` records how many training examples surpassed the maximum allowed sequence length.  

Now we iterate through the batch:

> `for example in batch:`

Where each `example` is the `(x, y)` string pair.  

Both `x` and `y` are transformed into lists of integers, each representing tokens from the vocabulary.  

We now check if the number of tokens in tokenized string `x` is greater than `max_seq_len`. If so, we discard that example. Truncation could be done instead, but that’s a separate concern. For now, the goal is to prevent prompts from overflowing the model’s context window. If discarded, increment `exceed_len` by 1. (Though the naming could be improved, it represents the number of examples that exceeded `max_seq_len`.)  

If it does not exceed `max_seq_len`, then continue by updating `max_len` with:  

> `max_len = max(max_len, len(x))`

Here, `max_seq_len` refers to the fixed hyperparameter (e.g. 2048 or 4096 tokens), while `max_len` is the longest sequence of input tokens seen so far in this batch. The former filters out overly long examples, the latter is used to determine how much padding is needed later.  

Then, transform `x` into a tensor and append it to `x_data`. This will be the input tensor to the model as one example sequence.  

For `y`, it’s more nuanced. We cannot simply tokenize and append because we need alignment for the loss function. First, calculate the amount of left padding required, accounting for a shift (`-1`). Then create a left padding tensor, and concatenate it with `y`. Finally, append the result to `y_data`.  

---

Let’s illustrate with a concrete example. Suppose we have an `(x, y)` pair where:  

**x:**  

```
<SOU>What is 2+2?<EOU>
<SOA>4<EOA>
```

**y:**  

```
4<EOA>
```

(We omit the system prompt here for brevity.)  

Tokenized, `x` might look like:  
`[4, 4909, 4862, 981, 3172, 981, 2356, 5, 6, 411, 7]`  

And `y`:  
`[411, 7]`  

Here, `411` corresponds to token “4” and `7` corresponds to `<EOA>`. Importantly, `y` is always the suffix of `x`.  

For `x`, we immediately convert it into a tensor and append it to `x_data`.  

For `y`, convert to a tensor and compute:  

`num_left_pad = len(x) - len(y) - 1 = 11 - 2 - 1 = 8`  

Then create:  

`y_left_pad = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2])`  

(assuming `pad_id = 2`).  

Concatenate with `y`:  

`torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 411, 7])`  

Append this to `y_data`.  

Unrolling side by side:  

```
x: [4, 4909, 4862, 981, 3172, 981, 2356, 5,   6, 411, 7]
y: [2,    2,    2,   2,    2,   2,    2, 2, 411,   7]
```

Notice that the targets are padding until we hit token id `411`. This design ensures the model is not trained to predict the system prompt or the user queries — only the assistant’s outputs.  

When the model sees:  

```
<SOU>What is 2+2?<EOU>
<SOA>
```  

It is expected to predict token `411` (the number 4). Then, given the sequence with `411` included:  

```
<SOU>What is 2+2?<EOU>
<SOA>4
```

The model should predict `<EOA>` (`7`).  

Thus, this remains next-token prediction, but now constrained to assistant responses. Instead of uncontrolled continuation, the model learns structured QA behavior.  

Do note the if conditional block where `if num_left_pad < 0`, meaning the target response is longer than the input, the example is malformed and skipped. This normally shouldn't occur, since `y` is expected to be a suffix of `x`, but added as a safety check. 

---

### Step 2: Validation Check

After populating `x_data` and `y_data`, the function performs a quick check:

```python
# return x_data, y_data
assert len(x_data) != 0, f"All examples has been skipped due to all chat conversations exceeding {max_seq_len=}"
if exceed_len/len(batch) >= 0.1:
    warnings.warn(f"{100 * exceed_len/len(batch):.2f}% of examples in this batch has been skipped due to assistant responses exceeding {max_seq_len=}")

max_len = max_len if dynamic else max_seq_len
```

The first assertion ensures that the dataset did not collapse entirely. If `len(x_data)` is zero, it means that *all* of the examples in the batch were skipped, usually because they exceeded the maximum allowed sequence length (`max_seq_len`). This would be a problematic error, since the model would have no valid training examples to work with, and so the assertion is a necessary safeguard.

The second check introduces a soft warning mechanism. It calculates the ratio of discarded examples (`exceed_len / len(batch)`) and, if at least 10% of the examples in the batch were rejected, issues a warning. This does not stop execution but serves as a red flag: your dataset or chosen hyperparameters may be suboptimal, and the model could be missing out on significant amounts of training data.

Finally, `max_len` is updated. If the `dynamic` flag is set to `True`, the function uses the maximum length observed in the current batch. This keeps padding minimal, ensuring better efficiency since padding consumes memory and compute without adding training signal. However, if `dynamic` is `False`, the function defaults to the global `max_seq_len`, which ensures all batches are consistently padded to the same fixed length. This option is particularly useful when stress-testing hardware capacity, making sure everything fits into memory.

---

### Step 3: Return Tensors

Finally, the function decides how to pad `x_data` and `y_data` to a consistent length:  

```python
x_data = torch.stack([
    torch.concat((x, torch.full((max_len - len(x),), pad_id, device=device)), dim=-1)
    for x in x_data
])

y_data = torch.stack([
    torch.concat((y, torch.full((max_len - len(y),), pad_id, device=device)), dim=-1)
    for y in y_data
])

assert x_data.shape == y_data.shape
assert len(x_data.shape) == 2
return x_data, y_data
```

---

Let’s break it down step by step.  

**Padding the Input (`x_data`)**:

At this point, `x_data` is a list of 1D tensors, each tensor corresponding to the tokenized version of a single input string. Since these strings will almost always be of different lengths, we cannot directly batch them together into a single 2D tensor. PyTorch requires all rows in a tensor to have the same length (no ragged tensors). To fix this, we add **padding tokens** so that every example reaches the same length.  

This line handles it:  

```python
torch.concat((x, torch.full((max_len - len(x),), pad_id, device=device)), dim=-1)
```

Here,  

- `x` is the original sequence of token IDs.  
- `torch.full((max_len - len(x),), pad_id, device=device)` creates a 1D tensor filled with the padding token ID (`pad_id`), with just enough elements to extend `x` to `max_len`.  
- `torch.concat(...)` joins the original sequence `x` with the padding tensor along the last dimension, effectively extending `x` to the full `max_len`.  

After this, each input sequence has exactly the same length, ensuring they can all be stacked together.  

**Padding the Targets (`y_data`)**

The process for `y_data` is nearly identical. Each target sequence is extended with padding tokens until it also reaches `max_len`. One subtle difference is that `y_data` already had **left padding** applied in Step 1 to align assistant responses with the correct positions in the input. This step adds any **right padding** that is still needed so that `y_data` fully matches the shape of `x_data`.  

This ensures both `x_data` and `y_data` align perfectly in shape, with every row corresponding to one example in the batch.  

**Stacking into Final Tensors**

Finally, both padded lists are wrapped in `torch.stack([...])`.  
This function takes a list of tensors of identical shape and combines them into a single tensor by adding a new leading dimension. In this case, that leading dimension corresponds to the **batch size**.  

- After stacking, `x_data` has shape `(batch_size, max_len)`.  
- Similarly, `y_data` has shape `(batch_size, max_len)`.  

This step transforms our batch from a *list of variable-length tensors* into two consistent 2D tensors ready for model training.  



Before returning, the function runs two assertions:  

```python
assert x_data.shape == y_data.shape
assert len(x_data.shape) == 2
```

To make sure that inputs and targets are aligned and that the final tensors are indeed two-dimensional, with one axis for the batch and one for the sequence length.  

If either check fails, it indicates a problem in preprocessing that must be fixed before training continues.

The function then returns the pair `(x_data, y_data)`, now guaranteed to be properly padded, aligned, and in a form suitable for direct use in the model’s forward pass.  

---

## Model Evaluation

The `eval_model` function provides the evaluation mechanism for measuring how well the model performs in generating assistant responses to user inputs during supervised fine-tuning. Here's the function signature: 

```python
@torch.no_grad()
def eval_model(model: LLaMaTransformer,
               criterion: CrossEntropyLoss,
               tokenizer: Tokenizer,
               dataset_loader: JSONDatasetLoader,
               use_amp: bool,
               full_eval: bool,
               pad_id: int,
               max_seq_len: int,
               dynamic: bool,
               device: str) -> float:
```

The `@torch.no_grad()` decorator is used so that gradients are not tracked during evaluation. This reduces computational overhead since we are not performing backpropagation (so gradients aren't needed).

### Parameters

* **model**: The model that's currently undergoing fine-tuning.
* **criterion**: The loss function (`CrossEntropyLoss`) used to measure prediction error.
* **tokenizer**: Tokenizer used to encode text into token IDs.
* **dataset_loader**: Instance of `JSONDatasetLoader` that manages training and validation data.
* **use_amp**: Boolean controlling automatic mixed precision (BF16 or FP32) for performance optimization.
* **full_eval**: Determines whether to evaluate the entire validation set or a single batch.
* **pad_id**: Token ID used for padding, ignored by the loss function.
* **max_seq_len**: Maximum allowable sequence length.
* **dynamic**: Whether to use dynamic padding (based on longest sequence in batch) or fixed padding.
* **device**: Target device ('cuda' or 'cpu').

---

### Full Evaluation Path

When `full_eval=True`, the function runs through the **entire validation dataset**, calculating loss across all batches:

```python
if full_eval:  # Meaning we want to iterate over the entire validation epoch
    current_val_epoch = dataset_loader.val_epoch
    losses = []
    while current_val_epoch == dataset_loader.val_epoch:
        batch = dataset_loader.get_batch(train=False, increment_val_idx=True)

        x, y = tokenize_and_pad_data(batch=batch, tokenizer=tokenizer, pad_id=pad_id, max_seq_len=max_seq_len,
                                     dynamic=dynamic, device=device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16 if use_amp else torch.float32):
            pred = model(x)

            B, T, C = pred.shape
            loss = criterion(pred.reshape(B * T, C), y.reshape(B * T))

        losses.append(loss.item())

    return sum(losses)/len(losses)
```

Here’s what happens step-by-step:

1. **Initialize tracking variables:** The function first records the current validation epoch (`current_val_epoch`) and creates an empty list `losses` to store batch-level losses.
2. **Iterate through validation data:** It loops until one full validation epoch is processed.
3. **Batch retrieval:** Each batch of validation examples is fetched from the loader.
4. **Tokenization & Padding:** The batch is passed through `tokenize_and_pad_data()` to ensure uniform tensor sizes.
5. **Forward pass:** The model processes `x` under mixed precision, outputting logits of shape `(batch, seq_len, vocab_size)`.
6. **Loss computation:** The predictions and labels are reshaped into 2D matrices, and cross-entropy loss is computed.
7. **Aggregation:** Each batch loss is appended to the list, and finally, the average loss across all batches is returned.

This mode is used at the **end of each epoch** to measure the model’s complete validation performance.

---

### Single-Batch Evaluation Path

When `full_eval=False`, only a **single batch** of validation data is evaluated. This is used for quick checks during training, at evaluation intervals:

```python
else:  # Just want a single evaluation
    batch = dataset_loader.get_batch(train=False, increment_val_idx=False)

    x, y = tokenize_and_pad_data(batch=batch, tokenizer=tokenizer, pad_id=pad_id, max_seq_len=max_seq_len,
                                 dynamic=dynamic, device=device)

    with torch.autocast(device_type=device, dtype=torch.bfloat16 if use_amp else torch.float32):
        pred = model(x)
        B, T, C = pred.shape
        loss = criterion(pred.reshape(B * T, C), y.reshape(B * T))

    return loss.item()
```

This version skips looping and calculates the loss for a single batch. It is far faster and is typically used to monitor progress during training — allowing the user to see whether the loss is trending downward.

---

### Usage Context

During fine-tuning, `eval_model()` is called repeatedly:

* **Full evaluation:** Once per epoch to get the complete validation loss.
* **Single-batch evaluation:** At regular intervals (e.g., every few hundred optimizer steps) for quick feedback.

This provides a balance between **speed** and **accuracy** in tracking model performance. Over time, decreasing validation loss (and its corresponding perplexity) signals effective alignment and improved response generation behavior.

---








