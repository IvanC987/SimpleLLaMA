# Supervised Fine-Tuning Dataset

## Introduction: The Role of SFT Data

Supervised Fine-Tuning (SFT) data is fundamentally different from pretraining data. While pretraining uses raw text to teach language patterns, SFT
uses **structured conversations** to teach behavioral alignment—how to follow instructions, adopt roles, and provide helpful responses.

The quality and diversity of SFT data directly determine how well the model transitions from "language predictor" to "helpful assistant."

---

## Dataset Sources

This project uses three high-quality open-source conversational datasets from HuggingFace, chosen for their diversity, scale, and real-world
authenticity:

### 1. lmsys-chat-1m

- **Description**: Large-scale corpus of real user interactions with 25 different LLMs (GPT-4, Claude, Gemini, etc.)
- **Size**: ~1 million conversations
- **Link**: [https://huggingface.co/datasets/lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)

### 2. ShareGPT_Vicuna_unfiltered

- **Description**: Community-curated conversations from the ShareGPT/Vicuna lineage
- **Size**: ~53,000 examples
- **Link**:
[https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)

### 3. smol-smoltalk

- **Description**: Compact, high-quality conversational dataset designed for efficient SFT
- **Size**: ~460,000 examples
- **Link**: [https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)

---

## Why Real Conversational Data?

### The Authenticity Advantage

While synthetic data generation (via LLM APIs like GPT-4, Claude, Gemini) is a common approach, **real user conversations provide critical
advantages**:

**1. Natural Language Variation**
- Real users have inconsistent grammar, spelling errors, and colloquialisms
- Queries are often incomplete, ambiguous, or poorly phrased
- This diversity prevents the model from overfitting to "clean" synthetic patterns

**2. Authentic Distribution**
- Reflects how people *actually* use chat assistants (not how we think they should)
- Captures edge cases and unexpected query types
- Includes contextual nuances that synthetic generation often misses

**3. No Generation Artifacts**
- LLM-generated synthetic data can be repetitive (similar phrasing patterns across examples)
- May inherit biases or stylistic quirks from the generator model
- Real data avoids these "fingerprints"

**4. Proven Quality**
- These datasets have been battle-tested by the community
- Used successfully in training models like Vicuna, StableLM, and others
- Less risk than untested synthetic pipelines

### Trade-offs Acknowledged

Real data isn't perfect:
- May contain noise or low-quality examples (requires filtering)
- Can't perfectly control category distribution like synthetic generation
- Potential for data contamination (examples appearing in model pretraining)

However, for a **1.3B parameter model** trained on **50B pretrain tokens**, the authenticity and diversity outweigh these concerns.

---

## Data Processing Pipeline

The raw HuggingFace datasets undergo several transformation steps before use:

### Step 1: Format Standardization

Each dataset has a different schema. We convert all to a unified JSON structure:

```json
{
  "Template": ["CUSTOM"],
  "User": ["What is 2+2?"],
  "Assistant": ["4"]
}
```

Multi-turn conversations are preserved:

```json
{
  "Template": ["CUSTOM"],
  "User": ["Hello!", "How are you doing today?"],
  "Assistant": ["Hi there! How can I help?", "I'm doing well, thanks for asking!"]
}
```


### Step 2: ASCII Filtering

Since the BPE tokenizer was trained only on ASCII text, all non-ASCII characters are filtered or normalized:

- Examples with non-ASCII content are discarded
- This prevents tokenization mismatches during training
- Ensures consistency with pretraining data distribution

Examples of normalization:

- "café" → "cafe"
- "π" → "pi"
- Em dashes ("—") → hyphens ("-")

### Step 3: Length Filtering

Examples exceeding max_seq_len (2048 or 4096 tokens) are discarded:

```python
tokenized_length = len(tokenizer.encode(full_prompt).ids)
if tokenized_length > max_seq_len:
    discard()
```


This prevents out-of-memory errors during training and truncation artifacts that could confuse the model

### Step 4: Deduplication

Near-duplicate conversations are removed to: 

- Reduce memorization risk
- Increase effective dataset diversity
- Prevent overfitting to repeated patterns

### Step 5: Template Assignment

Most examples use "Template": ["CUSTOM"], which defaults to the project's standard system prompt:

`You are Simple LLaMA, a helpful and factual assistant. Answer clearly, stay on topic, and use context from the conversation. If something is unclear or risky, point it out briefly. Avoid making things up or giving unsupported claims. Be concise, useful, and aligned with the user's goal.`

A small subset (~15-20%) includes custom templates for role-playing or style adaptation:

```json
{
  "Template": ["Explain concepts as if you are a patient teacher."],
  "User": ["What is gravity?"],
  "Assistant": ["Gravity is the force that pulls objects toward each other..."]
}
```

---

Example Conversations from Processed Dataset

Below are representative samples showing the final JSON format after processing:

Example 1: Multi-turn with topic switch

```json
{
    "Template": ["CUSTOM"],
    "User": [
        "Can you explain the greenhouse effect in simple terms?",
        "Switching topics, how do I organize a messy garage?"
    ],
    "Assistant": [
        "The greenhouse effect is when gases in Earth's atmosphere trap heat from the sun, like glass in a greenhouse. This keeps Earth warm, but
human activities have increased these gases, causing more heat retention and climate change.",
        "Start by sorting items into categories: keep, donate, recycle, trash. Use the '5-box method' and work in small sections. Install wall hooks
for tools, use clear bins for visibility, and label everything."
    ]
}
```


Example 2: Logical reasoning

```json
{
    "Template": ["CUSTOM"],
    "User": [
        "All birds have feathers. A penguin is a bird. Does a penguin have feathers?"
    ],
    "Assistant": [
        "Yes, a penguin has feathers. The reasoning follows: All birds have feathers, and since a penguin is a bird, it must also have feathers.
Penguin feathers serve purposes like insulation and waterproofing."
    ]
}
```

Example 3: Custom template (role-playing)

```json
{
    "Template": [
        "Explain concepts as if you are a wise, patient librarian. Use book metaphors."
    ],
    "User": [
        "What is a database?"
    ],
    "Assistant": [
        "Ah, a database is like a magical card catalog for this library. It doesn't hold the books themselves, but knows exactly where each one is.
When you need information, you consult the catalog, and it points you to the exact shelf and page. It's perfect order for a world of chaotic
information."
    ]
}
```

These examples demonstrate:
- Natural language variation (some verbose, some concise)
- Multi-turn conversation handling
- Template-based role adaptation
- Diverse task types (QA, reasoning, instruction-following)


---


## JSONDatasetLoader

Finally, the **`JSONDatasetLoader`** class. This is used for loading, processing, and serving the SFT dataset during fine-tuning. It handles both the training and validation data splits, manages batching, and keeps track of epoch progress.

---

### Constructor

Let’s start with the constructor, step by step:

```python
def __init__(self, json_filepath: str, batch_size: int, train_split: float):
    assert batch_size > 0
    assert 0 < train_split <= 1

    # Load in the dataset, should be a list of dicts
    # Should have User, Assistant, and Template keys, of types list[str], list[str] and str respectively
    with open(json_filepath, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    random.shuffle(dataset)
```

The constructor accepts three arguments:

* **`json_filepath`**: the path to the JSON dataset file (e.g., `merged_ft_dataset.json`).
* **`batch_size`**: how many examples are returned per batch.
* **`train_split`**: the proportion of examples allocated for training versus validation.

After confirming valid inputs, the JSON file is opened and loaded. This file should contain a list of dictionaries where each dictionary corresponds to a conversation sample, containing the three keys: `User`, `Assistant`, and `Template`. The dataset is shuffled to ensure randomness before splitting.

---

Next, the dataset is converted into a list of formatted `(x, y)` string pairs using the **`format_training_prompt`** function:

```python
dataset = [format_training_prompt(user=d["User"],
                                  assistant=d["Assistant"],
                                  template=(d["Template"][0])
                                  ) for d in dataset]

n = int(len(dataset) * train_split)
self.train_dataset = dataset[:n]
self.val_dataset = dataset[n:]
```

Each entry is passed through `format_training_prompt`, which transforms raw JSON entries into fully formatted text prompts and target responses suitable for tokenization. This function handles insertion of special tokens (`<SOT>`, `<SOU>`, `<EOU>`, `<SOA>`, `<EOA>`) and confirms that the final assistant output aligns correctly with the training objective. (More detail about this will be covered in the next section, `Prompt Formatting`)

The dataset is then divided into **training** and **validation** sets according to the provided split ratio. For instance, if `train_split=0.99`, 99% of examples go to training and 1% to validation.

---

Finally, we initialize several bookkeeping variables:

```python
self.batch_size = batch_size
self.train_epoch = 0
self.val_epoch = 0
self.train_idx = 0
self.val_idx = 0

# Remove from memory
del dataset
```

* `train_epoch` and `val_epoch` track the number of completed epochs for each split.
* `train_idx` and `val_idx` track where we are in the dataset for each epoch.
* The original `dataset` variable is deleted to conserve memory, as it can be quite large in full-scale fine-tuning.

---

### `get_batch` Method

The **`get_batch`** method is the main interface for retrieving data batches during training and evaluation.

```python
def get_batch(self, train: bool, increment_val_idx=True):
    # "increment_val_idx" is set to False when needing to eval a small section

    if train:
        batch = self.train_dataset[self.train_idx: self.train_idx + self.batch_size]
        self.train_idx += self.batch_size

        if self.train_idx + self.batch_size >= len(self.train_dataset):
            self.train_idx = 0
            self.train_epoch += 1
            random.shuffle(self.train_dataset)
```

When **`train=True`**, the function slices a batch from `self.train_dataset` using the current index. After retrieving the batch, it increments the index to point to the next section. If the index reaches the end of the dataset, it resets back to zero, increments the epoch counter, and reshuffles the training data before proceeding to the nexte poch.

This cyclical behavior ensures that the data pipeline continuously feeds new permutations of the dataset throughout fine-tuning.

---

When **`train=False`**, the loader instead retrieves data from the validation set:

```python
else:
    batch = self.val_dataset[self.val_idx: self.val_idx + self.batch_size]
    if increment_val_idx:
        self.val_idx += self.batch_size

    if self.val_idx + self.batch_size >= len(self.val_dataset):
        self.val_idx = 0
        self.val_epoch += 1

    return batch
```

Here, the **`increment_val_idx`** parameter is important. During full validation (when evaluating the entire validation set), this flag remains **`True`** so that the loader moves through the data sequentially. However, for **quick validation intervals** during training — which happen often — `increment_val_idx` is set to **`False`**, ensuring the same small validation batch is reused for efficiency, to avoid excess computational cost. 

