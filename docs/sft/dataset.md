# Supervised Fine-Tuning Dataset Creation

This document explains in detail how the **Supervised Fine-Tuning (SFT) dataset** was created for the SimpleLLaMA project. The dataset contains **216,000 synthetic samples** generated using DeepSeek and Gemini 2.5 (Flash and Pro). It is designed specifically for a 1.3B parameter model — which is much smaller and simpler than the 7B+ models often targeted by publicly available datasets — and therefore required careful curation and balance.

---

## Why Synthetic Instead of Using Hugging Face Datasets?

While there are many instruction-tuning datasets available on Hugging Face, such as **Alpaca, ShareGPT, OASST**, etc., they are generally constructed for **larger-scale models (7B–70B parameters)**. Using them directly could misalign a smaller model in several ways:

- **Level of difficulty** – Datasets like OASST or ShareGPT contain complex, nuanced conversations. For a 1.3B model, this would often lead to failure cases. A controlled dataset is used to confirm examples are at the right difficulty level.  
- **Distribution mismatch** – Public datasets are often skewed toward open-domain chit-chat, or contain an imbalanced ratio of tasks. Here, the dataset was constructed with explicit category proportions (see below).  
- **Control over content** – By generating synthetic data, we can tightly control what categories exist, their frequency, and their complexity. This reduces noise and makes the model more reliable for its intended use.  

In short: **synthetic generation provides balance, scalability, and suitability for a smaller model**, while avoiding the pitfalls of “overshooting” with datasets made for much larger systems.

---

## Dataset Composition

The dataset is broken down into **8 categories**, each representing a different skill domain. These categories were chosen to ensure broad but manageable coverage of practical use cases. Below is the distribution:

| Category | Share | Purpose | Example |
|----------|-------|---------|---------|
| **Instruction Following** | 30% | Core alignment skill: teaching the model to obey explicit instructions. | “Summarize this article in 2 sentences.” |
| **User Conversation** | 20% | Multi-turn dialogues, simulating back-and-forth exchanges. Adds robustness to conversational flow. | “Hello! How are you?” → “I’m good, how are you?” |
| **Reasoning & Logic** | 15% | Arithmetic, logical reasoning, multi-step thought. Strengthens structure in answers. | “If Sarah has 5 apples and eats 2, how many are left?” |
| **QA (Factual)** | 10% | Short, grounded factual responses. Prevents drift and strengthens retrieval-like capabilities. | “What is the capital of Canada?” |
| **Error Correction** | 5% | Simple grammar/spelling fixes. Teaches token-level precision. | Input: “She go to market.” → Output: “She goes to the market.” |
| **Structured Transformation** | 5% | Converting between formats, summarization, bullet-pointing, JSON transformations. | “Turn this paragraph into a bulleted list.” |
| **Long Response Generation** | 5% | Encourages extended outputs, storytelling, and maintaining coherence across long sequences. | “Write a short story about a lost robot.” |
| **Identity / Miscellaneous** | 10% | Handles “Who are you?” queries, opinions, small-talk, and personality alignment. | “Who are you?” → “I’m SimpleLLaMA, your assistant.” |

---

### Subcategories

Within each major category, subcategories were mixed in to increase diversity. For example:

- **Instruction Following**
    - Summarization (“Summarize this paragraph in 3 bullet points”).  
    - Classification (“Classify this review as positive or negative”).  
    - Style constraints (“Explain this like I’m 5”).  

- **User Conversation**
    - Multi-turn casual conversation.  
    - Task-based (helping with scheduling, answering follow-ups).  

- **Reasoning & Logic**
    - Arithmetic problems.  
    - Deductive logic.  
    - Simple word problems.  

- **Structured Transformation**
    - Bullet point compression.  
    - Extracting keywords.  

This is to make sure that within each category, the model isn’t just repeating a narrow skill, but instead sees varied applications of the same general ability.

---

## Generation Pipeline

### Step 1: Few-Shot Prompting

Examples were generated using **few-shot prompts** with Gemini and DeepSeek. A handful of verified examples were embedded into the system prompt, ensuring that outputs were consistent with the required format:

```json
{
  "Template": ["Speak like a teacher"],
  "User": ["Why does the sun set?"],
  "Assistant": ["The sun appears to set because the Earth rotates, making it seem like the sun is moving across the sky."]
}
```

---

### Step 2: Strict Formatting

The output had to always be a **list of JSON dicts** with exactly three keys: `Template`, `User`, `Assistant`. Each value was wrapped in a list of strings to ensure uniformity with the later dataset loader.

---

### Step 3: ASCII Enforcement

Since the tokenizer was trained only on **ASCII English text**, all outputs were passed through a normalization step that replaced or removed non-ASCII characters. For example:

- “café” → “cafe”  
- “π” → “pi”  
- “–” (en dash) → “-”  

This avoided subtle tokenization mismatches that could destabilize training.

---

### Step 4: Quality Controls

- **Multi-model variation** – Three different LLMs (DeepSeek, Gemini Flash, Gemini Pro) were used for generation. This provided diverse styles and avoided overfitting to one model’s quirks.  
- **Non-ASCII filter** – All responses passed through a script that rejected or normalized invalid characters.  
- **Manual inspection** – Roughly **0.5% of the dataset (≈1,000 samples)** was manually checked. Results showed very low hallucination or formatting issues.  
- **Broad scope** – Prompts were intentionally kept **simple and general** (instruction following, reasoning, factual questions). This avoided cases where even state-of-the-art models might hallucinate.

---

## Example of Prompt Construction

Below is a simplified version of the generation prompt used for instruction-following data:

```
You are generating high-quality instruction-following examples for an SFT dataset.
Make sure responses are ASCII only.
Each example must be a JSON dict with keys: "Template", "User", "Assistant".
Responses must be coherent, varied, and reflect the style in "Template".
Generate 5 JSON dicts, each containing a unique instruction-response pair.
```

In the actual generation script, varying values such as `top_p`, `temperature`, fewshot examples, were injected into the prompt to prevent the LLM from generating the same, or extremely similar, results. 

---

## Generation Samples

Here are some examples within the SFT dataset: 

```json
[
    {
        "Template": [
            "CUSTOM"
        ],
        "User": [
            "Can you explain the greenhouse effect in simple terms and how it relates to climate change? I've heard the term a lot but don't fully grasp the mechanism.",
            "Switching topics completely, what's your favorite method for organizing a messy garage that hasn't been cleaned in years? I need practical steps to tackle this overwhelming project."
        ],
        "Assistant": [
            "The greenhouse effect is when certain gases in Earth's atmosphere (like carbon dioxide and methane) trap heat from the sun, similar to how glass traps heat in a greenhouse. Normally this keeps Earth warm enough for life, but human activities have increased these gases, causing more heat retention and global climate changes. Burning fossil fuels and deforestation are major contributors to this enhanced greenhouse effect.",
            "Start by sorting items into categories: keep, donate, recycle, trash. Use the '5-box method' - one box for each category plus a 'relocate' box for items that belong elsewhere. Work in small sections to avoid overwhelm. Install wall hooks for tools, use clear bins for visibility, and label everything. Schedule regular maintenance sessions to prevent future buildup."
        ]
    },
    {
        "Template": [
            "CUSTOM"
        ],
        "User": [
            "All birds have feathers. A penguin is a bird. Does a penguin have feathers?"
        ],
        "Assistant": [
            "Yes, a penguin has feathers. The reasoning follows deductively: All birds have feathers, and since a penguin is classified as a bird, it must also possess feathers. This holds true even though penguins are flightless-their feathers serve other purposes like insulation and waterproofing, but they still fit the definition of birds in this regard."
        ]
    },
    {
        "Template": [
            "Explain concepts as if you are a wise, patient, and old librarian. Use metaphors related to books, stories, and knowledge."
        ],
        "User": [
            "Can you explain what a 'database' is?"
        ],
        "Assistant": [
            "Ah, a wonderful question. Imagine this library is the whole of your information. A 'database' is like a magical card catalog. It doesn't hold the books themselves, but it knows exactly where each one is. It has a card for every story, every fact, every character. When you need to find something, you don't wander the stacks for ages. You simply consult the catalog, and it points you to the exact shelf, the exact book, the exact page. It's a system of perfect order for a world of chaotic information."
        ]
    }
]
```


- The first example shows a `CUSTOM` system prompt and demonstrates flexibility — answering a science question and then switching to a practical, unrelated task.  
- The second example uses categorical logic (“all birds have feathers”), which strengthens the model’s structured reasoning abilities.  
- The third example illustrates a style template (the whimsical librarian), which enriches the dataset with persona-following behavior.

Most entries use `CUSTOM` as the template, which defaults to the general system prompt. Around 20–25% explicitly define styles or roles, broadening stylistic adaptability and improving robustness.
More will be discussed in the next section, Prompt Formatting. 


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

