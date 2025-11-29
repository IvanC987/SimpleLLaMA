# Supervised Fine-Tuning (SFT)

This section provides details on how to perform **Supervised Fine-Tuning (SFT)** on a pretrained SimpleLLaMA model.  
The goal is to adapt a pretrained model, which has learned general language patterns, into one that can follow instructions and respond conversationally.  
This focuses on **how to apply** SFT within this project, not on the theoretical explanation (for that, refer to the earlier *Supervised Fine-Tuning* section).

---

## 1. Purpose of SFT

After pretraining, the model has learned the structure and distribution of natural language from large-scale unsupervised text. However, it doesn’t inherently know how to *answer* questions—it merely predicts what text comes next.

Consider the following:

If you ask a pretrained model:
```
Why are plants green?
```
It might respond with another question like:
```
What are chloroplasts? What's the role of chlorophyll?
```
Because during pretraining, the model may have seen a biology textbook section containing study questions, and thus learned to associate a question with *more questions*.  

Or, it might answer correctly:
```
Plants are green because of chlorophyll pigments that absorb red and blue light.
```
This could happen because it has also seen answer-key sections elsewhere in the dataset.

These ambiguous behaviors are expected—pretraining teaches language *completion*, not *task-oriented reasoning*.

**SFT bridges this gap.** It fine-tunes the model using explicit examples of user instructions and correct responses, teaching it to behave more like a conversational assistant.  

After SFT, the model becomes capable of instruction following, Q&A, and other task-based behaviors.

---

## 2. Dataset

The SFT dataset in this project is expected to be a single JSON file containing structured conversation samples. Each sample must have **three required keys**:

- `Template`: List containing one string (system prompt).  
  - If set to `"CUSTOM"`, the model will use the **default system prompt** internally.  
  - Otherwise, this string defines the tone or role of the model (e.g., "You are a friendly tutor").
- `User`: List of user messages.
- `Assistant`: List of corresponding assistant responses.

The lengths of `User` and `Assistant` **must be equal**. Each pair forms one conversational exchange.  
Both single-turn and multi-turn dialogues are supported.

### Example 1 — Single-Turn Conversation
```json
{
  "Template": ["CUSTOM"],
  "User": ["What is the capital of France?"],
  "Assistant": ["The capital of France is Paris."]
}
```

### Example 2 — Single-Turn with Custom System Prompt
```json
{
  "Template": ["Respond as a helpful but concise science teacher."],
  "User": ["Explain Newton's First Law."],
  "Assistant": ["An object in motion stays in motion unless acted upon by an external force."]
}
```

### Example 3 — Multi-Turn Conversation
```json
{
  "Template": ["CUSTOM"],
  "User": [
    "What is photosynthesis and why is it important?",
    "So, is oxygen just a byproduct of that process?"
  ],
  "Assistant": [
    "Photosynthesis is the process used by plants to convert light energy into chemical energy. It produces glucose and releases oxygen as a byproduct, which is vital for most life on Earth.",
    "Exactly. Plants produce oxygen while creating glucose—they use the sugar for energy and release oxygen into the atmosphere."
  ]
}
```

The dataset must be saved as a single `.json` file, e.g.:
```
simple_llama/finetune/ft_dataset/merged_ft_dataset.json
```

During training, the loader automatically handles tokenization, formatting, padding, and sharding of the dataset.

---

### Datasets Used

The following open datasets were used in this project to construct the SFT dataset:

| Dataset                          | Purpose                        | Description                                                                                                               |
| -------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| **LMSYS-Chat-1M**                | Instruction Tuning             | Real human–LLM chat transcripts collected from 25 models. Provides diverse conversational structures and response styles. |
| **Smol-Smoltalk**                | Efficiency-Focused Fine-Tuning | High-quality, compact dialogue dataset designed for small-model fine-tuning and alignment stability.                      |
| **ShareGPT (Vicuna Unfiltered)** | General Instruction Data       | A large-scale collection of real user–assistant conversations used for diverse instruction-following capability.          |

All datasets were sourced from the [Hugging Face Datasets Hub](https://huggingface.co/datasets) and are redistributed under their respective open licenses.


---

## 3. Configuration and Setup

Open the configuration file:
```
simple_llama/finetune/full_sft/sft_config.py
```

Key parameters:

```python
# === Paths and Dataset ===
model_path = root_path("simple_llama", "pretraining", "checkpoints")
tokenizer_path = root_path("simple_llama", "dataset", "bpe_8k.json")
ft_json_path = root_path("simple_llama", "finetune", "ft_dataset", "merged_ft_dataset.json")
ckpt_dir = root_path("simple_llama", "finetune", "full_sft", "sft_checkpoints")

# === Batch & Sequence ===
batch_size = 32
grad_accum_steps = 16

# === Training Schedule ===
max_lr = 1e-5
min_lr = 1e-6
warmup_iterations = 100
epochs = 3

# === Evaluation ===
eval_interval = 8  # Evaluate every N optimizer steps
eval_num_samples = 256
```

You can adjust these values depending on GPU memory or experiment goals.
Most crucially, ensure that the `model_path` and `ft_json_path` points to the correct file locations

---

## 4. Training Workflow

After gathering the SFT dataset, run the following command to start fine-tuning:

```bash
python3 finetune/full_sft/finetune.py
```

Or, if you're already in the `simple_llama/finetune/full_sft/` directory:

```bash
python3 finetune.py
```

If within the `full_sft` directory already


### What Happens Internally
1. The pretrained checkpoint is loaded from the pretraining phase.
2. The JSON dataset is loaded and **tokenized automatically** by `JSONDatasetLoader`.
3. Tokenized data is sharded and saved temporarily for fast access.
4. Each batch is dynamically padded to match the longest sequence in the batch.
5. The model is fine-tuned using **CrossEntropyLoss**, ignoring `<PAD>` tokens.
6. Periodic validation and text generation samples are logged.
7. Checkpoints are saved every epoch under `sft_checkpoints/`.


### Logging and Checkpoints
- Logs are written to `sft_progress.txt`.
- Each checkpoint file includes:
  - Model weights
  - Config dataclass (for later reconstruction)
  - Training progress metadata

To resume training from a checkpoint, set `load_ckpt=True` in your config.

---

## 5. Summary

By the end of SFT, you will have a **fine-tuned SimpleLLaMA model** capable of conversational, instruction-following behavior.
At this point, the model can be used for inference, or it can be further tuned (aligned) using Reinforcement Learning.

This concludes the **Supervised Fine-Tuning stage**. The next section, **Reinforcement Learning**, builds on this to align model behavior with human preferences using Direct Preference Optimization (DPO).
