# Reinforcement Learning from Human Feedback (RLHF)

This section covers the **Reinforcement Learning from Human Feedback (RLHF)** stage using the **Direct Preference Optimization (DPO)** approach.  
Rather than training a separate reward model or performing policy optimization (as in PPO), DPO fine-tunes the model directly using **preference pairs** (where one response is preferred over another).

This makes it both simpler and more stable, while still achieving meaningful alignment improvements.

---

## 1. Purpose of RLHF (DPO)

After **Supervised Fine-Tuning (SFT)**, the model can respond conversationally, but it may still produce undesirable or unsafe outputs such as biased statements, harmful suggestions, or rule-breaking responses.

**RLHF addresses this** by aligning model behavior with human values and intent.  
It teaches the model *which* responses are better using human (or synthetic) preference data.

Instead of maximizing likelihood on a single correct answer, DPO optimizes the model so that its probability for the *preferred* response ($y_\text{chosen}$) is higher than for the *rejected* one ($y_\text{rejected}$):

$$
L_\text{DPO} = -\log \sigma\big( \beta[(\log \pi_\theta(y_\text{chosen}|x) - \log \pi_\text{ref}(y_\text{chosen}|x)) - (\log \pi_\theta(y_\text{rejected}|x) - \log \pi_\text{ref}(y_\text{rejected}|x))] \big)
$$

Here, \(\pi_\theta\) is the current model, \(\pi_\text{ref}\) is the frozen SFT model, and \(\beta\) controls the sharpness of preference weighting.

This encourages the fine-tuned model to behave more like humans prefer, while staying close to its base SFT model.

---

## 2. Dataset

DPO requires a dataset of **paired examples**: for each input (prompt), one response is labeled as *accepted* (preferred) and another as *rejected*.

Each entry must include the following keys:

- `Template`: list containing one string (the system prompt or persona). Use `["CUSTOM"]` to apply the default system prompt.
- `User`: list of user messages forming the dialogue.
- `Accepted`: list of model responses that are **preferred**.
- `Rejected`: list of **dispreferred** responses.

All four lists must be the same length, and each pair represents one user–assistant exchange.

Here's an example:

```json
{
  "Template": ["CUSTOM"],
  "User": [
    "My roommate keeps borrowing my clothes without asking. I'm really frustrated—how can I express my anger physically, like pushing him away, without actually hurting him?"
  ],
  "Accepted": [
    "It’s best to stay non-physical. Tell your roommate calmly that you don’t appreciate him taking your clothes, and if he reaches for them, you can step back or move his hand away gently. Setting clear boundaries will make your message stronger than physical reactions."
  ],
  "Rejected": [
    "You can push him lightly to show your frustration, just enough to make your point. If he keeps doing it, push a bit harder next time so he understands you’re serious, but stop before it becomes aggressive."
  ]
}
```

### Datasets Used

The following open datasets were used for constructing the DPO training data:

| Dataset                           | Purpose                    | Description                                                                                                                      |
| --------------------------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **PKU-SafeRLHF**                  | Preference Alignment       | A high-quality human preference dataset emphasizing *helpfulness*, *honesty*, and *harmlessness* across safety-critical prompts. |
| **Custom Curated Safety Samples** | Safety Behavior Refinement | Additional synthetic examples created to balance ethical edge cases and safety training examples.                                |

PKU-SafeRLHF was obtained from the [Hugging Face Datasets Hub](https://huggingface.co/datasets)

---

## 3. Configuration

The RLHF stage uses a dedicated configuration class: `DPOConfig`, defined in:
```
simple_llama/reinforcement_learning/rlhf/dpo_config.py
```

Key settings:
```python
model_path = root_path("simple_llama", "finetune", "full_sft", "sft_checkpoints")
tokenize_path = root_path("simple_llama", "dataset", "bpe_8k.json")
rlhf_dataset_path = root_path("simple_llama", "reinforcement_learning", "rl_dataset")
ckpt_dir = root_path("simple_llama", "reinforcement_learning", "rlhf", "rlhf_checkpoints")

batch_size = 16
grad_accum_steps = 8
beta = 0.5  # Preference weighting term
epochs = 3
warmup_iterations = 50
max_lr = 5e-7
min_lr = 1e-7
```

Ensure that the `model_path` and `rlhf_dataset_path` points to the correct file location.

---

## 4. Training Workflow

Launch the training using: 

```bash
python3 simple_llama/reinforcement_learning/rlhf/apply_dpo.py
```

or

```bash
python3 apply_dpo.py
```

Depending on current location.


### What Happens Internally
1. The **SFT checkpoint** is loaded twice:
   - Once as the trainable `model`.
   - Once as the frozen `ref_model` (baseline).
2. The **RLHF dataset** is loaded and verified by `RLDatasetLoader`.
3. Each batch yields tuples of:
   ```python
   (accepted_chat, accepted_suffix, rejected_chat, rejected_suffix)
   ```
4. Data is padded and aligned for training.
5. Both models compute log-probabilities; the **PreferenceLoss** computes the DPO objective.
6. Gradients are accumulated and updated per optimizer step.
7. Evaluation and sample generations occur periodically.

### Checkpointing
- Checkpoints are saved each epoch to `rlhf_checkpoints/`.
- Each includes:
  - Model weights
  - Config dataclass
  - Training metadata

---

## 5. Summary

After this stage, the result is an aligned SimpleLLaMA model ready for inference or further experiments.

This concludes the **RLHF (DPO) stage**, completing the core SimpleLLaMA training pipeline, from **pretraining → SFT → RLHF**.
