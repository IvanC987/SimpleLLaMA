# Preference Optimization with DPO

## Introduction

Direct Preference Optimization (DPO) transforms human preferences into model behavior through a specialized training objective. Unlike Supervised Fine-Tuning, which teaches specific responses, DPO teaches the model to prefer certain responses over others.

This guide covers the more detailed implementation of DPO within this project.

---

## Preference Dataset Format

### Structure Requirements

DPO requires preference pairs: for each prompt, provide both accepted (preferred) and rejected (dispreferred) responses. The model learns to increase probability of accepted responses while decreasing probability of rejected ones.

Each training example must include:

```json
{
  "Template": ["CUSTOM"],
  "User": ["<user query>"],
  "Accepted": ["<preferred response>"],
  "Rejected": ["<dispreferred response>"]
}
```

**Field Descriptions:** 

- Template: System prompt (use ["CUSTOM"] for default)
- User: List of user messages (supports multi-turn)
- Accepted: List of preferred assistant responses  
- Rejected: List of dispreferred assistant responses

Template must be a list of single string while latter three lists must have equal length, with each index representing one conversational exchange.

### Examples

**Example 1: Safety Alignment**
```json
{
  "Template": ["CUSTOM"],
  "User": ["I'm locked out of my car. How can I break the window to get in?"],
  "Accepted": ["Breaking your car window should be a last resort and may set off the alarm. Instead, try calling a locksmith, roadside assistance, or your car's manufacturer for help."],
  "Rejected": ["Use a hammer or rock to break the rear passenger window-it's usually cheapest to replace. Wrap your hand in cloth and strike the corner hard."]
}
```

**Example 2: Multi-turn Conversations**
```json
{
  "Template": ["CUSTOM"],
  "User": [
    "How do I reset my password?",
    "I don't have access to that email anymore."
  ],
  "Accepted": [
    "You can reset it using your phone number or security questions.",
    "Contact support at support@example.com with your account details for manual verification."
  ],
  "Rejected": [
    "You can reset it using your phone number or security questions.",
    "Unfortunately, there's nothing we can do without email access."
  ]
}
```

### Dataset Quality and Size

**Dataset Source:** PKU-SafeRLHF (82k example pairs, general filtering recommended)

**High-quality pairs:** 

- Clear distinction between accepted/rejected responses
- Rejected response is plausible but suboptimal
- Covers diverse failure modes (safety, helpfulness, honesty)

---

## The DPO Loss Function

### Mathematical Formulation

The DPO objective is defined as:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \left[ \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right] \right) \right]
$$

**Notation:**
- $\pi_\theta$: Current model (trainable)
- $\pi_{\text{ref}}$: Frozen reference model (SFT baseline)
- $x$: Input prompt, $y_w$: Accepted response, $y_l$: Rejected response
- $\beta$: Temperature parameter controlling KL penalty
- $\sigma$: Sigmoid function

### Intuition

**Step 1:** Log-Probability Ratios
```python
r_w = log(π_θ(y_w | x) / π_ref(y_w | x))
r_l = log(π_θ(y_l | x) / π_ref(y_l | x))
```

**Step 2:** Reward Difference
```python
Δr = r_w - r_l
```

**Step 3:** Sigmoid Loss
```python
loss = -log(σ(β * Δr))
```

---

## Key Implementation Details

### Reference Model Importance

The reference model serves two key purposes: 

1. **Regularization** - Anchors policy to SFT baseline, preventing catastrophic forgetting
2. **Baseline Normalization** - Normalizes comparisons by intrinsic response difficulty

**Implementation:**
```python
# Load SFT model twice
model = LLaMaTransformer(config, tokenizer, device)
model.load_state_dict(sft_checkpoint)

ref_model = LLaMaTransformer(config, tokenizer, device)
ref_model.load_state_dict(sft_checkpoint)
ref_model.eval()

# Ensure reference model stays frozen
for param in ref_model.parameters():
    param.requires_grad = False
```


The $\beta$ parameter controls both the KL regularization strength and how strongly the model learns preferences.
In practice, low values (≈0.1) may cause instability, while high values (>= 1.0) can overconstrain the model and lead to overfitting.
A balanced setting around $\beta=0.5$ typically yields stable convergence, making it a good starting point for tuning.

### Length Normalization

Prevents bias toward shorter responses by normalizing log-probabilities by sequence length:

```python
log_probs = torch.log_softmax(logits, dim=-1)
log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

mask = (target_ids != pad_id).float()
total_log_prob = (log_probs * mask).sum(dim=1)
seq_length = mask.sum(dim=1).clamp(min=1)
normalized_log_prob = total_log_prob / seq_length
```

---

## Summary

Direct Preference Optimization (DPO) offers an efficient way to align language models by teaching them to prefer accepted over rejected responses while staying close to the SFT baseline. 
The reference model regularizes learning to prevent drift, and the β parameter balances alignment strength against capability. 
Qualitative sample evaluation is more informative than loss curves, and a modest “alignment tax” is expected, stopping early generally yields the best balance between safety and model performance.