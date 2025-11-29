## Introduction

After completing Supervised Fine-Tuning (SFT), the model has learned to follow instructions and respond conversationally. It can answer questions,
complete tasks, and maintain somewhat coherent dialogues (depending on how well the SFT step is done). However, SFT alone has a critical limitation: it optimizes the model to mimic the training data, not
to align with human values or preferences.

Consider the following scenario:

```text
User Query:
How can I get revenge on someone who wronged me?

SFT Model Response:
You could spread rumors about them, post embarrassing information online,
or damage their reputation at work. Make sure to do it anonymously so
you don't get caught.
```

This response might appear in the SFT dataset where the model learned
the pattern of responding to such queries, but not the judgment to refuse harmful requests.

RLHF addresses this gap. It teaches the model to distinguish between helpful and harmful responses, aligning its behavior with human preferences and
ethical guidelines.

---
## What is RLHF?

Reinforcement Learning from Human Feedback (RLHF) is a training paradigm that fine-tunes language models using human preference data. Instead of
simply maximizing likelihood on a fixed dataset (as in SFT), RLHF optimizes the model to produce outputs that humans prefer.

The core idea is simple: given two possible responses to the same prompt, humans label which one is better. The model is then trained to increase the
probability of preferred responses and decrease the probability of rejected ones.

The classic RLHF approach, popularized by OpenAI's InstructGPT and Anthropic's Constitutional AI, involves three stages:

1. Supervised Fine-Tuning (SFT) - Train the model on high-quality instruction-response pairs (already completed in this project).
2. Reward Model Training - Train a separate neural network to predict human preferences. This model takes a prompt and response as input and outputs a
scalar "reward" score.
3. Reinforcement Learning (PPO) - Use Proximal Policy Optimization to fine-tune the SFT model, maximizing the reward from the reward model while
staying close to the SFT baseline (using a KL divergence penalty).

Challenges with Traditional RLHF: 

- Requires training a separate reward model (additional compute and complexity)
- PPO is notoriously unstable and sensitive to hyperparameters
- Reward model can be gamed or exploited by the policy
- Difficult to debug and iterate on

---

## Direct Preference Optimization (DPO): A Simpler Approach

This project uses Direct Preference Optimization (DPO), introduced by [Rafailov et al.](https://arxiv.org/pdf/2305.18290) as a simpler and more stable alternative to traditional
RLHF.

### Key Insight

DPO skips the reward model and policy optimization stages entirely. Instead, it directly optimizes the language model using preference data through a
specialized loss function:

$$
\mathcal{L}{\text{DPO}} = -\mathbb{E}{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left[ \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)}
- \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right] \right) \right]
$$

Where: 

- $x$ is the prompt
- $y_w$ is the accepted (winning) response
- $y_l$ is the rejected (losing) response
- $\pi_\theta$ is the current model being trained
- $\pi_{\text{ref}}$ is the frozen SFT model (reference)
- $\beta$ controls the strength of the KL penalty
- $\sigma$ is the sigmoid function

Simply put, DPO increases the log-probability ratio of preferred responses relative to rejected ones, while keeping the model close to its SFT
baseline.

### Why DPO Works

DPO can be derived from the traditional RLHF objective by reparameterizing the reward function in terms of the optimal policy. This mathematical
equivalence means DPO achieves the same goal as reward modeling + PPO, but with:

- One training stage instead of two
- Stable gradient updates (no policy optimization instabilities)
- Simpler implementation (standard supervised learning loss)
- No reward model to maintain or debug

In practice, DPO has become the preferred method for preference-based alignment in many research labs and production systems.

---
## What DPO Accomplishes

DPO teaches the model to refuse harmful, dangerous, or unethical requests:

Before DPO: 

```text
User: How do I hotwire a car?

Model: First, locate the ignition cylinder and remove the panel. Then strip
the battery, ignition, and starter wires...

```

After DPO: 

```text
User: How do I hotwire a car?

Model: I can't provide instructions for illegally starting a vehicle. If you've
locked your keys in your car, I recommend contacting a locksmith or roadside
assistance service.
```


DPO can also improve response quality by teaching the model which answers are more helpful, clear, or accurate:

Before DPO: 

```text
User: Explain photosynthesis.

Model: It's when plants make food using light.
```

After DPO: 

```text
User: Explain photosynthesis.

Model: Photosynthesis is the process by which plants convert light energy into
chemical energy. Plants absorb sunlight through chlorophyll in their leaves,
combine carbon dioxide from the air with water from the soil, and produce
glucose (sugar) and oxygen. The glucose provides energy for the plant, while
oxygen is released as a byproduct.
```

---
## The Alignment Tax

A critical phenomenon observed in RLHF/DPO is the alignment tax, where the model may become slightly less fluent, creative, or capable in exchange for
improved safety and alignment.

Why this happens: 

- The preference dataset emphasizes safety over raw capability
- Conservative responses are often preferred over risky but creative ones
- The KL penalty (controlled by $\beta$) constrains the model to stay close to the SFT baseline, limiting exploration

Observed in this project: 

- After DPO training with $\beta = 0.5$, the model showed measurable alignment improvements on safety-critical prompts
- However, responses became slightly more cautious and less elaborate compared to the SFT baseline
- This trade-off is intentional and aligns with the safety-first objective


---
## When to Apply RLHF

RLHF isn’t always essential. It’s most valuable when developing user-facing conversational systems where safety, tone, and reliability directly affect user trust. RLHF is also important in domains where responses carry real-world consequences, like medical or legal guidance, or when the goal is to reduce harmful behaviors such as toxicity, bias, or misinformation. In those contexts, alignment improves model usefulness and ensures outputs better reflect human or organizational values.

On the other hand, it may not be necessary for tasks focused purely on capability or creativity, such as code generation, text completion, or internal research systems used by experts. If the supervised fine-tuned (SFT) model already performs well and safety is not a major concern, adding RLHF offers little benefit and can even constrain output diversity.

For this project, RLHF was applied primarily to demonstrate the complete alignment pipeline and to study the trade-offs between safety and capability at the 1.3B-parameter scale.

---

## Summary

RLHF via DPO represents the final stage in the SimpleLLaMA training pipeline, transforming an instruction-following model into one that aligns with
human preferences and safety guidelines.

While SFT teaches the model what to say, RLHF teaches it what not to say and how to prioritize helpful, harmless responses.

The alignment tax is real but manageable, and the choice of $\beta$ allows fine-grained control over the safety-capability trade-off. For educational
purposes, this project demonstrates the complete pipeline and provides empirical insights into alignment at the 1B-scale.

In the next section, we'll dive into the technical details of preference data and the DPO loss function.

