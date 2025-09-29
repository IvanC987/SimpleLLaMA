The learning rate (LR) is one of the most important hyperparameters in training deep neural networks, where it's used to adjusts the learning rate dynamically during training, instead of keeping it fixed.

This project includes a custom `Scheduler` class that implements warmup and three different scheduling strategies: **cosine decay**, **linear decay**, and **constant LR**.

---

### Why Use a Scheduler?

Schedulers help address two common issues in optimization:

- **Exploding/vanishing gradients** – keeping the LR too high/low throughout training often leads to instability or poor convergence.  
- **Training dynamics** – a model often benefits from a short *warmup* phase (slowly ramping LR up), followed by a gradual *decay* to smaller values.  
- **Generalization** – decaying the LR near the end of training often improves final accuracy/perplexity.

Instead of manually adjusting LR mid-training, a scheduler automates the process.

---

### Scheduler Implementation

The `Scheduler` class wraps around a PyTorch optimizer. It is initialized with a few key parameters:

```python
class Scheduler:
    def __init__(self, torch_optimizer: Optimizer, schedule: str, training_steps: int,
                 warmup_steps: int, max_lr: float, min_lr: float):
        # schedule ∈ ["cosine", "linear", "constant"]
        # training_steps = total number of steps
        # warmup_steps = steps spent ramping LR up
        # max_lr = peak LR
        # min_lr = final LR (ignored for "constant")
```

- **schedule**: strategy ("cosine", "linear", or "constant").  
- **training_steps**: total steps in training run.  
- **warmup_steps**: number of warmup steps (linear ramp up).  
- **max_lr**: highest LR used during training.  
- **min_lr**: final LR (for decay-based schedules).  

---

**Warmup**

During warmup, LR increases linearly from near zero to `max_lr`:

```python
def _update_warmup(self, current_step: int):
    lr = (max(1, current_step) / self.warmup_steps) * self.max_lr
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr
    return lr
```

This prevents unstable updates at the beginning of training.

---

**Cosine Decay**

Cosine decay smoothly lowers the LR from `max_lr` to `min_lr`:

```python
def _update_cosine(self, current_step: int):
    current_step -= self.warmup_steps
    scale = (current_step / self.decay_steps) * math.pi
    lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(scale))
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr
    return lr
```

This schedule is popular in modern LLM training because it decays aggressively at first, then flattens out.

---

**Linear Decay**

Linear decay reduces LR steadily over time:

```python
def _update_linear(self, current_step: int):
    current_step -= self.warmup_steps
    lr = self.max_lr - (current_step / self.decay_steps) * (self.max_lr - self.min_lr)
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr
    return lr
```

Simpler than cosine, but still effective.

---

**Constant**

Sometimes you may want to keep LR fixed at `max_lr` (e.g., for debugging).

```python
if schedule == "constant":
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = max_lr
```

---

**Step Method**

The central logic is in the `step` method, which updates LR depending on the phase of training:

```python
def step(self, current_step: int):
    if current_step < self.warmup_steps and self.schedule != "constant":
        self.current_lr = self._update_warmup(current_step)
        return

    if self.schedule == "cosine":
        self.current_lr = self._update_cosine(current_step)
    elif self.schedule == "linear":
        self.current_lr = self._update_linear(current_step)
    elif self.schedule == "constant":
        self.current_lr = self.max_lr
```

This ensures the correct schedule is applied at every step.

---

### Visualizing the Schedules

To make things concrete, below are plots showing how the LR evolves across steps:
(All are 100k total steps, 1k of which is warmup steps, max_lr set to 1e-3 and min_lr set to 1e-4)

**Cosine with Warmup:**

![Cosine LR](../../images/lr_cosine.png)

**Linear with Warmup:**

![Linear LR](../../images/lr_linear.png)

**Constant LR:**

![Constant LR](../../images/lr_constant.png)

You can generate these plots using the included test script in the class (`__main__` block).

---

### Summary

- **Warmup** prevents instability at the start of training.  
- **Cosine decay** → smooth, effective, widely used in LLMs.  
- **Linear decay** → simpler, still works well.  
- **Constant** → mostly for experiments/debugging.  

This custom scheduler is flexible, checkpointable, and provides good control for projects like this.





