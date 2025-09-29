# Optimizers: From SGD to AdamW

**Note:** This section covers four important optimizers used for updating model weights via backpropagation: SGD, SGD with Momentum, Adam, and AdamW. While other optimizers like AdaGrad and RMSProp exist, they fall outside the scope of this guide.

This section is heavy in mathematical derivation and can be skipped without impacting further sections. It assumes the reader has a fair understanding of backpropagation and gradient computation. If not, the videos [DL Chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U) and [DL Chapter 4](https://www.youtube.com/watch?v=tIeHLnjs5U8) by 3Blue1Brown are highly recommended.

---

## SGD: The Fundamental Optimizer

Stochastic Gradient Descent (SGD) is the most basic and historically important optimizer, forming the foundation for more advanced variants.

The parameter update rule is:

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L(\theta_t)
$$

**Key Components:**  

- $\theta_{t+1}$: The updated parameter.
- $\theta_t$: The current parameter.
- $\eta$: The learning rate.
- $\nabla_\theta L(\theta_t)$: The gradient of the loss function with respect to the parameter $\theta$ at timestep $t$.

The update is performed by subtracting the product of the learning rate and the gradient from the current parameter value. This nudges the parameter value towards a (local) minimum. Over many iterations, this process converges to a final set of parameters for the model. It is simple, yet effective.

---

## SGD With Momentum

SGD with Momentum is a variant designed to accelerate convergence and reduce oscillations (the "zigzag" behavior) often seen in vanilla SGD.

It introduces a velocity term, which is a running average of past gradients.

### Velocity Update
$$
v_t = \beta v_{t-1} + \nabla_\theta L(\theta_t)
$$

### Parameter Update
$$
\theta_{t+1} = \theta_t - \eta v_t
$$

We introduce a new velocity term, $v_t$, which accumulates the gradient direction. This velocity is composed of the current batch's gradient plus a fraction of the previous timestep's velocity ($\beta v_{t-1}$). The hyperparameter $\beta$ is typically set to a value like 0.9.

The learning rate $\eta$ then scales the entire velocity term in the final parameter update. This separation makes the roles clear: the velocity determines the **direction** and the learning rate determines the **step size**.

The purpose is to 'smooth out' the path taken during descent. The velocity term acts as a weighted sum of past gradients, giving inertia to the optimization process. If recent gradients point in a consistent direction, the velocity builds up, leading to faster progress.

Let's trace the first few steps, starting at $t=1$ (assuming $v_0 = 0$):  

- **$t=1$:** $v_1 = \beta \cdot 0 + g_1 = g_1$ → $\theta_2 = \theta_1 - \eta g_1$
- **$t=2$:** $v_2 = \beta v_1 + g_2 = \beta g_1 + g_2$ → $\theta_3 = \theta_2 - \eta (\beta g_1 + g_2)$
- **$t=3$:** $v_3 = \beta v_2 + g_3 = \beta^2 g_1 + \beta g_2 + g_3$ → $\theta_4 = \theta_3 - \eta (\beta^2 g_1 + \beta g_2 + g_3)$

As this process continues, the influence of a gradient from step $t-i$ is scaled by $\beta^i$. With $\beta=0.9$, the optimizer "remembers" the direction of the last ~30 steps. This leads to a smoother, faster path towards the minimum.

The general form of the velocity at step $t$ can be expressed as:
$$
v_t = \sum_{i=1}^{t} \beta^{t-i} \nabla_\theta L(\theta_i)
$$
---

## Adam Optimizer

The Adam (Adaptive Moment Estimation) optimizer, introduced in 2015, builds upon SGD with Momentum by incorporating an adaptive learning rate for each parameter. It uses estimates of both the first moment (the mean of gradients, like momentum) and the second moment (the uncentered variance of gradients) of the gradients.

Adam introduces a third hyperparameter, $\beta_2$, which controls the decay rate for the second moment ($\beta$ in SGD With Momentum is now referred to as $\beta_1$). Let $g_t$ be the shorthand for $\nabla_\theta L(\theta_t)$.

The parameter update rule is:

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

We update the parameter by subtracting the bias-corrected first moment ($\hat m_t$), normalized by the square root of the bias-corrected second moment ($\sqrt{\hat v_t}$), and scaled by the learning rate $\eta$. A small constant $\epsilon$ (e.g., $10^{-8}$) is added to prevent division by zero.

### First Moment (Momentum)
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

This is similar to the velocity in SGD with Momentum. The key difference is the use of $(1-\beta_1)$ as a scaling factor for the current gradient. This design, along with the subsequent bias correction, ensures that the expected value of $m_t$ is approximately equal to the expected value of the gradients $E[g_t]$. The purpose is identical: to smooth the gradient signal by incorporating past information.

### Second Moment (Variance)
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

This calculation is analogous to the first moment but uses a different hyperparameter, $\beta_2$ (typically set to 0.999), and operates on the squared gradients. The squared value provides a measure of the magnitude of the gradients, irrespective of their sign. A large $v_t$ indicates that the parameter has historically had large gradients. This term is used to adaptively scale the learning rate down for such parameters.

### Bias Correction
A critical step in Adam is bias correction. At the beginning of training (low $t$), the moving averages $m_t$ and $v_t$ are initialized to zero and are therefore biased towards zero. Bias correction compensates for this initial bias.

$$
\hat m_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat v_t = \frac{v_t}{1 - \beta_2^t}
$$

Let's see why this is necessary at $t=1$:  

- $m_1 = \beta_1 \cdot 0 + (1-\beta_1) g_1 = (1-0.9) g_1 = 0.1 \cdot g_1$. This is much smaller than the true gradient $g_1$.
- $\hat m_1 = \frac{m_1}{1 - 0.9^1} = \frac{m_1}{0.1} = g_1$. The bias correction scales $m_1$ back up to the correct magnitude.

As $t$ increases, $\beta^t$ approaches zero, and the denominator $1-\beta^t$ approaches 1, making the bias correction factor negligible. This is why Adam is called an *adaptive learning rate* optimizer: the effective step size for each parameter is $\eta / \sqrt{\hat v_t}$, which adjusts based on the historical magnitude of the parameter's gradients.

---

## AdamW Optimizer

AdamW is a refined version of Adam that is arguably the most widely used optimizer today. The key difference lies in how it handles **Weight Decay**.

### The Problem with L2 Regularization in Adam

Traditionally, L2 regularization is added directly to the loss function:
$$
L_{total}(\theta) = L_{original}(\theta) + R(\theta) = L_{original}(\theta) + \frac{\lambda}{2} \|\theta\|^2
$$

The gradient of the regularization term is $\nabla_\theta R(\theta) = \lambda \theta$. Therefore, the gradient used for the update in standard Adam becomes:
$$
\nabla L_{total} = \nabla L_{original}(\theta_t) + \lambda \theta_t
$$

This intertwining of weight decay with the adaptive gradient mechanism is suboptimal. Because Adam adapts the learning rate per parameter based on $v_t$, the effective strength of the weight decay becomes coupled with the history of the gradients. This coupling can prevent weight decay from acting as a consistent regularizer, often making the hyperparameter $\lambda$ harder to tune and leading to worse generalization.

### The Solution: Decoupled Weight Decay

AdamW decouples the weight decay term from the adaptive gradient update. Instead of adding it to the loss, weight decay is applied directly to the weights *after* the adaptive update.

The parameter update rule for AdamW is:

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} \right) - \eta \lambda \theta_t
$$

**Key Difference:** The term $- \eta \lambda \theta_t$ is now separate. It is applied directly to the weights as a straightforward weight decay, independent of the adaptive scaling factor $1/(\sqrt{\hat v_t} + \epsilon)$.

This decoupling ensures that the weight decay is applied with a strength determined solely by the product $\eta \lambda$, making it a true regularization term that acts consistently on the weights. This approach has been shown to improve generalization performance and makes the tuning of the weight decay hyperparameter $\lambda$ more reliable and consistent across different models and datasets. AdamW is often the default choice for training modern deep neural networks, particularly Transformers.


## Optimizer Summary Table

| Optimizer         | Update Rule (simplified)                                                                                                                                                     | Extra Hyperparams      | Intuition                                                                                   | Pros                                                                                  | Cons                                                                 |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|--------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| **SGD**           | $\theta_{t+1} = \theta_t - \eta g_t$                                                                                                                                         | –                      | Take a step opposite the gradient.                                                         | Simple, fast, widely understood.                                                     | Can be very noisy, slow convergence.                                |
| **SGD + Momentum**| $v_t = \beta v_{t-1} + g_t$ <br> $\theta_{t+1} = \theta_t - \eta v_t$                                                                                                        | $\beta$ (momentum)     | Averages past gradients to smooth updates (“ball rolling downhill”).                       | Faster convergence, reduces zigzagging.                                              | Still uses global LR, sensitive to hyperparam tuning.               |
| **Adam**          | $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ <br> $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ <br> $\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}$ | $\beta_1$, $\beta_2$, $\epsilon$ | Combines momentum (trend of gradients) with variance scaling (stability).                  | Adaptive per-parameter learning rate; works well out of the box.                     | Can overfit, sometimes worse generalization than SGD.                |
| **AdamW**         | $\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} - \eta \lambda \theta_t$                                                                         | $\beta_1$, $\beta_2$, $\epsilon$, $\lambda$ | Same as Adam, but with **decoupled weight decay**.                                          | Strong generalization; default choice for Transformers and large-scale models.       | More compute/memory than SGD; still sensitive to hyperparams.       |
