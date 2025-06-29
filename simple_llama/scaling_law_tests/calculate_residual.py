import numpy as np
import matplotlib.pyplot as plt


"""
Using the results from 'hoffman_equation_fit.py' file, the resulting calculations seems a bit off
Not wildly off, but compared to the values in the testing models, deviations range from ~[-0.17, +0.17]
Which on this scale is fairly substantial. Especially considering these are datapoints used to fit the equation


Plot:  
The values on the x-axis is based on the pattern
[55M Param, 69M Param, ... 417M Param] model trained on 0.5B tokens
Then the next 6 data points is the same pattern,
[55M Param, 69M Param, ... 417M Param] model but trained on 0.8B tokens
And so on. 


Observations:
There is clearly a pattern here. 
Based on the 36 datapoints gathered, as the model size and tokens trained scales up, so does the residual. 
More precisely, as those two values increases, the residual increases, meaning the equation to overestimate the loss value
"""


# GPT helped me created the following based on my supplied datapoints:

# Define model sizes and token counts (scaled appropriately)
model_sizes = np.array([55.1, 69.2, 119.6, 170.9, 222.3, 417.5] * 6) * 1e6   # in raw parameter count
token_counts = np.array(
    [0.5] * 6 + [0.8] * 6 + [1.3] * 6 + [1.7] * 6 + [2.5] * 6 + [3.8] * 6
) * 1e9  # in raw token count

# Observed training losses
observed_losses = np.array([
    3.459, 3.394, 3.258, 3.202, 3.145, 3.065,
    3.234, 3.176, 3.048, 2.992, 2.935, 2.854,
    3.083, 3.024, 2.905, 2.849, 2.756, 2.674,
    2.973, 2.907, 2.787, 2.735, 2.721, 2.636,
    2.919, 2.848, 2.727, 2.654, 2.606, 2.524,
    2.865, 2.794, 2.671, 2.596, 2.509, 2.387
])

# From c, alpha, and beta is fixed from Hoffman
c = 1.69
alpha = 0.34
beta = 0.28
A = 230.307
B = 288.669

# From full equation fit
# c = 1.7798
# A = 422.9894
# B = 90595.5491
# alpha = 0.3534
# beta = 0.5753

# Predicted loss function
def predicted_loss(N, D):
    return c + A / (N ** alpha) + B / (D ** beta)


# Compute predicted losses
predicted_losses = predicted_loss(model_sizes, token_counts)


# Compute residuals
residuals = predicted_losses - observed_losses

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(range(len(residuals)), residuals, color='red', label='Residuals (Predicted - Observed)')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Data Point Index')
plt.ylabel('Residual (Loss Error)')
plt.title('Residuals Between Predicted and Observed Losses')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
