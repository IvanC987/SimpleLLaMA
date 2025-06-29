import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


"""
Fitting only the values A and B in 'hoffman_equation_fit.py' resulted in fair guesses, but often overestimate the loss. 
That is likely due to me only have 36 datapoints, which isn't sufficient enough to fully fit the equation. 
And so I decided to just fit all 5 values instead of just alpha and beta, hoping it would lead to a better result
"""


# Define the flexible loss model
def flexible_loss_model(xdata, c, A, B, alpha, beta):
    N, D = xdata
    return c + A / (N ** alpha) + B / (D ** beta)


# Input data
model_sizes = np.array([55.1, 69.2, 119.6, 170.9, 222.3, 417.5] * 6) * 1e6  # in raw parameters
token_counts = np.array(
    [0.5] * 6 + [0.8] * 6 + [1.3] * 6 + [1.7] * 6 + [2.5] * 6 + [3.8] * 6
) * 1e9  # in raw tokens


observed_losses = np.array([
    3.459, 3.394, 3.258, 3.202, 3.145, 3.065,
    3.234, 3.176, 3.048, 2.992, 2.935, 2.854,
    3.083, 3.024, 2.905, 2.849, 2.756, 2.674,
    2.973, 2.907, 2.787, 2.735, 2.721, 2.636,
    2.919, 2.848, 2.727, 2.654, 2.606, 2.524,
    2.865, 2.794, 2.671, 2.596, 2.509, 2.387
])


# Stack the input features
xdata = np.vstack((model_sizes, token_counts))


# Initial parameter guess and fitting
initial_guess = [1.5, 100, 100, 0.3, 0.3]
popt, _ = curve_fit(
    flexible_loss_model,
    xdata,
    observed_losses,
    p0=initial_guess,
    bounds=(0, np.inf),
    maxfev=10000
)


# Unpack fitted parameters
c_fit, A_fit, B_fit, alpha_fit, beta_fit = popt

# Predict the final loss for a specific configuration
N_test = 1248e6  # 1248M parameters
D_test = 50e9    # 50B tokens
final_loss = flexible_loss_model((np.array([N_test]), np.array([D_test])), *popt)[0]

# Print results
print("Fitted Parameters:")
print(f"  c      = {c_fit:.4f}")
print(f"  A      = {A_fit:.4f}")
print(f"  B      = {B_fit:.4f}")
print(f"  alpha  = {alpha_fit:.4f}")
print(f"  beta   = {beta_fit:.4f}")
print()
print(f"Predicted Loss for 1248M model on 50B tokens: {final_loss:.4f}")




# Simulate the loss curve across training steps for the 1248M model up to 50B tokens
token_range_partial = np.linspace(0.1e9, D_test, 100)
loss_curve = flexible_loss_model((np.full_like(token_range_partial, N_test), token_range_partial), *popt)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(token_range_partial / 1e9, loss_curve, label='1248M params on up to 50B tokens')
plt.xlabel('Tokens Seen (B)')
plt.ylabel('Predicted Training Loss')
plt.title('Predicted Training Loss Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

