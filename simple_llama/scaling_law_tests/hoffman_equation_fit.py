import numpy as np
from scipy.optimize import curve_fit


model_sizes = [55.1, 69.2, 119.6, 170.9, 222.3, 417.5] * 6   # Model size in millions
num_tokens = [0.5] * 6 + [0.8] * 6 + [1.3] * 6 + [1.7] * 6 + [2.5] * 6 + [3.8] * 6    # Token in billions

losses = [
    # 0.5B tokens
    3.459, 3.394, 3.258, 3.202, 3.145, 3.065,
    # 0.8B tokens
    3.234, 3.176, 3.048, 2.992, 2.935, 2.854,
    # 1.3B tokens
    3.083, 3.024, 2.905, 2.849, 2.756, 2.674,
    # 1.7B tokens
    2.973, 2.907, 2.787, 2.735, 2.721, 2.636,
    # 2.5B tokens
    2.919, 2.848, 2.727, 2.654, 2.606, 2.524,
    # 3.8B tokens
    2.865, 2.794, 2.671, 2.596, 2.509, 2.387
]



N_vals = np.array(model_sizes) * 1e6        # model sizes
D_vals = np.array(num_tokens) * 1e9       # tokens in billions
L_vals = np.array(losses)             # training losses


# Fixed parameters from Hoffman
c = 1.69
alpha = 0.34
beta = 0.28

# Other fixed parameters
# c = 1.8172
# alpha = 0.3478
# beta = 0.3658


# Define the loss model
def loss_model(xdata, A, B):
    N, D = xdata
    return c + A / (N**alpha) + B / (D**beta)


# Prepare inputs
xdata = np.vstack((N_vals, D_vals))


# Fit A and B
popt, pcov = curve_fit(loss_model, xdata, L_vals)
A_fit, B_fit = popt


print(f"Fitted A: {A_fit:.3f}, Fitted B: {B_fit:.3f}")
