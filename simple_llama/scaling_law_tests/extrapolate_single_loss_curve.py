import matplotlib.pyplot as plt
import numpy as np


"""
Plots the loss curve using values from Hoffman
"""

# Constants from previous fitting
c = 1.69
alpha = 0.34
beta = 0.28
A = 230.307
B = 288.669

# Fixed model size and dataset size
N = 1266.8e6  # 1266.8M parameters
D = 50e9    # 50B tokens

# Simulate training curve: loss as a function of partial token exposure
# We'll simulate the model being trained on a subset of D, ranging from 0.1B to 50B
token_range_partial = np.linspace(0.1e9, D, 100)

# Compute training loss curve
loss_curve = c + A / (N ** alpha) + B / (token_range_partial ** beta)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(token_range_partial / 1e9, loss_curve, label=f'{int(N/1e6)}M params on {int(D/1e9)}B tokens')
plt.xlabel('Tokens Seen (B)')
plt.ylabel('Predicted Training Loss')
plt.title('Predicted Training Loss Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
