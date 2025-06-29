import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Reuse the fitted parameters
c = 1.69
alpha = 0.34
beta = 0.28
A = 230.307
B = 288.669

# Define the sweep ranges
param_range = np.arange(1.0, 1.5 + 0.01, 0.1)  # in billions
token_range = np.arange(40, 60 + 0.01, 1)    # in billions


# Convert to raw units
param_grid, token_grid = np.meshgrid(param_range * 1e9, token_range * 1e9)

# Compute loss surface
loss_surface = c + A / (param_grid ** alpha) + B / (token_grid ** beta)

# Find the combination with the lowest loss
min_index = np.unravel_index(np.argmin(loss_surface), loss_surface.shape)
optimal_N = param_grid[min_index]
optimal_D = token_grid[min_index]
min_loss = loss_surface[min_index]

# Create 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(param_grid / 1e9, token_grid / 1e9, loss_surface, cmap='viridis', alpha=0.9)
ax.set_title('Loss Surface over Model Size and Dataset Size')
ax.set_xlabel('Model Size (B parameters)')
ax.set_ylabel('Dataset Size (B tokens)')
ax.set_zlabel('Predicted Loss')
ax.scatter(optimal_N / 1e9, optimal_D / 1e9, min_loss, color='red', s=50, label='Optimal Point')
ax.legend()
plt.tight_layout()
plt.show()

(optimal_N / 1e9, optimal_D / 1e9, min_loss)
