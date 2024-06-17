import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb

# Generate a grid of a and b values in the range of -128 to 127
a_values = np.linspace(-128, 127, 256)
b_values = np.linspace(-128, 127, 256)

# Create a meshgrid
a, b = np.meshgrid(a_values, b_values)
L = 80  # Fix L value to 50 for visualization

# Combine L, a, and b to form Lab values
lab_colors = np.stack([np.full_like(a, L), a, b], axis=-1)

# Convert Lab colors to RGB for visualization
rgb_colors = lab2rgb(lab_colors)

# Create a mask to filter out invalid colors
valid_mask = np.all((rgb_colors >= 0) & (rgb_colors <= 1), axis=-1)

# Apply the mask
valid_rgb_colors = np.where(valid_mask[..., np.newaxis], rgb_colors, 1)

# Plot the valid colors
plt.figure(figsize=(8, 8))
plt.imshow(valid_rgb_colors, extent=(-128, 127, -128, 127), origin="lower")
plt.xlabel("a")
plt.ylabel("b")
plt.title("Lab Color Space (L=50)")
plt.grid(False)
plt.show()
