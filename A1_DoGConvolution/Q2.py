import numpy as np
from scipy.signal import correlate2d
import matplotlib.pyplot as plt

def gaussian_template(size, sigma=4):
    """Generate a Gaussian template."""
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

def michelson(max_value, min_value):
    """Calculate the Michelson contrast."""
    if max_value + min_value == 0:
        return 0  # Return 0 or another appropriate value when max + min is zero
    return (max_value - min_value) / (max_value + min_value)

def rms(intensity):
    mean_intensity = np.mean(intensity)
    return np.sqrt(np.mean((intensity - mean_intensity) ** 2))
N = 256
k_values = range(1, N//2)  # Spatial frequencies

# Initialize arrays to store the Michelson contrasts
michelson_contrasts1 = []
michelson_contrasts2 = []
rms_values = []

# Create a grid of coordinates (x, y) with the origin at the center
x = np.linspace(-N/2, N/2, N)
y = np.linspace(-N/2, N/2, N)
X, Y = np.meshgrid(x, y)

for k in range(1,N//2):
    # Calculate the raised sinusoid
    I = (1 + np.sin(2 * np.pi * k * X / N)) / 2

    gaussian1 = gaussian_template(50,4)
    gaussian2 = gaussian_template(50,5)
    DoG = gaussian1 - gaussian2

    # Perform cross-correlation between the image and the Gaussian template
    correlation1 = correlate2d(I, gaussian1, mode='valid')
    correlation2 = correlate2d(I, gaussian2, mode='valid')
    correlation_DoG = correlate2d(I, DoG, mode='valid')
    
    michelson1 = michelson(np.max(correlation1), np.min(correlation1))
    michelson2 = michelson(np.max(correlation2), np.min(correlation2))
    rms1 = rms(correlation_DoG )

    michelson_contrasts1.append(michelson1)
    michelson_contrasts2.append(michelson2)
    rms_values.append(rms1)

    # Find the indices of the peak correlation
    #y, x = np.unravel_index(np.argmax(correlation1), correlation1.shape)

plt.figure(figsize=(10, 5))
plt.plot(k_values, michelson_contrasts1, label='Gaussian $\sigma=4$ - Michelson Contrast')
plt.plot(k_values, michelson_contrasts2, label='Gaussian $\sigma=5$ - Michelson Contrast')
plt.plot(k_values, rms_values, label='Difference of Gaussians - RMS Contrast')
max_rms_contrast = max(rms_values)
max_rms_index = rms_values.index(max_rms_contrast)
print(max_rms_index)
plt.xlabel('Spatial Frequency k')
plt.ylabel('Michelson/RMS Contrast')
plt.title('Michelson/RMS Contrast vs. Spatial Frequency')
plt.legend()
plt.grid(True)
plt.show()
