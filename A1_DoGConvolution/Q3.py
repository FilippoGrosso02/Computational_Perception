import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def make_image():
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    colors = {
        'red': [255, 0, 0],
        'green': [0, 255, 0],
        'black': [0, 0, 0],
        'white': [255, 255, 255],
        'blue': [0, 0, 255]
    }

    image[:, :64] = colors['red']
    image[:, 64:128] = colors['green']
    image[:, 128:192] = colors['black']
    image[:, 192:] = colors['white']

    square_size = 10  

    points_positions = [
        (32, 64), (32, 128), (32, 192),   # Red block
        (96, 64), (96, 128), (96, 192),   # Green block
        (160, 64), (160, 128), (160, 192), # Black block
        (224, 64), (224, 128), (224, 192)  # White block
    ]

    point_colors = {
        'black': [0, 0, 0],
        'green': [0, 255, 0],
        'white': [255, 255, 255],
        'red': [255, 0, 0]
    }

    color_sequence = ['black', 'green', 'white', 'black', 'red', 'white', 'green', 'white', 'red', 'green', 'black', 'red']

    for (pos, color_key) in zip(points_positions, color_sequence):
        x, y = pos
        # Calculate the top-left corner of the square
        x_start = x - square_size // 2
        y_start = y - square_size // 2
        # Draw the square onto the image
        image[y_start:y_start+square_size, x_start:x_start+square_size] = point_colors[color_key]

    return image

def gaussian_template(size, sigma=4):
    """Generate a Gaussian template."""
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

img = make_image()

gaussian1 = gaussian_template(50, 4)
gaussian2 = gaussian_template(50, 5)
DoG = gaussian1 - gaussian2

# EXTRACT RED PART
red_channel = img[:, :, 0]
red_convolved = convolve(red_channel.astype(float), DoG, mode="reflect")


# EXTRACT GREEN PART
green_channel = img[:, :, 1]
green_convolved = convolve(green_channel.astype(float), DoG, mode="reflect")


# TOTAL

total_convolved = red_convolved  + green_convolved

plt.imshow(total_convolved, cmap='viridis')
plt.axis('off')
plt.show()

#fig, ax = plt.subplots()
#ax.imshow(img)

#ax.axis('off')  # Turn off the axes
#plt.show()