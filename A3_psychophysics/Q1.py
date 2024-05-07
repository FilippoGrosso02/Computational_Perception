import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

np.random.seed(2) 
# Function to create a 2D Gabor filter
def make2DGabor(M, kx, ky):
    
    sigma = 0.5 * M / np.sqrt(kx**2 + ky**2)
    x = np.linspace(-M//2, M//2, M)
    y = np.linspace(-M//2, M//2, M)
    X, Y = np.meshgrid(x, y)
    cos2D = np.cos(2 * np.pi * (kx * X + ky * Y) / M)
    sin2D = np.sin(2 * np.pi * (kx * X + ky * Y) / M)
    g = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    Gaussian2D = g / (2 * np.pi * sigma**2)
    cosGabor = Gaussian2D * cos2D
    sinGabor = Gaussian2D * sin2D
    return cosGabor, sinGabor


# Function to simulate the response of a binocular complex cell with a disparity shift
def simulate_binocular_cell_response(image_size, gabor_filter, disparityx, disparityy):
    mean_response = np.zeros((len(velocities_y), len(velocities_x)))

    shifted_cosGabor = np.roll(np.roll(gabor_filter[0], disparityx, axis=1), disparityy, axis=0)
    shifted_sinGabor = np.roll(np.roll(gabor_filter[1], disparityx, axis=1), disparityy, axis=0)
    
    noise_image = np.random.rand(*image_size)
    left_image = noise_image 

    response_left_cos = convolve2d(left_image, gabor_filter[0], mode='valid')
    response_left_sin = convolve2d(left_image, gabor_filter[1], mode='valid')

    response_cos = convolve2d(left_image, shifted_cosGabor, mode='valid')
    response_sin = convolve2d(left_image, shifted_sinGabor, mode='valid')

    for i, vx in enumerate(velocities_x):
        for j, vy in enumerate(velocities_y):

            rolled_response_cos = np.roll(np.roll(response_cos, -vx, axis=1), vy, axis=0)
            rolled_response_sin = np.roll(np.roll(response_sin, -vx, axis=1), vy, axis=0)

            # Compute mean response
            mean_response[j, i] = np.mean(np.sqrt((response_left_cos - rolled_response_cos)**2 + (response_left_sin - rolled_response_sin)**2))
    
    return mean_response

#-----------------------------------------------
# Set up parameters
image_size = (256, 256)
velocities_x = list(range(-8, 9)) 
velocities_y = list(range(-8, 9)) 
orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Orientations in radians for 0, 45, 90, and 135 degrees

# Initialize the mean response matrix for each orientation
mean_responses = {theta: np.zeros((len(velocities_x), len(velocities_y))) for theta in orientations}

# Generate velocity pairs and compute mean responses for each orientation

# --------------- Mean responses 0 orientation
plt.figure(figsize=(6, 4))
plt.suptitle('Orientation: 0 degrees')
plt.subplot(111)
kx = np.cos(0) * 4  # 4 cycles per M samples, adjust as necessary
ky = np.sin(0) * 4
gabor_filter = make2DGabor(32, kx, ky)
a0 = simulate_binocular_cell_response(image_size, gabor_filter,2,0)
plt.imshow(a0, extent=[velocities_x[0], velocities_x[-1], velocities_y[0], velocities_y[-1]])

# --------------- Mean responses 45 orientation
plt.figure(figsize=(6, 4))
plt.suptitle('Orientation: 45 degrees')
plt.subplot(111)
kx = np.cos(45) * 4  # 4 cycles per M samples, adjust as necessary
ky = np.sin(45) * 4
gabor_filter = make2DGabor(32, kx, ky)
a45 = simulate_binocular_cell_response(image_size, gabor_filter,1,1)
plt.imshow(a45, extent=[velocities_x[0], velocities_x[-1], velocities_y[0], velocities_y[-1]])

# --------------- Mean responses 90 orientation
plt.figure(figsize=(6, 4))
plt.suptitle('Orientation: 90 degrees')
plt.subplot(111)
kx = np.cos(90) * 4  # 4 cycles per M samples, adjust as necessary
ky = np.sin(90) * 4
gabor_filter = make2DGabor(32, kx, ky)
a90 = simulate_binocular_cell_response(image_size, gabor_filter,0,0)
plt.imshow(a90, extent=[velocities_x[0], velocities_x[-1], velocities_y[0], velocities_y[-1]])

# --------------- Mean responses 135 orientation
plt.figure(figsize=(6, 4))
plt.suptitle('Orientation: 135 degrees')
plt.subplot(111)
kx = np.cos(135) * 4  # 4 cycles per M samples, adjust as necessary
ky = np.sin(135) * 4
gabor_filter = make2DGabor(32, kx, ky)
a135 = simulate_binocular_cell_response(image_size, gabor_filter,1,-1)
plt.imshow(a135, extent=[velocities_x[0], velocities_x[-1], velocities_y[0], velocities_y[-1]])


# change to (6,0) to build a different high order cell
true_vx = 2
true_vy = 0

summed_mean_responses = np.zeros((len(velocities_y), len(velocities_x)))

# 0 degree gabor
summed_mean_responses += simulate_binocular_cell_response(image_size, gabor_filter, true_vx * 1, 0)
#45  degree
summed_mean_responses += simulate_binocular_cell_response(image_size, gabor_filter, round((true_vx + true_vy)*0.5), round((true_vx + true_vy)*0.5))
#90 degree 
summed_mean_responses += simulate_binocular_cell_response(image_size, gabor_filter, 0, true_vy*1)
#45  degree
summed_mean_responses += simulate_binocular_cell_response(image_size, gabor_filter, round((true_vx - true_vy)*0.5), round((-true_vx + true_vy)*0.5))

mean_of_summed_responses = summed_mean_responses / len(orientations)

# Plot the mean of the summed responses
plt.figure(figsize=(8, 6))
plt.title('Mean of Summed Responses for Four Normal Velocity Cells')
plt.imshow(mean_of_summed_responses, extent=[velocities_x[0], velocities_x[-1], velocities_y[0], velocities_y[-1]])
plt.xlabel('Velocity X')
plt.ylabel('Velocity Y')
plt.colorbar()
plt.show()