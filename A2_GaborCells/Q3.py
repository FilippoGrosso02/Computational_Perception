import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def make2DGabor(M, kx, ky):
    k = np.sqrt(kx**2 + ky**2)
    sigma = 0.5 * M / k
    x = np.arange(-M//2, M//2)
    y = np.arange(-M//2, M//2)
    X, Y = np.meshgrid(x, y)
    cos2D = np.cos(2 * np.pi / M * (kx * X + ky * Y))
    sin2D = np.sin(2 * np.pi / M * (kx * X + ky * Y))
    g = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-x**2 / (2 * sigma**2))
    Gaussian2D = np.outer(g, g)
    cosGabor = Gaussian2D * cos2D
    sinGabor = Gaussian2D * sin2D
    return cosGabor, sinGabor

def generate_stereo_image_pairs(image_size, disparity):
    noise_image = np.random.rand(*image_size)
    left_image = noise_image
    right_image = np.roll(noise_image, -disparity, axis=1)
    return left_image, right_image

def simulateComplexCellResponse(image, gabor_filter):

    convolved_image = np.sqrt(convolve2d(image, gabor_filter[0], mode = 'valid')**2 + convolve2d(image, gabor_filter[1], mode = 'valid')**2)
    return convolved_image

def simulate_binocular_cell_response(left_image, right_image, gabor_filter, disparity):

    shifted_cosGabor = np.roll(gabor_filter[0], disparity, axis=1)
    shifted_sinGabor = np.roll(gabor_filter[1], disparity, axis=1)

    response_left_cos = convolve2d(left_image, gabor_filter[0], mode='valid')
    response_right_cos = convolve2d(right_image, shifted_cosGabor, mode='valid')
    
    response_left_sin = convolve2d(left_image, gabor_filter[1], mode='valid')
    response_right_sin = convolve2d(right_image, shifted_sinGabor, mode='valid')

    # Combine responses for complex cell simulation
    complex_response = np.sqrt((response_left_cos - response_right_cos)**2 + (response_left_sin - response_right_sin)**2)

    return np.mean(complex_response)



# Setup parameters
plt.figure(figsize=(12, 6))
orientations = [0, 45, 90, 135]
image_size = (256, 256)



#Lets do stereo images with disparity 0

left_image, right_image = generate_stereo_image_pairs(image_size, 0)
disparities = list(range(-8, 9)) 
orientations = [0, 45, 90, 135]

for theta in orientations:
    responses = {}
    gabor_filter = make2DGabor(128, np.cos(np.deg2rad(theta)) * 4, np.sin(np.deg2rad(theta)) * 4)
    for disparity in disparities:
        

        responses[disparity] = simulate_binocular_cell_response(left_image, right_image, gabor_filter, disparity)
    plt.plot(disparities,responses.values(), label=f'Bin. cell w. orientation {theta}')



plt.title("Disparity 0 - Mean Response by Tuned Disparity ")
plt.xlabel("Tuned Disparity")
plt.ylabel("Mean Response")
plt.legend()

# ---------------------------------------------------------------------------------------------------------------
# Setup parameters Disparity 2
plt.figure(figsize=(12, 6))
orientations = [0, 45, 90, 135]
image_size = (256, 256)



#lets do stereo images with disparity 2

left_image, right_image = generate_stereo_image_pairs(image_size, 2)
disparities = list(range(-8, 9)) 
orientations = [0, 45, 90, 135]

for theta in orientations:
    responses = {}
    gabor_filter = make2DGabor(128, np.cos(np.deg2rad(theta)) * 4, np.sin(np.deg2rad(theta)) * 4)
    for disparity in disparities:
        

        responses[disparity] = simulate_binocular_cell_response(left_image, right_image, gabor_filter, disparity)
    plt.plot(disparities,responses.values() , label=f'Bin. cell w. orientation {theta}')


plt.title("Disparity 2 - Mean Response by Tuned Disparity ")
plt.xlabel("Tuned Disparity")
plt.ylabel("Mean Response")
plt.legend()

plt.show()

