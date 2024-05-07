import numpy as np

from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def make2DGabor(M, kx, ky):
    
    k     =  np.sqrt(kx*kx + ky*ky)
    sigma =  0.5 * M/k        # N/k is number of pixels in one wavelength 

    x = range(-M//2, M//2)
    y = range(-M//2, M//2)
    [X,Y] = np.meshgrid(x,y)
    cos2D = np.cos(  2*np.pi/M * (kx * X + ky * Y) )
    sin2D = np.sin(  2*np.pi/M * (kx * X + ky * Y) )

#  Here the x are the column indices and y are the row indices, which is
#  what we want.   x increases to right and y increases down.

    xarray = np.array(x)
    g = 1/(np.sqrt(2*np.pi)*sigma) *  np.exp(- xarray*xarray / (2 * sigma*sigma) )
    Gaussian2D = np.outer(g,g)

    cosGabor = Gaussian2D * cos2D
    sinGabor = Gaussian2D * sin2D
    return cosGabor, sinGabor

def make_sin(N,k):
    x = np.linspace(-N/2, N/2, N)
    y = np.linspace(-N/2, N/2, N)
    X, Y = np.meshgrid(x, y)

    I = (1 + np.sin(2 * np.pi * k * X / N)) / 2

    return I

def make_noise(N):
    image = np.random.randint(2, size=(N, N))
    return image
def simulateComplexCellResponse(image, gabor_filter):

    convolved_image = np.sqrt(convolve2d(image, gabor_filter[0], mode = 'valid')**2 + convolve2d(image, gabor_filter[1], mode = 'valid')**2)
    return convolved_image

def simulateCellResponse(image, gabor_filter):
    convolved_image = convolve2d(image, gabor_filter, mode = 'valid')

    return convolved_image

# Parameters 
N = 256
M = 32
k_values = np.arange(1, 128, 1)  # Coarse frequency range for example

# Initialize variables for storing the optimal responses and k-values
max_mean_response_cos = 0
max_mean_response_sin = 0
max_mean_response_complex = 0
optimal_k_cos = k_values[0]
optimal_k_sin = k_values[0]
optimal_k_complex = k_values[0]

# Create 1st figure
plt.figure(figsize=(12,6))

# Processing for cosine Gabor
for k in k_values:
    sin_image = make_sin(N, k)
    gabor_filter = make2DGabor(M, 4, 0)
    mean_response = np.mean(simulateCellResponse(sin_image, gabor_filter[0]))
    if mean_response > max_mean_response_cos:
        max_mean_response_cos = mean_response
        optimal_k_cos = k

print(optimal_k_cos)
optimal_sin_image = make_sin(N, optimal_k_cos)
cos_response = simulateCellResponse(optimal_sin_image, make2DGabor(M, 4, 0)[0])

# Plot cosine Gabor response
plt.subplot(2, 3, 1)
plt.imshow(cos_response, cmap='gray')
plt.title('Cosine Gabor Response')

# Plot histogram for cosine Gabor response
plt.subplot(2, 3, 4)  # 2nd row for the histogram
plt.hist(cos_response.ravel(), bins=50, color='gray')
plt.title('Histogram of Cosine Gabor Response')

# Processing and plotting for sine Gabor
for k in k_values:
    sin_image = make_sin(N, k)
    gabor_filter = make2DGabor(M, 4, 0)
    mean_response = np.mean(simulateCellResponse(sin_image, gabor_filter[1]))
    if mean_response > max_mean_response_sin:
        max_mean_response_sin = mean_response
        optimal_k_sin = k

print(optimal_k_sin)
optimal_sin_image = make_sin(N, optimal_k_sin)
sine_response = simulateCellResponse(optimal_sin_image, make2DGabor(M, 4, 0)[1])

# Plot sine Gabor response
plt.subplot(2, 3, 2)  
plt.imshow(sine_response, cmap='gray')
plt.title('Sine Gabor Response')

# Plot histogram for sine Gabor response
plt.subplot(2, 3, 5) 
plt.hist(sine_response.ravel(), bins=50, color='gray')
plt.title('Histogram of Sine Gabor Response')

# Processing and plotting for complex cell
K_mean_values = []
for k in k_values:
    sin_image = make_sin(N, k)
    gabor_filter = make2DGabor(M, 4, 0)
    mean_response = np.mean(simulateComplexCellResponse(sin_image, gabor_filter))
    K_mean_values.append(mean_response)
    if mean_response > max_mean_response_complex:
        max_mean_response_complex = mean_response

        optimal_k_complex = k
        
print(optimal_k_complex)
optimal_sin_image = make_sin(N, optimal_k_complex)
complex_cell_response = simulateComplexCellResponse(optimal_sin_image, make2DGabor(M, 4, 0))

# Plot complex cell response
plt.subplot(2, 3, 3)
plt.imshow(complex_cell_response, cmap='gray')
plt.title('Complex Cell Response')

# Plot histogram for complex cell response
plt.subplot(2, 3, 6)  
plt.hist(complex_cell_response.ravel(), bins=50, color='gray')
plt.title('Histogram of Complex Cell Response')


# Display the figure with all subplots



#----------------------------------------------------------


# RANDOM NOISE
# Parameters # for make sin
N = 256
M = 32

noise_img = make_noise(N)
# Create a figure for subplots
plt.figure(figsize=(12, 6))

cos_response = simulateCellResponse(noise_img, make2DGabor(M, 4, 0)[0])

# Plot cosine Gabor response
plt.subplot(2, 3, 1)  
plt.imshow(cos_response, cmap='gray')
plt.title('Cosine Gabor Response')

# Plot histogram for cosine Gabor response
plt.subplot(2, 3, 4)  
plt.hist(cos_response.ravel(), bins=50, color='gray')
plt.title('Histogram of Cosine Gabor Response')

sine_response = simulateCellResponse(noise_img, make2DGabor(M, 4, 0)[1])

# Plot sine Gabor response
plt.subplot(2, 3, 2) 
plt.imshow(sine_response, cmap='gray')
plt.title('Sine Gabor Response')

# Plot histogram for sine Gabor response
plt.subplot(2, 3, 5)  
plt.hist(sine_response.ravel(), bins=50, color='gray')
plt.title('Histogram of Sine Gabor Response')

complex_cell_response = simulateComplexCellResponse(noise_img, make2DGabor(M, 4, 0))

# Plot complex cell response
plt.subplot(2, 3, 3)  
plt.imshow(complex_cell_response, cmap='gray')
plt.title('Complex Cell Response')

# Plot histogram for complex cell response
plt.subplot(2, 3, 6) 
plt.hist(complex_cell_response.ravel(), bins=50, color='gray')
plt.title('Histogram of Complex Cell Response')

complex_mean_response = np.mean(complex_cell_response)
max_k_index = np.argmax(K_mean_values)
max_k_value = k_values[max_k_index]

plt.figure(figsize=(8, 6))

# Plotting K_mean_values
plt.plot(k_values, K_mean_values, label='Complex Cell Mean Response')

# Adding title and axis labels
plt.title('Mean Values vs. K')
plt.xlabel('K')
plt.ylabel('Mean complex response')

# Adding a horizontal line at the value of complex mean response
plt.axhline(y=complex_mean_response, color='r', linestyle='--', label='Noise Complex Mean Response')

# Highlighting the k value for which the two lines intercept
plt.scatter(max_k_value, complex_mean_response, color='black', label=f'Max K Mean Value: {max_k_value}')

# Adding legend
plt.legend()

# Show plot
plt.show()