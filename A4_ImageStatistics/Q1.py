import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib.colors as colors
#from Q1_starter_Python import draw_slanted_plane

def deg_to_rad(deg):
    return np.deg2rad(deg)

def rad_to_deg(rad):
    return np.rad2deg(rad)

def xy_to_pix(x, N, FOV_DEG):
    return x / np.tan(deg_to_rad(FOV_DEG / 2)) * N / 2

def XYZ_to_xy(X, Y, Z):
    x = X / Z
    y = Y / Z
    return x, y

def xy_to_xpyp(x, y, N, FOV_DEG):
    return (N / 2 + xy_to_pix(x, N, FOV_DEG), N / 2 + xy_to_pix(y, N, FOV_DEG))

def compute_points_per_bin(N_PIX, N_POINTS, slant, FOV_DEG):
    # Replace with actual logic to compute the number of points per bin.
    return np.random.poisson(lam=50, size=N_PIX)

def draw_slanted_plane(N_PIX, N_POINTS, slant, FOV_DEG):  
    '''   
    N_POINTS is the number of points that would be dropped for the case of slant = 0.
    When slant is different from 0, we will need to drop more dots so
    that dots can hit anywhere in the field of view.
    '''
    f = 50
    
    if abs(slant) >= 90 - FOV_DEG/2:
        raise ValueError("we require FOV_DEG/2 < 90 - slant")
        
    # make a canvas for drawing the image
    img = Image.new("L", (N_PIX, N_PIX), 255)  # Grayscale image with white background
    draw = ImageDraw.Draw(img)

    # Compute the bounds of the plane
    Y = f * np.tan(deg_to_rad(FOV_DEG/2)) / (1 - np.tan(deg_to_rad(FOV_DEG/2)) * (np.tan(deg_to_rad(abs(slant)))))
    delta_Z = Y * np.tan(deg_to_rad(abs(slant)))
    XMAX = (f + delta_Z) * np.tan(deg_to_rad(FOV_DEG/2))
    YMAX =  Y / np.cos(deg_to_rad(abs(slant)))

    # Compute the number of augmented points
    N_POINTS_AUGMENTED = int(N_POINTS * XMAX * YMAX / (f**2 * np.tan(deg_to_rad(FOV_DEG/2))**2))

    for i in range(N_POINTS_AUGMENTED):
        # Drop point randomly in the Z=f plane
        X = XMAX * (2*random.random()-1)
        Y0 = YMAX * (2*random.random()-1)

        # Rotate plane by theta degrees about the parametric line (X,Y,Z) = (t, 0, f)
        Y = Y0 * np.cos(deg_to_rad(-slant))
        Z  = f + Y0 * np.sin(deg_to_rad(-slant))
        (x, y) = XYZ_to_xy(X,Y,Z)
        (x,y) = xy_to_xpyp(x, y, N_PIX, FOV_DEG)

        # Check if points fall within field of view
        if 0 <= x < N_PIX and 0 <= y < N_PIX:
            draw.point([(x,y)], fill=0)  # Use black color (0) for drawing

    img_array = np.array(img)
    return img_array

def compute_conditional_probabilities(N_PIX, N_POINTS, slant, FOV_DEG):
    f = 50
    N_bins = 30
    if abs(slant) >= 90 - FOV_DEG/2:
        raise ValueError("We require FOV_DEG/2 < 90 - slant")

    Y = f * np.tan(deg_to_rad(FOV_DEG/2)) / (1 - np.tan(deg_to_rad(FOV_DEG/2)) *( np.tan(deg_to_rad(abs(slant)))))
    delta_Z = Y * np.tan(deg_to_rad(abs(slant)))
    XMAX = (f + delta_Z) * np.tan(deg_to_rad(FOV_DEG/2))
    YMAX =  Y / np.cos(deg_to_rad( abs(slant)))

    N_POINTS = int(N_POINTS * XMAX * YMAX / (f**2 * np.tan(deg_to_rad(FOV_DEG/2))**2))

    histogram = np.zeros(N_bins)

    for _ in range(N_POINTS):
        X = XMAX * (2 * random.random() - 1)
        Y0 = YMAX * (2 * random.random() - 1)

        Y = Y0 * np.cos(deg_to_rad(-slant))
        Z = f + Y0 * np.sin(deg_to_rad(-slant))
        (x, y) = XYZ_to_xy(X, Y, Z)
        (x, y) = xy_to_xpyp(x, y, N_PIX, FOV_DEG)

        if 0 <= x < N_PIX and 0 <= y < N_PIX:
            bin_index = int(y / N_PIX * len(histogram))  # Convert y position to bin index
            histogram[bin_index] += 1

    # Normalize histogram to obtain conditional probabilities
    conditional_probabilities = histogram / np.sum(histogram)
    return conditional_probabilities
    


    
def compute_log_likelihood_slant(image, slant, probabilities, N_PIX):
    weighted_values = []
    for y in range(N_PIX):
        for x in range(N_PIX):
            if image[y, x] == 0:  # Check if pixel value is 0 (black, indicating a point)
                slant_key = slant
                bin_index = int(y / N_PIX * len(probabilities[slant_key]))  # Convert y position to bin index
                conditional_probability = probabilities[slant_key][bin_index]
                if conditional_probability > 0:
                    weighted_values.append(np.log(conditional_probability))
                else:
                    weighted_values.append(-np.inf)  # Log of zero goes to negative infinity
    return sum(weighted_values)

def main():
    N_PIX = 256
    N_POINTS = 2000
    FOV_DEG = 50
    slant_range = range(-60, 61, 10)
    slant_values_to_plot = [-60, 0, 60]

    # Create a 3x1 subplot grid
    fig, axes = plt.subplots(1, 3, figsize=(6, 18))

    # Generate and plot images for specified slants
    for i, slant in enumerate(slant_values_to_plot):
        image = draw_slanted_plane(N_PIX, N_POINTS, slant, FOV_DEG)
        flipped_image = np.flipud(image)  # Flip the image along the vertical axis
        ax = axes[i]
        ax.imshow(flipped_image, cmap='gray', interpolation='nearest')
        ax.set_title(f'Image at {slant} degree slant (flipped)')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

    plt.show()

    
    probabilities = []
    probabilities_file = {}
    images = {}
    for slant in slant_range:
        image = draw_slanted_plane(N_PIX, N_POINTS, slant, FOV_DEG)
        images[slant] = image  # Store the generated image
        prob = compute_conditional_probabilities(N_PIX, N_POINTS, slant, FOV_DEG)
        probabilities.append(prob)
        probabilities_file[slant] = prob
    
    probabilities = np.array(probabilities).T
    norm = colors.PowerNorm(gamma=0.3, vmin=probabilities.min(), vmax=probabilities.max())

    plt.imshow(probabilities, aspect='auto', cmap='plasma', norm=norm, extent=[min(slant_range), max(slant_range), 0, N_PIX])
    plt.colorbar(label='Probability')
    plt.xlabel('Slant (degrees)')
    plt.ylabel('Y Position')
    plt.title('Conditional Probabilities')
    plt.show()

    # PART B -------------------------------------- ----------------------

    # load up probabilities from the file: we could use pickle
    probabilities = probabilities_file
    N_bins = 30



    fig, axes = plt.subplots(5, 2, figsize=(20, 25), gridspec_kw={'hspace': 1, 'wspace': 0.5})
    slants = [-60, -30, 0, 30, 60]

    for i, slant in enumerate(slants):
        # Initialize the count for each y bin
        y_bin_counts = [0] * N_bins
        
        # Iterate over all y and x positions in the image
        for y in range(N_PIX):
            for x in range(N_PIX):
                if images[slant][y, x] == 0:  # Check if pixel value is 0 (black, indicating a point)
                    bin_index = int(y * N_bins / N_PIX)  # Convert y position to bin index
                    if bin_index >= N_bins:
                        bin_index = N_bins - 1  # Ensure the bin index is not out of range
                    y_bin_counts[bin_index] += 1

        # First column: Number of points per y bin
        axes[i, 0].bar(range(N_bins), y_bin_counts, align='edge', width=1, color='skyblue')
        axes[i, 0].set_title(f'Slant {slant} degrees: Points per Y Bin')
        if i == len(slants) - 1:  # Only label the bottom row
            axes[i, 0].set_xlabel('Y Bin Index')
        axes[i, 0].set_ylabel('Number of Points')

        log_likelihoods = []
        for s in slant_range:
            log_likelihoods.append(compute_log_likelihood_slant(images[slant], s, probabilities, N_PIX))
        axes[i, 1].plot(slant_range, log_likelihoods, color='green')
        axes[i, 1].set_title(f'Log Likelihood for Slant {slant} degrees')
        axes[i, 1].set_xticks(np.arange(-60, 61, 10))
        if i == len(slants) - 1:  # Only label the bottom row
            axes[i, 1].set_xlabel('Slant (degrees)')
        axes[i, 1].set_ylabel('Log Likelihood')

    plt.show()

if __name__ == "__main__":
    main()