import numpy as np
from scipy.signal import convolve2d
from make2DGabor import make2DGabor

def apply_gabor(image, M, kx, ky):

    # Generate Gabor filter
    cosGabor, sinGabor = make2DGabor(M, kx, ky)
    
    # Apply Gabor filter to image
    filtered_cos = convolve2d(image, cosGabor, mode='valid')
    filtered_sin = convolve2d(image, sinGabor, mode='valid')
    filtered_img_magnitude = np.sqrt(filtered_cos**2 + filtered_sin**2)
    
    return filtered_img_magnitude

def Q2(image, M= 32):

    # Apply Gabor filters for horizontal (0 rad) and vertical (pi/2 rad) orientations
    horizontal_response = apply_gabor(image, M, 4, 0) # Horizontal kx=4, ky=0
    vertical_response = apply_gabor(image, M, 0, 4)   # Vertical kx=0, ky=4
    
    # Sum responses in left-right and top-bottom halves
    lr_diff = max(np.abs(horizontal_response[:, :image.shape[1]//2].sum() + vertical_response[:, image.shape[1]//2:].sum()), np.abs(vertical_response[:, :image.shape[1]//2].sum() + horizontal_response[:, image.shape[1]//2:].sum()))
    tb_diff = max(np.abs(horizontal_response[:image.shape[0]//2, :].sum() + vertical_response[image.shape[0]//2:, :].sum()),np.abs(vertical_response[:image.shape[0]//2, :].sum() + horizontal_response[image.shape[0]//2:, :].sum()))
    

    if lr_diff > tb_diff:
        return 'lr'
    else:
        return 'tb'

