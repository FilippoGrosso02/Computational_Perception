import numpy as np
import matplotlib.pyplot as plt

def make2DGabor(M, kx, ky):
    '''
    This function returns a 2D cosine Gabor and a 2D sine Gabor with 
    center frequency (k0,k1) cycles per M samples.   

    The sigma of the Gaussian is chosen automatically to be one half cycle
    of the underlying sinusoidal wave.   
     
    Example:  make2DGabor( M, cos(theta)*k, sin(theta)*k  ) returns a tuple of
    array containing the cosine and sine Gabors.
  
    e.g.    (cosGabor, sinGabor) = make2DGabor(32,4,0)
  
    Note that a more general definition of a Gabor would pass the sigma as a
    parameter, rather than defining it in terms of k and N.   I've used a 
    more specific definition here because it is all we need for Assignment 2.
    '''
    
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

def createComplexCell(M, kx, ky):
    cosGabor, sinGabor = make2DGabor(M, kx, ky)
    
    complexCellMagnitude = np.sqrt(cosGabor**2 + sinGabor**2)
    return complexCellMagnitude

def makeGaussianRidge(M, orientation, ridge_width=4):
    # Convert orientation to radians
    theta = np.deg2rad(orientation)
    
    # Define the standard deviation of the Gaussian
    sigma = ridge_width / np.sqrt(8 * np.log(2))  

    # Create a coordinate grid
    x = np.arange(-M//2, M//2)
    y = np.arange(-M//2, M//2)
    X, Y = np.meshgrid(x, y)

    
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)

    
    gaussian_ridge = np.exp(-Y_rot**2 / (2 * sigma**2))

    # Normalize the ridge to have a maximum of 1
    gaussian_ridge /= gaussian_ridge.max()

    return gaussian_ridge

def create_oriented_edge(M, orientation, steepness=5):

    theta = np.deg2rad(orientation)

    x = np.arange(-M//2, M//2)
    y = np.arange(-M//2, M//2)
    X, Y = np.meshgrid(x, y)

    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)

    edge = 1 / (1 + np.exp(-steepness * Y_rot))

    return edge

def simulateComplexCellResponse(image, gabor_filter):
    
    image = image.flatten()

    gabor1 = gabor_filter[0].flatten()
    gabor2 = gabor_filter[1].flatten()
    response_image= np.sqrt(np.dot(image ,  gabor1)**2 + np.dot(image , gabor2)**2)

    return float(response_image)

# RIDGES
plt.figure(figsize=(10, 8))  
plt.suptitle("Gaussian Ridges at Various Orientations") 
orientations = range(0, 180, 15)

for i, orientation in enumerate(orientations):
    plt.subplot(3, 4, i+1)
    img = makeGaussianRidge(32, orientation)
    plt.imshow(img, cmap='gray', extent=[-16, 15, -16, 15], origin='lower')
    plt.title(f"Ridge {orientation}°")
    plt.axis('off')

# Show the response to ridges
x, y = [], []
for orientation in orientations:
    response = simulateComplexCellResponse(makeGaussianRidge(32, orientation), make2DGabor(32, 4, 0))

    x.append(orientation)
    y.append(response)

plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.title("Response to Gaussian Ridges")  
plt.xlabel("Orientation (degrees)")
plt.ylabel("Response")

# EDGES
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
plt.suptitle("Oriented Edges at Various Orientations")  # Overall title for the edges

for i, orientation in enumerate(orientations):
    plt.subplot(3, 4, i+1)
    img = create_oriented_edge(32, orientation)
    plt.imshow(img, cmap='gray', extent=[-16, 15, -16, 15], origin='lower')
    plt.title(f"Edge {orientation}°")
    plt.axis('off')

# Show the response to edges
x, y = [], []
for orientation in orientations:
    response = simulateComplexCellResponse(create_oriented_edge(32, orientation), make2DGabor(32, 4, 0))
    print(response)
    x.append(orientation)
    y.append(response)

plt.figure(figsize=(10, 5))
plt.plot(x, y, color = "red")
plt.title("Response to Oriented Edges")  # Title for the edge response plot
plt.xlabel("Orientation (degrees)")
plt.ylabel("Response")

plt.show()

