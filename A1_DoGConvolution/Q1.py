from PIL import Image, ImageDraw, ImageOps
import numpy as np
# generate random point

# Create a blank image 
N =256
width, height = N,N # Define the size of the image
image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)

# Generate random postitions in field
X_width = 80
Z_depth = 50
h = 1.9
line_height = 0.5
# generate lines
num_lines = 5000
for i in range (num_lines): 

    X= (np.random.rand(1) - 0.5) *  X_width 
    Y = -h
    Z = (np.random.rand(1)) *  Z_depth

    Y_top = Y + line_height
    
    # convert into eye coordinates and flip
    x_eye = (X/Z)
    y_eye = (Y/Z)
    y_top = (Y_top/Z)
    print(x_eye)
    print(y_eye)

    # Invert y-axis for correct orientation

    # Calculate the screen coordinates (scaling and translating for visibility) (!! what is this?)
    screen_center_x, screen_center_y = width // 2, height // 2
    screen_x = screen_center_x + int(x_eye * width // 2)  # Adjust the scale factor as needed
    screen_y = screen_center_y - int(y_eye * height // 2) # the minus sign is to counterbalance the fact that the image drawing starts from the top
    screen_y_top = screen_center_y - int(y_top * height // 2) # the minus sign is to counterbalance the fact that the image drawing starts from the top

    if (screen_x < 0 or screen_x > width ):
        print(screen_x)
        print("X dont draw")
        num_lines += 1
        continue
    if (screen_y < 0 or screen_y > height): 
        print(screen_y)
        print("Y dont draw")
        num_lines += 1
        continue

    left = screen_x 
    right = screen_x 
    top = screen_y_top
    bottom = screen_y 

    draw.line((left, top, right, bottom), fill="black")


image.show()  # This will display the image using an image viewer
