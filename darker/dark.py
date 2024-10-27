import cv2
import numpy as np

def darken_orange(image_path, output_path, darkening_factor=0.5):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the orange color in HSV
    lower_orange = np.array([10, 50, 50])
    upper_orange = np.array([30, 255, 255])

    # Create a mask for the orange color
    mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

    # Darken the orange color in the original image
    image[mask > 0] = image[mask > 0] * darkening_factor

    # Save the result
    cv2.imwrite(output_path, image)

# Replace 'your_image_path.jpg' and 'output_image_path.jpg' with your actual file paths
darken_orange('reward.png', 'dark_reward.png')
