import cv2
import numpy as np
import os

def combine_red_lines(image_folder, output_path):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

    if not image_files:
        print("No image files found in the specified folder.")
        return

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)

    # Create an empty image to combine the red lines
    combined_image = first_image
    red_lines_mask = cv2.inRange(first_image, (0, 0, 200), (50, 50, 255))
    # Loop through each image and combine the red lines with the original background
    for image_file in image_files[:-90]:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        cv2.imwrite('t.png', image)
        red_lines_mask = cv2.inRange(image, (0, 0, 200), (50, 50, 255))
        red_lines = cv2.bitwise_and(image, image, mask=red_lines_mask)
        ret, red_lines_mask = cv2.threshold(red_lines_mask, 1, 255, cv2.THRESH_BINARY)
        red_lines[red_lines_mask>0] = [0, 0, 255]
        combined_image = cv2.add(combined_image, red_lines)

    # Save the combined image
    lower_pink = np.array([140, 100, 100])
    upper_pink = np.array([180, 255, 255])
    image_hsv = cv2.cvtColor(combined_image, cv2.COLOR_RGB2HSV)
    pink_mask = cv2.inRange(image_hsv, lower_pink, upper_pink)
    combined_image[pink_mask > 0] = [0, 0, 255]
    cv2.imwrite(output_path, combined_image)
    print(f"Combined red lines with background saved to {output_path}")


image_folder_path = "./wallstreet/"
output_image_path = "wallstreet.jpg"

combine_red_lines(image_folder_path, output_image_path)
