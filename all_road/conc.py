import cv2
import os
import numpy as np
# Set the path to the folder containing your PNG images
image_folder = './Wall'

# Get a list of all PNG files in the folder
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
# Sort the images to ensure they're in the correct order
images.sort(key=lambda x: int(x.split(".")[0]))

img_array = []
for filename in images:
    img_path = os.path.join(image_folder, filename)
    img = cv2.imread(img_path)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
# out = cv2.VideoWriter('Wall.avi',cv2.VideoWriter_fourcc(*'DIVX'), 3, size)
out = cv2.VideoWriter('Wall.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 6, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()