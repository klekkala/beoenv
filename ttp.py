import pickle
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
with open('labels', 'rb') as f:
    label = pickle.load(f)
masks = torch.load('masks.pt', map_location=torch.device('cpu'))
pixel_dict = {}
for idx, mask in enumerate(masks):
    if len(label[idx])<=6:
        continue
    non_zero_indices = torch.nonzero(mask[0]).numpy()
    for i in range(len(non_zero_indices)):
        coordinate = (non_zero_indices[i][1], non_zero_indices[i][0])
        if coordinate in pixel_dict.keys():
            pixel_dict[coordinate].append(label[idx])
        else:
            pixel_dict[coordinate] = [label[idx]]

for key, value in pixel_dict.items():
    obj = sorted(value, key=lambda x: float(x[-5:-1]))[-1]
    pixel_dict[key] = obj[:-6]

max_x = max(x for x, _ in pixel_dict.keys()) + 1
max_y = max(y for _, y in pixel_dict.keys()) + 1



unique_values = set(pixel_dict.values())
colors = plt.cm.get_cmap('tab20', len(unique_values))

color_map = {value: colors(i) for i, value in enumerate(unique_values)}

# Create an array to hold the color data
# image_array = np.full((max_y, max_x), [], dtype=object)
image_array = np.zeros([720, 1280, 3])
for (x, y), key in pixel_dict.items():
    color = color_map.get(key, (1, 1, 1, 1))  # Default to white (RGBA)
    image_array[y, x] = color[:-1]
print(image_array.shape)
# image_array = np.where(image_array == '', (0,0,0), image_array)
image_array = (image_array * 255).astype(np.uint8)
image = Image.fromarray(image_array)

# Save the image
image.save('output_image.png')
# # Convert the image_array to a format suitable for imshow
# image_array_rgb = np.array([[color_map.get(pixel, (1, 1, 1, 1)) for pixel in row] for row in image_array])


# plt.imshow(image_array_rgb, interpolation='none')
# plt.axis('off')  # Turn off axis
# plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0)