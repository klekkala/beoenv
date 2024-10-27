import numpy as np
from PIL import Image
path = '/lab/tmpig13d/result_1m/slide_block/object_texture/50'

obs = np.load(path+'/observation.npy', mmap_mode = 'r', allow_pickle=True)

v= np.load(path+'/value.npy', mmap_mode = 'r', allow_pickle=True)
print(np.where(v>=1)[0])
for idx,i in enumerate(np.where(v>=1)[0]):
    if idx>50:
        break
    img = Image.fromarray(obs[i])
    img.save(f'./check/{idx}.jpg')
    

