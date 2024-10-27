from beogym.beogym import BeoGym
import json
import math
import matplotlib.pyplot as plt
import cv2
import pickle,random

NYC = ['Wall_Street','Union_Square', 'Hudson_River']
Pits= ['CMU', 'Allegheny', 'South_Shore']
cities = NYC+Pits


for idx,city in enumerate(cities):
    env = BeoGym({'city':city,'data_path':'/home6/tmp/kiran/'})
    all_node = []
    for q in env.dh.Gdict.keys():
        all_node.append(q)
    x,y = zip(*all_node)
    plt.scatter(x, y, color='blue', s=1)
    plt.savefig(f'./bvi/{city}.jpg')
    plt.clf()

    del env
        
# with open('Wall_Street.pkl', 'wb') as f:
#     pickle.dump(res, f)