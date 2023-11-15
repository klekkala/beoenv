from beogym.beogym import BeoGym
import json
import math
import matplotlib.pyplot as plt
import cv2
import pickle
import numpy as np

fileName = '/home6/tmp/kiran/expert_3chan_beogym/skill_check/expert_3chan_unionsquare/5/50/'
env = BeoGym({'city':'Union_Square','data_path':'/home6/tmp/kiran/'})

aux = np.load(fileName+'aux.npy')
rew = np.load(fileName+'reward.npy')
all = np.load(fileName+'allp.npy')
print(all.shape)
source = (-16.368810357379488, -4.769885278439986)
goal = (-56.353110976479435, -52.67864397355789)


all=[]
for i in range(aux.shape[0]):
    if rew[i]>0:
        # x_r.append()
        all.append((aux[i][0]*400-200+source[1],aux[i][1]*400-200+source[0]))
print(len(all))




y_r,x_r = zip(*all)
plt.scatter(y_r, x_r, color='red', s=1)
# plt.scatter([source[0]],[source[1]], color = 'yellow', s=20)
# plt.scatter([goal[0]],[goal[1]], color = 'green', s=20)
plt.legend()
plt.savefig(f'check_ua.png')
plt.clf()