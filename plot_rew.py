from beogym.beogym import BeoGym
import json
import math
import matplotlib.pyplot as plt
import cv2
import pickle
import numpy as np

fileName = '/home6/tmp/kiran/expert_3chan_beogym/skill_check/expert_3chan_unionsquare/5/50/'
env = BeoGym({'city':'Union_Square','data_path':'/home6/tmp/kiran/'})


source = (-16.368810357379488, -4.769885278439986)
goal = (-56.353110976479435, -52.67864397355789)



all_node = []
for q in env.dh.Gdict.keys():
    all_node.append(q)
x,y = zip(*all_node)

dis_to_goal  =  math.sqrt((source[0] - goal[0])**2 + (source[1] - goal[1])**2)

threshold = 15
marks = []
for i in range(threshold):
    # self.marks.append(8*((self.agent.dis_to_goal/8)**(1/threshold))**i)
    marks.append(dis_to_goal/threshold*(i+1))
marks=marks[:-1]

# plt.scatter(y_r, x_r, color='red', s=1)
fig, ax = plt.subplots()
for i in marks:
    circle = plt.Circle(goal, i, edgecolor='orange', facecolor='none')
    ax.add_patch(circle)
plt.scatter([source[0], goal[0]], [source[1], goal[1]], color='green', s=40)
plt.scatter(x, y, color='blue', s=1)

ax.set_xlim(-59, 18)
ax.set_ylim(-81, 26)
# plt.scatter([source[0]],[source[1]], color = 'yellow', s=20)
# plt.scatter([goal[0]],[goal[1]], color = 'green', s=20)
plt.legend()
plt.savefig(f'check_uaca.png')
plt.clf()