# import pickle

# with open('Wall_Street.pkl', 'rb') as f:
#     paths = pickle.load(f)

from beogym.beogym import BeoGym
import json
import math
import matplotlib.pyplot as plt
import cv2
import pickle

env = BeoGym({'city':'Union_Square','data_path':'/home6/tmp/kiran/'})
# env.agent.reset((-56.42094821954678, -46.487781984396584))
# env.courier_goal = (-96.8055251689076, -94.02399239301302)
# paths=[]
# for i in range(1000):
#     for q in range(4):
        


env.shortest_rec()
