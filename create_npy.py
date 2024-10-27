from beogym.beogym import BeoGym
import json
import math
import matplotlib.pyplot as plt

NYC = ['Wall_Street','Union_Square', 'Hudson_River']
Pits= ['CMU', 'Allegheny', 'South_Shore']
cities = NYC+Pits


env = BeoGym({'city':'Wall_Street','data_path':'/home6/tmp/kiran/'})
print('load')
env.shortest_rec()
