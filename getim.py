from beogym.beogym import BeoGym
import json
import math
import matplotlib.pyplot as plt
import cv2
import pickle,random

NYC = ['Wall_Street','Union_Square', 'Hudson_River']
Pits= ['CMU', 'Allegheny', 'South_Shore']
cities = NYC+Pits

# for idx,city in enumerate(cities):
#     env = BeoGym({'city':city,'data_path':'/home6/tmp/kiran/'})
#     paths = env.dh.get_all_paths((-56.42094821954678, -46.487781984396584), (-96.8055251689076, -94.02399239301302), 500)
#     res=[]
#     for path in paths:
#         temp = [env.dh.Greversed[i] for i in path]
#         res.append(temp)
        
# with open('Wall_Street.pkl', 'wb') as f:
#     pickle.dump(res, f)
            
        # r = [[-74.43864352079171, -64.64343430139013], [-74.51293811671118, -64.79605375468066], [-74.58138978051558, -64.97606422511555], [-74.66274416945646, -65.31252743306574], [-74.73638867552069, -65.62223248145776], [-74.83542279137231, -65.70225596901801], [-74.93537009000721, -65.6986687844645], [-74.98293153718014, -65.63503421028673], [-75.06005632174097, -65.62305541022913]]
        # for id,p in enumerate(r):
        #     p = tuple(p)
        #     if id +1==len(r):
        #         env.agent.update_agent(p,p,env.agent.curr_angle)
        #     else:
        #         env.agent.update_agent(p,p,env.dh.get_angle(r[id], r[id + 1]))

        #     height, width, _ = env.agent.curr_view.shape
        #     size = min(height, width)
        #     x = (width - size) // 2
        #     y = (height - size) // 2
        #     p_view = env.agent.curr_view[y:y+size,x:x+size]
        #     # p_view = cv2.resize(p_view, (84, 84))
        #     cv2.imwrite(f'./4imgs/{id}.jpg', p_view)


for idx,city in enumerate(cities):
    env = BeoGym({'city':city,'data_path':'/home6/tmp/kiran/'})
    for i in range(4):
        pos = env.dh.sample_location()
        print(pos)
        view = env.dh.panorama_split(random.randint(0,359),  pos, 0, True)
        cv2.imwrite(f'./4imgs/{city}_{i}.jpg', view)
    del env
        
# with open('Wall_Street.pkl', 'wb') as f:
#     pickle.dump(res, f)