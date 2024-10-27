# import numpy as np

# import os
# import random

# base_path = '/lab/tmpig10f/kiran/expert_3chan_beogym/skill2/'

# games = ['expert_3chan_allegheny', 'expert_3chan_hudsonriver', 'expert_3chan_unionsquare', 'expert_3chan_CMU', 'expert_3chan_southshore', 'expert_3chan_wallstreet']

# from PIL import Image



# for game in games:


#     game_path = os.path.join(base_path, game+'/5/50/')
#     obss = np.load(game_path+'observation.npy', mmap_mode='r')
#     ter = np.load(game_path+'terminal_truncated.npy')

#     ter_indices = np.where(ter == 1)[0]
#     rdm = random.sample(range(0,len(ter_indices)-1), 1)
#     all_embedding=[]
#     values=[]
#     for i in rdm:
#         obs = np.asarray(obss[ter_indices[i]+1 : ter_indices[i+1]])
#     for idx,i in enumerate(obs):
#         image = Image.fromarray(i)
#         image.save(f"./continue/{game.split('_')[-1]}_{idx}.png")


from beogym.beogym import BeoGym
import json
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle,random

NYC = ['Wall_Street','Union_Square', 'Hudson_River']
Pits= ['CMU', 'Allegheny', 'South_Shore']
cities = NYC+Pits



for idx,city in enumerate(cities):
    env = BeoGym({'city':city,'data_path':'/home6/tmp/kiran/'})
    res=[]
    while True:
        env.reset()
        pos = env.dh.sample_location()
        view = env.dh.panorama_split(random.randint(0,359),  pos, 0, True)
        flag=0
        for i in range(5):
            _,_,_,_ = env.step(0)
            obs = env.agent.curr_view
            if np.all(obs == view):
                res=[]
                flag=1
                break
            else:
                res.append(obs)
                view = obs
        if flag==0:
            break
    for idx, view in enumerate(res):
        cv2.imwrite(f'./continue/{city}_{idx}.jpg', view)
        print('ss')
    del env
        
# with open('Wall_Street.pkl', 'wb') as f:
#     pickle.dump(res, f)