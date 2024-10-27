import numpy as np
from IPython import embed
import os
import sys
import matplotlib.pyplot as plt
import random
import cv2
tmp_list = ['50']
pathname = sys.argv[1]
dir_list = [name for name in os.listdir(pathname) if os.path.isdir(os.path.join(pathname, name)) and name!='frame8']
print(dir_list)

res={}
for game in dir_list:
    print('--------------------'+game+'---------------------')
    game_path=os.path.join(pathname,game)
    game_path=os.path.join(game_path,'5')
    all_val = []
    all_act = []
    all_epi = []
    all_rew = []
    all_ter = []
    all_limit = []
    all_id = []
    res[game] = []
    for directory in tmp_list:


        file_path = os.path.join(game_path)
        print(file_path, directory)
        file_path = os.path.join(file_path,directory)
        reward_path = os.path.join(file_path,'reward')
        action_path = os.path.join(file_path,'action')
        terminal_path = os.path.join(file_path,'terminal')
        value_path = os.path.join(file_path,'value_truncated.npy')
        obs_path = os.path.join(file_path,'observation')
        rew = np.load(reward_path,allow_pickle=True)
        negative_indices = np.where(rew < 0)[0]
        if len(negative_indices) > 0:
            rew[rew < 0] = 0
            print(f"{game} has negative rewrd, fixed")
        act = np.load(action_path,allow_pickle=True)
        ter = np.load(terminal_path,allow_pickle=True)
        val = np.load(value_path,allow_pickle=True)
        obs = np.load(obs_path,allow_pickle=True)

        ter[-1]=1
        indices = np.where(ter == 1)


        slices_a = []
        slices_r = []
        slices_v = []
        slice_epi = []
        slice_limit = [] 
        slices_ter = []
        slice_v_trun = []
        # Iterate through the indices and add slices to the lists
        start_idx = 0
        count = 0
        prev_idx = -1
        idxs=[]
        rew_indices = np.where(rew == 1)
        for i in rew_indices[0]:
            flag=1
            if ter[i]==1:
                continue
            for j in range(1,8):
                if rew[i-j]!=0 or ter[i-j]==1:
                    flag = 0
                    break
            if flag==1:
                idxs.append(i)
        os.mkdir(f'./frame8/{game}')
        for i in random.sample(idxs,2):
            for j in range(8):
                print(f'{game} {i-j} reward {rew[i-j]} value {val[i-j]}')
                cv2.imwrite(f'./frame8/{game}/{i-j}.jpg',obs[i-j])



        


        

