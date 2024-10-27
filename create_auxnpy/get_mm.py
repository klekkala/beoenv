import numpy as np
from IPython import embed
import os
import sys
import matplotlib.pyplot as plt

tmp_list = ['50']
pathname = sys.argv[1]
dir_list = [name for name in os.listdir(pathname) if os.path.isdir(os.path.join(pathname, name))]
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
        reward_path = os.path.join(file_path,'reward.npy')
        action_path = os.path.join(file_path,'action.npy')
        terminal_path = os.path.join(file_path,'terminal.npy')
        value_path = os.path.join(file_path,'value.npy')

        rew = np.load(reward_path,allow_pickle=True)
        negative_indices = np.where(rew < 0)[0]
        if len(negative_indices) > 0:
            rew[rew < 0] = 0
            print(f"{game} has negative rewrd, fixed")
        act = np.load(action_path,allow_pickle=True)
        ter = np.load(terminal_path,allow_pickle=True)
        val = np.load(value_path,allow_pickle=True)


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
        for idx in indices[0]:
            tmp_r = rew[start_idx:idx+1]
            rew_indices = list(np.where(tmp_r == 1))
            rew_indices = [i for i in rew_indices if i-5>0=0]

            res[game] += rew_indices

            start_idx = idx+1
            count += 1
print(res)


        


        

