import numpy as np
from IPython import embed
import os
import sys
import matplotlib.pyplot as plt
#tmp_list = ['1', '10', '50']
#pathname = '../all_1chan_all/'
tmp_list = ['5/50']
#pathname = '../random_bev_carla/'
pathname = sys.argv[1]
dir_list = [name for name in os.listdir(pathname) if os.path.isdir(os.path.join(pathname, name))]
# dir_list = [name for name in os.listdir(pathname)]
print(dir_list)
# fig, axes = plt.subplots(1, len(dir_list), figsize=(15, 5))
# axidx = 0

for game in dir_list:
    print('--------------------'+game+'---------------------')
    game_path=os.path.join(pathname,game)
    all_val = []
    all_act = []
    all_epi = []
    all_limit = []
    all_id = []
    for directory in tmp_list:
        file_path = os.path.join(game_path)
        print(file_path, directory)
        file_path = os.path.join(file_path,directory)
        reward_path = os.path.join(file_path,'reward.npy')
        action_path = os.path.join(file_path,'action.npy')
        terminal_path = os.path.join(file_path,'terminal.npy')

        rew = np.load(reward_path,allow_pickle=True)
        negative_indices = np.where(rew < 0)[0]
        if len(negative_indices) > 0:
            rew[rew < 0] = 0
            print(f"{game} has negative rewrd, fixed")
        ter = np.load(terminal_path,allow_pickle=True)


        ter[-1]=1
        indices = np.where(ter == 1)

        slices_r = []
        slices_v = []
        slice_epi = []
        slice_limit = [] 
        # Iterate through the indices and add slices to the lists
        start_idx = 0
        count = 0
        prev_idx = -1
        id_dict = {}
        for idx in indices[0]:
            slices_r.append(rew[start_idx:idx+1])
            slice_epi += [count]*(idx - (prev_idx+1) + 1)
            slice_limit += [idx]*(idx - (prev_idx+1) + 1)
            id_dict[count] = start_idx
            #print(prev_idx, idx, len(slice_limit))
            assert(len(slice_epi) == len(slice_limit) == idx+1)
            assert(ter[len(slice_epi)-1] == 1)
            assert(ter[slice_limit[-1]] == 1)
            prev_idx = idx

            start_idx = idx+1
            count += 1

        print(len(slice_epi))
        slice_epi += [count]*(rew.shape[0] - len(slice_epi))
        slice_limit += [(rew.shape[0]-1)]*(rew.shape[0] - len(slice_limit))
        assert(ter[len(slice_epi)-1] == 1)
        for abcd in range(rew.shape[0]):
            assert(ter[slice_limit[abcd]] == 1)
        assert(ter[slice_limit[-1]] == 1)
        np_epi = np.stack(slice_epi)
        np_limit = np.stack(slice_limit)
        for arr in slices_r:
           a=0.95
           powers = np.arange(arr.size)
           output = [np.sum(arr[i:] * a ** powers[: arr.size - i]) for i in range(arr.size)]
           slices_v+=output
           print(len(slices_v))
        print(len(slices_v))
        np.save(os.path.join(file_path,'value.npy'),slices_v)

        

