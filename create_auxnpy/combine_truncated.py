import numpy as np
import os
import sys
import matplotlib.pyplot as plt

tmp_list = ['50']

pathname = sys.argv[1]
dir_list = [name for name in os.listdir(pathname) if os.path.isdir(os.path.join(pathname, name))]
print(dir_list)

all_game_value = {}
assert(len(tmp_list)==1)
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
    for directory in tmp_list:
        file_path = os.path.join(game_path)
        print(file_path, directory)
        file_path = os.path.join(file_path,directory)
        value_path = os.path.join(file_path,'value_truncated.npy')


        value = np.load(value_path,allow_pickle=True)

        all_game_value[game] = value

        # sorted_indices = np.argsort(val)
        # reversed_indices = np.zeros_like(sorted_indices)
        # for i in range(len(sorted_indices)):
        #     reversed_indices[sorted_indices[i]] = i
        # sorted_val = val[sorted_indices]
        # res_val = np.stack((sorted_val, sorted_indices))
        # np.save(os.path.join(file_path,'sorted_value_truncated'),res_val)
        # np.save(os.path.join(file_path,'reversed_indices_truncated'),reversed_indices)
        # print(reversed_indices.shape)
all_value = np.array([])
game_idx={}
reversed={}
idx = 0
for key,value in all_game_value.items():
    all_value = np.concatenate((all_value,value))
    game_idx[key] = idx
    reversed[idx] = np.zeros_like(value)
    idx+=1
sorted_indices = np.argsort(all_value)
all_game_idx = sorted_indices//1000000
all_idx = sorted_indices%1000000
sorted_value = np.stack((all_value, all_game_idx, all_idx))
for i in range(sorted_value.shape[1]):
    reversed[sorted_value[1,i]][int(sorted_value[2,i])]=i

# com_name = ''
# for i in game_idx.keys():
#     com_name+=i.split('_')[-1][:4]
for key,value in game_idx.items():
    game_path=os.path.join(pathname,key)
    game_path=os.path.join(game_path,'5')
    file_path = os.path.join(game_path,tmp_list[0])
    game_name = key.split('_')[-1]
    np.save(os.path.join(file_path,f'sorted_value_truncated'),sorted_value)
    np.save(os.path.join(file_path,f'reversed_indices_truncated'),reversed[value])
