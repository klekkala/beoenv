import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pickle

tmp_list = ['5/50']
pathname = sys.argv[1]
dir_list = [name for name in os.listdir(pathname) if os.path.isdir(os.path.join(pathname, name))]
print(dir_list)

assert(len(tmp_list)==1)
for game in dir_list:
    print('--------------------'+game+'---------------------')
    if game=='expert_1chan_spacedemo':
        continue
    game_path=os.path.join(pathname,game)
    all_val = []
    all_epi = []
    all_rew = []
    all_ter = []
    all_limit = []
    all_id = []
    for directory in tmp_list:
        sorted_value_mapping_truncated = {}
        file_path = os.path.join(game_path)
        print(file_path, directory)
        file_path = os.path.join(file_path,directory)
        value_path = os.path.join(file_path,'value_truncated.npy')
        #indices_path = os.path.join(file_path,'reversed_indices_truncated.npy')
        # sorted_value_path = os.path.join(file_path,'sorted_value_truncated.npy')
        
        value = np.load(value_path,allow_pickle=True)
        #indices = np.load(indices_path,allow_pickle=True)
        # sorted_value = np.load(sorted_value_path,allow_pickle=True)

        for i in range(value.shape[0]):
            temp = sorted_value_mapping_truncated.get(value[i],[])
            if temp == []:
                sorted_value_mapping_truncated[value[i]] = [i]
            else:
                sorted_value_mapping_truncated[value[i]].append(i)


        print(sum([len(i) for i in sorted_value_mapping_truncated.values()]))

        with open(os.path.join(file_path,f'sorted_value_mapping_truncated'), "wb") as f:
            pickle.dump(dict(sorted(sorted_value_mapping_truncated.items())), f)


