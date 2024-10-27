import numpy as np
from IPython import embed
import os
import sys
import matplotlib.pyplot as plt

tmp_list = ['50']
pathname = sys.argv[1]
dir_list = [name for name in os.listdir(pathname) if os.path.isdir(os.path.join(pathname, name))]
print(dir_list)

for game in dir_list:
    print('--------------------'+game+'---------------------')
    game_path=os.path.join(pathname,game)
    game_path=os.path.join(game_path,'5')
    for directory in tmp_list:


        file_path = os.path.join(game_path)
        print(file_path, directory)
        file_path = os.path.join(file_path,directory)
        reward_path = os.path.join(file_path,'reward.npy')
        terminal_path = os.path.join(file_path,'terminal_truncated.npy')

        rew = np.load(reward_path,allow_pickle=True)
        negative_indices = np.where(rew < 0)[0]
        if len(negative_indices) > 0:
            rew[rew < 0] = 0
            print(f"{game} has negative rewrd, fixed")
        ter = np.load(terminal_path,allow_pickle=True)

        np.delete
        del_have=[]
        ter[-1]=1
        indices = np.where(ter == 1)[0]
        assert rew[indices[0]] != 0
        for idx in range(1, len(indices)):
            if rew[indices[idx]] == 0 :
                del_have+=[num for num in range(indices[idx-1]+1, indices[idx]+1)]

        files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        for i in files:
            tmp_np = np.load(os.path.join(file_path, i),allow_pickle=True)
            if isinstance(tmp_np, np.ndarray):
                if len(tmp_np.shape) >0 and tmp_np.shape[0] == 1000000:
                    tmp_np = np.delete(tmp_np, del_have,axis=0)
                    np.save(os.path.join(file_path, i), tmp_np)
                else:
                    print(tmp_np.shape)
                    print(i)
            else:
                print(i)
            
            


        



        

