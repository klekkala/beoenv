import numpy as np
from IPython import embed
import os
import sys
import matplotlib.pyplot as plt

tmp_list = ['5/50']
pathname = sys.argv[1]
dir_list = [name for name in os.listdir(pathname) if os.path.isdir(os.path.join(pathname, name))]
print(dir_list)

threshold = 10

for game in dir_list:
    print('--------------------'+game+'---------------------')
    game_path=os.path.join(pathname,game)
    for directory in tmp_list:

        file_path = os.path.join(game_path)
        print(file_path, directory)
        file_path = os.path.join(file_path,directory)
        try:
            reward_path = os.path.join(file_path,'reward.npy')
            terminal_path = os.path.join(file_path,'terminal.npy')
            rew = np.load(reward_path,allow_pickle=True)
            negative_indices = np.where(rew < 0)[0]
            if len(negative_indices) > 0:
                rew[rew < 0] = 0
                print(f"{game} has negative rewrd, fixed")
            ter = np.load(terminal_path,allow_pickle=True)
        except:
            continue

        ter[-1]=1
        indices = np.where(ter == 1)
        slice_trun_t = []
        start_idx = 0
        for idx in indices[0]:
            tmp_r = rew[start_idx:idx+1]
            rew_1 = np.where(tmp_r == 1)
            for r in rew_1[0]:
                if r == tmp_r.shape[0]:
                    break
                flag = 1
                for i in range(1, threshold+1):
                    if r+i==tmp_r.shape[0]:
                        flag = 0
                        break
                    if tmp_r[r+i] == 1:
                        flag = 0
                tmp_r[r] = flag
            tmp_r[-1] = 1
            slice_trun_t = np.concatenate((slice_trun_t,tmp_r))
            start_idx = idx + 1

        assert(len(slice_trun_t) == len(ter))
        print(pathname, game)
        np.save(pathname + game + '/5/50/terminal_truncated.npy', slice_trun_t)
        # rew = np.load(reward_path,allow_pickle=True)
        # for i in np.where(slice_trun_t!=rew)[0]:
        #     print(slice_trun_t[i])
        #     print(slice_trun_t[i+1])
        #     print(slice_trun_t[i+2])
        #     print(slice_trun_t[i+3])
        #     print(rew[i])
        #     print(rew[i+1])
        #     print(rew[i+2])
        #     print(rew[i+3])
        #     print(ter[i])
        #     print(ter[i+1])
        #     print(ter[i+2])
        #     print(ter[i+3])
        #     print('aa')
        # np.save(os.path.join(file_path,'value_truncated'),slice_v_trun)


        



        

