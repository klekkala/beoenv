import numpy as np
import shutil
import cv2
import os
import gc
base_path = '/lab/tmpig10f/kiran/expert_3chan_beogym/skill2/'
target_path = '/home3/tmp/kiran/4stack_beogym/'
envs=['expert_3chan_allegheny','expert_3chan_CMU','expert_3chan_hudsonriver', 'expert_3chan_southshore', 'expert_3chan_unionsquare', 'expert_3chan_wallstreet']
# envs=['expert_3chan_CMU','expert_3chan_hudsonriver', 'expert_3chan_southshore', 'expert_3chan_unionsquare', 'expert_3chan_wallstreet']

for env in envs:
    res=[]
    obs =  np.load(base_path + env +'/5/50/observation.npy')
    for i in range(len(obs)):
        res.append(np.transpose(np.concatenate([obs[i+t] if i+t>=0 else obs[0] for t in range(-3,1)], axis=2),(2,0,1)))

    res=np.array(res)
    os.makedirs(target_path + env +'/5/50/', exist_ok=True)
    np.save(target_path + env +'/5/50/4stack_observation.npy', res)
    shutil.copy(base_path + env +'/5/50/action.npy', target_path + env +'/5/50/action.npy')
    shutil.copy(base_path + env +'/5/50/reward.npy', target_path + env +'/5/50/reward.npy')
    shutil.copy(base_path + env +'/5/50/terminal.npy', target_path + env +'/5/50/terminal.npy')
    print(f'{env} done')
    gc.collect()




