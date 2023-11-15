#from beogym.beogym import BeoGym
import numpy as np
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.algorithm import Algorithm
import cv2
import os
import random
import torch
#import configs
from models.beogymmodels import ComplexNet
from ray.rllib.models import ModelCatalog
from IPython import embed

ModelCatalog.register_custom_model("ComplexNet", ComplexNet)

def seed_everything(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    env.seed(seed)
    return env


##union_square
#my_restored_policy = Policy.from_checkpoint("/lab/kiran/logs/rllib/beogym/notemp/1.a_Union_Square_singlegame_full_3CHAN_TCN_BEOGYM_EXPERT_3CHAN_UNIONWALL_STANDARD_1.0_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_11_04_01_16_29/checkpoint/")

#my_restored_policy = Policy.from_checkpoint("/lab/kiran/logs/rllib/beogym/notemp/1.a_Union_Square_singlegame_full_3CHAN_VEP_BEOGYM_EXPERT_3CHAN_UNIONWALL_STANDARD_1.0_0.1_1.0_same_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_11_05_14_52_20/checkpoint/")


#my_restored_policy = Policy.from_checkpoint("/lab/kiran/logs/rllib/beogym/notemp/1.a_Union_Square_singlegame_full_3CHAN_NVEP_BEOGYM_EXPERT_3CHAN_UNIONWALL_STANDARD_2.0_-1.0_2.0_same_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_11_12_18_10_06/checkpoint/")


##wall_street
#my_restored_policy = Policy.from_checkpoint("/lab/kiran/logs/rllib/beogym/notemp/1.a_Wall_Street_singlegame_full_3CHAN_TCN_BEOGYM_EXPERT_3CHAN_UNIONWALL_STANDARD_1.0_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_11_04_01_14_18/checkpoint/")

my_restored_policy = Policy.from_checkpoint("/lab/kiran/logs/rllib/beogym/notemp/1.a_Wall_Street_singlegame_full_3CHAN_NVEP_BEOGYM_EXPERT_3CHAN_UNIONWALL_STANDARD_1.0_0.1_1.0_same_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_11_09_01_41_08/checkpoint/")



#my_restored_policy = Policy.from_checkpoint("/lab/kiran/logs/rllib/beogym/notemp/1.a_Wall_Street_singlegame_full_3CHAN_NVEP_BEOGYM_EXPERT_3CHAN_UNIONWALL_STANDARD_1.0_-1.0_1.0_same_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_11_13_10_58_05/checkpoint/")


from envs import SingleBeoEnv
env = SingleBeoEnv({'env': 'Union_Square', 'data_path': '/home6/tmp/kiran/'})
#env = SingleBeoEnv({'env': 'Wall_Street', 'data_path': '/home6/tmp/kiran/'})


res = []

for _ in range(1):
    random_generated_int = random.randint(0, 2**31 - 1)
    env = seed_everything(random_generated_int, env)
    total = 0
    obs = env.reset()
    env.seed(random_generated_int)
    for _ in range(700):
        
        obs['obs'] = obs['obs'].astype(np.float32)

        action = my_restored_policy.compute_single_action(obs)[0]
        obs, reward, done, _ = env.step(action)
        # obs = cv2.resize(obs, (84, 84))
        total+=reward
        if done:
            break
    res.append(total)
average = sum(res) / len(res)
print(res)
print(average)
# with open('res.txt','w') as f:
#     f.write(str(res))
