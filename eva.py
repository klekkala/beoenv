#from beogym.beogym import BeoGym
import numpy as np
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.algorithm import Algorithm
import cv2
#import configs
from models.beogymmodels import ComplexNet
from ray.rllib.models import ModelCatalog
from IPython import embed

ModelCatalog.register_custom_model("ComplexNet", ComplexNet)
#my_restored_policy = Policy.from_checkpoint("/lab/kiran/logs/rllib/beogym/notemp/1.a_Wall_Street_singlegame_full_e2e_PolicyNotLoaded_0.0_20000_2000_notemp/23_11_03_01_08_55/checkpoint/")
my_restored_policy = Policy.from_checkpoint("/lab/kiran/logs/rllib/beogym/notemp/1.a_Wall_Street_singlegame_full_3CHAN_TCN_BEOGYM_EXPERT_3CHAN_WALLSTREET_STANDARD_1.0_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_11_03_15_03_38/checkpoint/")

#env = BeoGym({'city':'Union_Square','data_path':'/home6/tmp/kiran/'})

from envs import SingleBeoEnv
env = SingleBeoEnv({'env': 'Union_Square', 'data_path': '/home6/tmp/kiran/'})
#env = SingleBeoEnv({'env': 'Wall_Street', 'data_path': '/home6/tmp/kiran/'})
res = []

for _ in range(5):
    total = 0
    obs = env.reset()
    for _ in range(1000):
        
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
