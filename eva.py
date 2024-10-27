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
import matplotlib.pyplot as plt

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

# my_restored_policy = Policy.from_checkpoint("/lab/kiran/logs/rllib/beogym/notemp/1.a_Wall_Street_singlegame_full_3CHAN_NVEP_BEOGYM_EXPERT_3CHAN_UNIONWALL_STANDARD_2.0_-1.0_2.0_same_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_11_16_01_27_07/checkpoint/")
my_restored_policy = Policy.from_checkpoint("/lab/kiran/logs/rllib/beogym/notemp/Wall_Street/VEP/1.a_Wall_Street_singlegame_full_3CHAN_NVEP_BEOGYM_EXPERT_3CHAN_UNIONWALL_STANDARD_2.0_-1.0_2.0_same_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_11_16_22_21_35/checkpoint/")


#my_restored_policy = Policy.from_checkpoint("/lab/kiran/logs/rllib/beogym/notemp/1.a_Wall_Street_singlegame_full_3CHAN_NVEP_BEOGYM_EXPERT_3CHAN_UNIONWALL_STANDARD_1.0_-1.0_1.0_same_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_11_13_10_58_05/checkpoint/")


from envs import SingleBeoEnv
#env = SingleBeoEnv({'env': 'Union_Square', 'data_path': '/home6/tmp/kiran/'})
env = SingleBeoEnv({'env': 'Wall_Street', 'data_path': '/home6/tmp/kiran/'})


res = []

for _ in range(1):
    random_generated_int = random.randint(0, 2**31 - 1)
    env = seed_everything(random_generated_int, env)
    total = 0
    obs = env.reset()
    env.env.agent.reset(tuple(env.env.info['source']))
    env.env.courier_goal = tuple(env.env.info['goal'])
    env.seed(random_generated_int)
    posRec = []
    obsRec = []
    all_nodes=[]
    for i in env.env.dh.Gdict.keys():
        all_nodes.append(i)
    all_x,all_y =  zip(*all_nodes)
    plt.scatter(env.env.info['goal'][0], env.env.info['goal'][1], color='green',s=200)
    plt.scatter(all_x, all_y, color='blue', s=1)
    plt.xticks([])  # Remove x-axis tick labels
    plt.yticks([])  # Remove y-axis tick labels
    plt.xlabel('')  # Remove x-axis label
    plt.ylabel('')  # Remove y-axis label

    for _ in range(700):
        
        obs['obs'] = obs['obs'].astype(np.float32)
        x_r,y_r = env.env.agent.agent_pos_curr
        plt.scatter(x_r, y_r, color='red', s=40)
        figure = plt.gcf()
        figure.canvas.draw()
        img = np.array(figure.canvas.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        #plt.clf()
        posRec.append(img)
        obsRec.append(env.env.agent.curr_view)
        action = my_restored_policy.compute_single_action(obs)[0]
        obs, reward, done, _ = env.step(action)

        # obs = cv2.resize(obs, (84, 84))
        total+=reward
        if done:
            break
    res.append(total)

    height, width, _ = obsRec[0].shape
    for i in range(len(posRec)):
        posRec[i] = cv2.resize(posRec[i], (height, height))
    nn=0
    for left_img, right_img in zip(obsRec, posRec):
        # print(left_img.shape)
        # print(right_img.shape)
        concatenated_image = np.hstack((left_img, right_img))
        cv2.imwrite(f'/lab/kiran/beoenv/all_road/upload/city/Wall_Street/{nn}.jpg', concatenated_image)
        nn+=1
        # output_video.write(concatenated_image)

average = sum(res) / len(res)
print(res)
print(average)



