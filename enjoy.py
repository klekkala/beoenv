import gym
import numpy as np
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.algorithm import Algorithm
import cv2 
#from envs import SingleAtariEnv
#from arguments import get_args
#from IPython import embed
from arguments import get_args
import ray
import configs
#import graph_tool.all as gt
from ray.rllib.utils.annotations import override
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print, UnifiedLogger, Logger, LegacyLoggerCallback
from ray.tune.registry import get_trainable_cls
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from models.atarimodels import SingleAtariModel, SharedBackboneAtariModel, SharedBackbonePolicyAtariModel
# from models.beogymmodels import SingleBeogymModel, SharedBackboneBeogymModel, SharedBackbonePolicyBeogymModel
from ray.rllib.algorithms.ppo import PPOConfig
from configs import atari_config
from typing import Dict, Tuple
import gym
import distutils.dir_util
from gym import spaces
from ray.rllib.policy.sample_batch import SampleBatch
import specs
from IPython import embed
import shutil
import distutils.dir_util
from pathlib import Path
from envs import SingleAtariEnv
import pickle
import imageio

ModelCatalog.register_custom_model("model", SingleAtariModel)




#encodernet = Policy.from_checkpoint('/lab/kiran/logs/rllib/atari/4stack/1.a_DemonAttackNoFrameskip-v4_singlegame_full_4STACK_CONT_ATARI_EXPERT_4STACK_DEMONATTACK_STANDARD_0.1_0.01_512_512.pt_PolicyNotLoaded_0.0_20000_2000_4stack/23_07_27_15_53_30/checkpoint/')


# objects = []
# with (open('/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_VIP_ATARI_EXPERT_1CHAN_DEMONATTACK_STANDARD_50_0.95_32_3_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_02_16_30_00/checkpoint/policy_state.pkl', "rb")) as openfile:
#     while True:
#         try:
#             objects.append(pickle.load(openfile))
#         except EOFError:
#             break

#encodernet = Policy.from_checkpoint('/lab/kiran/logs/rllib/atari/lstm/1.a_DemonAttackNoFrameskip-v4_singlegame_full_e2e_PolicyNotLoaded_0.0_20000_2000_lstm/23_08_16_00_21_36/checkpoint/')
# encodernet = Policy.from_checkpoint('/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_e2e_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_11_15_12_08/checkpoint/')

#carnival

encodernet = Policy.from_checkpoint('/lab/tmpig10c/kiran/Dropbox/logs/rllib/atari/notemp/CarnivalNoFrameskip-v4/VEP/1.a_CarnivalNoFrameskip-v4_singlegame_full_1CHAN_NVEP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_1.0_0.1_1.0_same_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_11_16_14_25_16/checkpoint')
#Beam
encodernet = Policy.from_checkpoint('/lab/tmpig10c/kiran/Dropbox/logs/rllib/atari/notemp/BeamRiderNoFrameskip-v4/VEP/1.a_BeamRiderNoFrameskip-v4_singlegame_full_1CHAN_NVEP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_2.0_-1.0_2.0_same_triplet_32_0_0.0001_1.pt_PolicyNotLoaded_0.0_20000_2000_1.0_notemp/24_01_22_18_26_19/checkpoint')
#Phoenix
encodernet = Policy.from_checkpoint('/lab/tmpig10c/kiran/Dropbox/logs/rllib/atari/notemp/PhoenixNoFrameskip-v4/VEP/1.a_PhoenixNoFrameskip-v4_singlegame_full_1CHAN_NVEP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_2.0_-1.0_2.0_same_triplet_32_0_0.0001_1.pt_PolicyNotLoaded_0.0_20000_2000_1.0_notemp/24_01_19_21_41_55/checkpoint')
#Demon
encodernet = Policy.from_checkpoint('/lab/tmpig10c/kiran/Dropbox/logs/rllib/atari/notemp/DemonAttackNoFrameskip-v4/VEP/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_NVEP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_2.0_-1.0_2.0_same_triplet_32_0_0.0001_1.pt_PolicyNotLoaded_0.0_20000_2000_1.0_notemp/24_01_17_10_36_52/checkpoint')
#Space
encodernet = Policy.from_checkpoint('/lab/tmpig10c/kiran/Dropbox/logs/rllib/atari/notemp/SpaceInvadersNoFrameskip-v4/VEP/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_1CHAN_NVEP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_4.0_-1.0_4.0_same_triplet_32_0_0.0001_1.pt_PolicyNotLoaded_0.0_20000_2000_1.0_notemp/24_01_18_12_45_25/checkpoint')


args = get_args()
print(args.log + "/" + args.temporal + "/" + args.backbone + "/checkpoint/")

res=[]
rounds=1

# env = SingleAtariEnv({'env': 'SpaceInvadersNoFrameskip-v4', 'full_action_space': False, 'framestack': args.temporal == '4stack'})
env = SingleAtariEnv({'env': 'SpaceInvadersNoFrameskip-v4', 'full_action_space': False, 'framestack': '1chan'})

obs_np = []
act_np = []
rew_np = []
done_np = []
frames = []

count = 0
for i in range(rounds):
    reward = 0.0
    done = False
    total=0
    obs = env.reset()
    for q in range(1000):
        action = encodernet.compute_single_action(obs)[0]
        
        obs_np.append(obs)
        frame = np.uint8(cv2.resize(obs, (384, 384), interpolation=cv2.INTER_CUBIC))  # Convert frame to np.uint8
        frames.append(frame)
        obs, reward, done, _ = env.step(action)
        
        act_np.append(action)
        rew_np.append(reward)
        done_np.append(done)

        total += reward
        if done:
            break

    res.append(total)

average = sum(res) / len(res)
print(average)

imageio.mimsave('./atari_v/SpaceInvadersNoFrameskip-v4.gif', frames, fps=30)  # Change 'animation.gif' to your desired file name and path


# import cv2

# output_file = 'output_video.mp4'
# fps = 30
# frame_size = (84, 84)  # Size of each frame

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
# out = cv2.VideoWriter(output_file, fourcc, fps, frame_size,isColor=False)
# frames = obs_np
# for idx in range(len(frames)):
#     frame = frames[idx]
#     frame = np.uint8(frame)  # Ensure frame data type is uint8
#     out.write(frame)

# out.release()

# cv2.destroyAllWindows()


