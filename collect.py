import gym
import numpy as np
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.algorithm import Algorithm
import cv2 
import ray
#import configs
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
from typing import Dict, Tuple
import gym
import distutils.dir_util
from gym import spaces
from ray.rllib.policy.sample_batch import SampleBatch
from IPython import embed
import shutil
import distutils.dir_util
from pathlib import Path
from envs import SingleAtariEnv
import pickle
import argparse

dparser = argparse.ArgumentParser()


dparser.add_argument(
    "--stop_timesteps", type=int, default=1000000, help="Number of timesteps to train."
)
dparser.add_argument(
    "--dpath", type=str, default="/lab/tmpig14c/kiran/trained_4stack_", help="Number of timesteps to train."
)
dparser.add_argument(
    "--game", type=str, default="", help="machine to be training"
)

args = dparser.parse_args()


ModelCatalog.register_custom_model("model", SingleAtariModel)


choices = {
    'CarnivalNoFrameskip-v4': ('carnival', '/lab/kiran/logs/rllib/atari/4stack/1.a_CarnivalNoFrameskip-v4_singlegame_full_4STACK_CONT_ATARI_EXPERT_4STACK_TRAIN_RESNET_0.1_0.01_512_512.pt_PolicyNotLoaded_0.0_20000_2000_4stack/23_08_07_18_30_07/checkpoint/'),
    'DemonAttackNoFrameskip-v4': ('demonattack', '/lab/kiran/logs/rllib/atari/4stack/1.a_DemonAttackNoFrameskip-v4_singlegame_full_4STACK_CONT_ATARI_EXPERT_4STACK_TRAIN_RESNET_0.1_0.01_512_512.pt_PolicyNotLoaded_0.0_20000_2000_4stack/23_08_06_09_12_11/checkpoint/'),
    'AirRaidNoFrameskip-v4': ('airraid', '/lab/kiran/logs/rllib/atari/4stack/1.a_AirRaidNoFrameskip-v4_singlegame_full_4STACK_CONT_ATARI_EXPERT_4STACK_TRAIN_RESNET_0.1_0.01_512_512.pt_PolicyNotLoaded_0.0_20000_2000_4stack/23_08_06_01_30_05/checkpoint/'),
    'NameThisGameNoFrameskip-v4': ('namethisgame', '/lab/kiran/logs/rllib/atari/4stack/1.a_NameThisGameNoFrameskip-v4_singlegame_full_4STACK_CONT_ATARI_EXPERT_4STACK_TRAIN_RESNET_0.1_0.01_512_512.pt_PolicyNotLoaded_0.0_20000_2000_4stack/23_08_06_09_14_07/checkpoint/'),
    'SpaceInvadersNoFrameskip-v4': ('spaceinvaders', '/lab/kiran/logs/rllib/atari/4stack/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_4STACK_CONT_ATARI_EXPERT_4STACK_TRAIN_RESNET_0.1_0.01_512_512.pt_PolicyNotLoaded_0.0_20000_2000_4stack/23_08_06_09_16_00/checkpoint/')
}

encodernet = Policy.from_checkpoint(choices[args.game][1])



env = SingleAtariEnv({'env': args.game, 'full_action_space': False, 'framestack': '4stack' in choices[args.game][1]})

obs_np = []
act_np = []
rew_np = []
done_np = []

while True:
    reward = 0.0
    done = False
    total=0
    obs = env.reset()
    while True:
        action = encodernet.compute_single_action(obs)[0]
        
        obs_np.append(obs)
        
        obs, reward, done, _ = env.step(action)
        
        act_np.append(action)
        rew_np.append(reward)
        done_np.append(done)

        total += reward
        if done:
            break
    print(total)

    if len(act_np) > args.stop_timesteps:
        break




np.save(args.dpath + choices[args.game][0] + '/observation', np.array(obs_np))
np.save(args.dpath + choices[args.game][0] + '/action', np.array(act_np))
np.save(args.dpath + choices[args.game][0] + '/reward', np.array(rew_np))
np.save(args.dpath + choices[args.game][0] + '/terminal', np.array(done_np))


obs = np.array(obs_np)
rew = np.array(rew_np)
act = np.array(act_np)
ter = np.array(done_np)

ter[-1]=1
np.save(args.dpath + choices[args.game][0] + '/terminal', ter)
indices = np.where(ter == 1)


slices_a = []
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
    slices_a.append(act[start_idx:idx+1])
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

#for arr in slices_r:
#    a=0.95
#    powers = np.arange(arr.size)
#    output = np.array([np.sum(arr[i:] * a ** powers[: arr.size - i]) for i in range(arr.size)])
#    slices_v.append(output)


#originally it was episode... instead of limit
#data1 = np.concatenate(all_val[0], axis=0)
np.save(args.dpath + choices[args.game][0] + '/limit', np_limit)
np.save(args.dpath + choices[args.game][0] + '/episode', np_epi)
np.save(args.dpath + choices[args.game][0] + '/id_dict', id_dict)
