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
    "--dpath", type=str, default="/lab/tmpig14c/kiran/expert_notemp_atariucharcolorrllib/expert_notemp_", help="Number of timesteps to train."
)
dparser.add_argument(
    "--game", type=str, default="DemonAttackNoFrameskip-v4", help="machine to be training"
)

args = dparser.parse_args()


ModelCatalog.register_custom_model("model", SingleAtariModel)

"""
choices = {
    'DemonAttackNoFrameskip-v4': ('demonattack', '/lab/kiran/logs/rllib/atari/4stack/DemonAttackNoFrameskip-v4/1.a_DemonAttackNoFrameskip-v4_singlegame_full_e2e_PolicyNotLoaded_0.0_20000_2000_4stack/24_01_09_00_46_58/checkpoint/'),
    'SpaceInvadersNoFrameskip-v4': ('spaceinvaders', '/lab/kiran/logs/rllib/atari/4stack/SpaceInvadersNoFrameskip-v4/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_e2e_PolicyNotLoaded_0.0_20000_2000_4stack/24_01_06_17_55_54/checkpoint/')
    }

"""
choices = {
    'DemonAttackNoFrameskip-v4': ('demonattack', '/lab/kiran/logs/rllib/atari/notemp/DemonAttackNoFrameskip-v4/1.a_DemonAttackNoFrameskip-v4_singlegame_full_e2e_PolicyNotLoaded_0.0_20000_2000_notemp/24_01_14_21_48_45/checkpoint/'),
    'SpaceInvadersNoFrameskip-v4': ('spaceinvaders', '/lab/kiran/logs/rllib/atari/notemp/SpaceInvadersNoFrameskip-v4/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_e2e_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_17_18_44_06/checkpoint/')
    }

encodernet = Policy.from_checkpoint(choices[args.game][1])



env = SingleAtariEnv({'env': args.game, 'full_action_space': False, 'framestack': '4stack' in choices[args.game][1]})

print('framestack is', '4stack' in choices[args.game][1])
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
        obs = (obs*255).astype(np.uint8)
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




np.save(args.dpath + choices[args.game][0] + '/5/50/observation', np.array(obs_np))
np.save(args.dpath + choices[args.game][0] + '/5/50/action', np.array(act_np))
np.save(args.dpath + choices[args.game][0] + '/5/50/reward', np.array(rew_np))
np.save(args.dpath + choices[args.game][0] + '/5/50/terminal', np.array(done_np))


ter = np.array(done_np)
ter[-1]=1
np.save(args.dpath + choices[args.game][0] + '/5/50/terminal', ter)
