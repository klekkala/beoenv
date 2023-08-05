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

ModelCatalog.register_custom_model("model", SingleAtariModel)




encodernet = Policy.from_checkpoint('/lab/kiran/logs/rllib/atari/4stack/1.a_DemonAttackNoFrameskip-v4_singlegame_full_4STACK_CONT_ATARI_EXPERT_4STACK_DEMONATTACK_STANDARD_0.1_0.01_512_512.pt_PolicyNotLoaded_0.0_20000_2000_4stack/23_07_27_15_53_30/checkpoint/')


args = get_args()
print(args.log + "/" + args.temporal + "/" + args.backbone + "/checkpoint/")

res=[]
rounds=50

env = SingleAtariEnv({'env': args.set, 'full_action_space': False, 'framestack': args.temporal == '4stack'})

obs_np = []
act_np = []
rew_np = []
done_np = []

count = 0
for i in range(rounds):
    reward = 0.0
    done = False
    total=0
    obs = env.reset()
    for q in range(1000):
        action = encodernet.compute_single_action(obs)[0]
        
        obs_np.append(obs)
        
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


#with open('Name.txt','w') as f:
#    f.write(str(res))
