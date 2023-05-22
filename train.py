import sys
from PIL import Image
from datetime import datetime
import tempfile
import yaml
import random



import numpy as np
import math, argparse, csv, copy, time, os
from pathlib import Path

import atari_config
import argparse
import ray
from ray.rllib.utils.annotations import override
from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from models.AtariModels import VaeNetwork as TorchVae
from models.AtariModels import PreTrainedResNetwork as TorchPreTrainedRes
from models.AtariModels import ResNetwork as TorchRes
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print, UnifiedLogger, Logger, LegacyLoggerCallback
from ray.tune.registry import get_trainable_cls

args = parser.parse_args()

#Only 4 functions
# each_env
# all_env

#use_lstm needs to be incorporated here


#No Sequential transfer. Single task on all envs.
def train_singleenv(env, prtr, setting, str_logger):

    algo = PPO(config=config)
    #get adapter and policy from setting.
    # run manual training loop and print results after each iteration
    for _ in range(args.stop_timesteps):
        result = algo.train()
        print(pretty_print(result))
        # stop training of the target train steps or reward are reached
        if result["timesteps_total"] >= args.stop_timesteps:
            policy = algo.get_policy()
            policy.export_checkpoint("./tmp/atari_checkpoint")
            break
    algo.stop()


#No Sequential transfer. Multi task on all envs

#envs -> list
#str_logger -> string
#prtr -> my_model (in the form of ckpt)
#adapter -> ckpt
#policy -> ckpt
def train_multienv(envs, str_logger, prtr=None, adapter=None, policy=None)

    # modify atari_config to incorporate multienv
    #config.env_name


    algo = PPO(config=config)
    # run manual training loop and print results after each iteration
    for _ in range(args.stop_timesteps):
        result = algo.train()
        print(pretty_print(result))
        # stop training of the target train steps or reward are reached
        if result["timesteps_total"] >= args.stop_timesteps:
            policy = algo.get_policy()
            policy.export_checkpoint("./tmp/atari_checkpoint")
            break
    algo.stop()



#sequential learning
#this function reuses the train_singleenv function
def seqtrain_singleenv(env, prtr, adapter, policy, str_logger):
    if env!=envs[0]:
        my_restored_policy = Policy.from_checkpoint("./tmp/atari_checkpoint")
        policy=algo.get_policy()
        policy.set_weights(my_restored_policy.get_weights())
    for each in blah:
        
        train_singleenv()
        if env!=envs[0]:
            my_restored_policy = Policy.from_checkpoint("./tmp/atari_checkpoint")
            model, adapter, policy=algo.get_policy()
            policy.set_weights(my_restored_policy.get_weights())

    algo.stop()