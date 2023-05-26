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
# seq_env uses each_env in a for loop
# all_env: depends on the prtr, adapter, policy modules.. you train in a specific way.

#use_lstm needs to be incorporated here



#Generic train fucntion that is used across all the below setups

#list of envs, what is the backbone, 
#No Sequential transfer. Single task on all envs.
def rllib_loop(envclass, config, adapter=None, policy=None):

    #update config to include envclass


    algo = PPO(config=config)
    #get adapter and policy from setting.
    # run manual training loop and print results after each iteration

    #you need to load the weights into the adapter or policy here!*********

    for _ in range(args.stop_timesteps):
        result = algo.train()
        print(pretty_print(result))
        # stop training of the target train steps or reward are reached
        if result["timesteps_total"] >= args.stop_timesteps:
            policy = algo.get_policy()
            policy.export_checkpoint("./tmp/atari_checkpoint")
            break
    algo.stop()








#Train singleenv.

#Generic train fucntion that is used across all the below setups

#list of envs, what is the backbone, 
#No Sequential transfer. Single task on all envs.
#TECHNICALY, TRAINING THE ENTIRE MODEL ON ALL THE ENVIRONMENTS
#IS ALSO SINGLEENV
def single_train(env_name, trainset, adapter, policy, expname, str_logger):


    # modify atari_config to incorporate multienv
    #config.env_name



    #construct the environment from envs.py
    envclass = 

    #construct the spec based on the environment/tasks (for rl_module api)
    #SingleAgentspec 
    #pick the config based on environments and tasks
    #.rl_module_spec = SingleAgentspec
    
    #do all the config overwrites here
    config = something

    train_using_rllib(envclass, config, prtr, adapter, policy)



#sequential learning
#this function reuses the train_singleenv function
def seq_train(env_name, trainset, adapter, policy, expname, str_logger):

    # modify atari_config to incorporate multienv
    #config.env_name


    #construct the environment
    envclass = depend_on_the_env_pick()

    #construct the spec based on the environment/tasks
    #SingleAgentspec = 
    #pick the config based on environments and tasks
    #.rl_module_spec = SingleAgentspec
    specs.generate_specs()
    
    config = something


    #In the forloop base config and spec stays the same
    for each in all:
        if env!=envs[0]:
            
            #config override new game in the environment.. set it in the env_parameters
            #in the for loop set the previous models weights
            my_restored_policy = Policy.from_checkpoint("./tmp/atari_checkpoint")
            policy=algo.get_policy()
            policy.set_weights(my_restored_policy.get_weights())

        train_singleenv()



"""
NOT YET IMPLEMENTED, WAITING FOR RLLIB BUG TO GET FIXED
#No Sequential transfer. Multi task on all envs
#envs -> list of envs
#str_logger -> string
#prtr -> my_model (in the form of ckpt)
#adapter -> ckpt/new model
#policy -> ckpt/new model
def train_multienv(envs, str_logger, prtr=None, adapter=None, policy=None)

    # modify atari_config to incorporate multienv
    #config.env_name


    #the catalogue is fixed.
    #The architecture of the entire model stays the same

    #construct the environment
    envclass = depend_on_the_env_pick()

    #construct the spec based on the environment/tasks
    MultiAgentspec = 

    #pick the config based on environments and tasks
    train_using_rllib(envclass, config, prtr, adapter, policy)
"""

