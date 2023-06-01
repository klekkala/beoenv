import sys
from PIL import Image
from datetime import datetime
import tempfile
import yaml
import random



import numpy as np
import math, argparse, csv, copy, time, os
from pathlib import Path

from arguments import get_args
import ray
import config
from ray.rllib.utils.annotations import override
from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

#import model
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print, UnifiedLogger, Logger, LegacyLoggerCallback
from ray.tune.registry import get_trainable_cls

args = get_args()

#Only 4 functions
# each_env
# seq_env uses each_env in a for loop
# all_env: depends on the prtr, adapter, policy modules.. you train in a specific way.

#use_lstm needs to be incorporated here


def pick_config():
    # modify atari_config to incorporate the environments
    if env_name == 'Atari':
        use_config = config.atari
    elif env_name == 'Beo':
        use_config = config.beo
    elif env_name == 'Carla':
        use_config = config.carla



#Generic train fucntion that is used across all the below setups

#list of envs, what is the backbone, 
#No Sequential transfer. Single task on all envs.
def rllib_loop(config, backbone=None, policy=None):

    #load the config
    algo = PPO(config=config)

    #get backbone and policy from setting.
    backbone_ckpt = Policy.from_checkpoint(backbone).get_weights()
    policy_ckpt = Policy.from_checkpoint(policy).get_weights()

    #you need to load the weights into the backbone or policy here!
    plc = algo.get_policy().get_weights()
    for params in plc.keys():
        #load the policy
        if 'mlp' in params:
            plc[i] = res_wts[i]
        #load the backbone

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




#THINGS TO CHANGE IN THE CONFIG FILE
#which config to pick, env name, str_logger


#Train singleenv.

#Generic train fucntion that is used across all the below setups

#list of envs, what is the backbone, 
#No Sequential transfer. Single task on all envs.
#TECHNICALY, TRAINING THE ENTIRE MODEL ON ALL THE ENVIRONMENTS
#IS ALSO SINGLEENV
def single_train(str_logger, backbone, adapter, policy):

    # modify atari_config to incorporate the environments
    use_config = pick_config()

    #construct the environment from envs.py based on the env_name
    #envclass = 

    # modify atari_config to incorporate the current environment
    #do all the config overwrites here
    config.override(env_name=envclass)

    
    #start the training loop
    rllib_loop(config.atari_config, adapter, policy)



#sequential learning
#this function reuses the train_singleenv function
def seq_train(str_logger, backbone, adapter, policy):

    # modify atari_config to incorporate the environments
    use_config = pick_config()

    #construct the environment
    allenvs = depend_on_the_env_pick()
    
    #In the forloop base config and spec stays the same
    for eachenv in allenvs:
        if env!=envs[0]:
            #in the for loop set the previous models weights
            backbone = backbone
            config.override(env_name = eachenv)
            rllib_loop(config, backbone)
        else:
            #adapter, policy, backbone
            config.override(env_name = eachenv)
            rllib_loop(config, adapter, policy)




#Multi task on all envs
def train_multienv(str_logger, backbone, adapter, policy)

    #pick the config based on environments and tasks
    use_config = pick_config()

    #construct the environment
    envclass = depend_on_the_env_pick() #atari or beo or smth

    #construct the spec based on the environment/tasks
    #returns multipolicies and multimap
    multistuff = specs.generate_specs()

    # modify atari_config to incorporate the current environment
    #do all the config overwrites here
    config.override(env_name=envclass)
    
    #multistuff is a tuple
    #adapter and policy is a list
    train_using_rllib(config, backbone, policy)


