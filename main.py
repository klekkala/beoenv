import sys
from PIL import Image
from datetime import datetime
import tempfile
import yaml
import random



import numpy as np
import math, argparse, csv, copy, time, os
from pathlib import Path

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



if __name__ == "__main__":
    # Load the hdf5 files into a global variable

    torch, nn = try_import_torch()

    if args.tune:
        args.config = '/lab/kiran/BeoEnv/tune.yaml'

    #extract data from the config file
    if args.machine is not None:
        with open(args.config, 'r') as cfile:
            config_data = yaml.safe_load(cfile)

    args.num_workers, args.num_envs, args.num_gpus, args.gpus_worker, args.cpus_worker, args.roll_frags = config_data[args.machine]
    
    ray.init(local_mode=args.local_mode)

    
    #log directory
    str_logger = args.env_name + "_" + args.model + "_" + str(args.stop_timesteps) + "_lr" + str(args.lr) + "_lam" + str(args.lambda_) + "_kl" + str(args.kl_coeff) + "_cli" + str(args.clip_param) + "_ent" + str(args.entropy_coeff) + "_gam" + str(args.gamma) + "_buf" + str(args.buffer_size) + "_bat" + str(args.batch_size) + "_num" + str(args.num_epoch)

    #to the config you need to add str_logger!!!!!!! '/' + str_logger

    #training, once finished, save the logs
    
    if mode == train:
        #if the program is run for all the games independently
        if args.setting == 'eachgame':
            #get the list of train or test environments from args.trainset

            for each game:            
                #env, prtr_model, str_logger
                if args.expname == 'adapter':
                    train.train_singletask(envs, args.prtr, args.expname, str_logger)
                
                elif args.expname == 'adapterpolicy':
                    lkajsdlfkj
                
                elif args.expname == 'policy':
                    lkajsdlkj
        



        #if its all games
        elif args.setting == 'allgames' and args.expname == 'full':
            if ** == experiment:
                train.multi_env(envs, args.prtr, args.expname, str_logger)
            
            elif ** == experiment:
                lkajsdlfkj
            
            else:
                lkajsdlkj


        #if its seq games
        else:
            if args.expname == 'full':
                train.seq_single_env()



        #finetuning, once finished, save the logs
        #for finetuning end condition is some reward vlaue which needs to be achieved
        if args.expname == 'singletask':
            tune.train_singletask(env, end_cond)

    if mode == eval:
        #evaluation
        enjoy.evaluate()
