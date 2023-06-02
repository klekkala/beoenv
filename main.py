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

import train
import configs
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print, UnifiedLogger, Logger, LegacyLoggerCallback
from ray.tune.registry import get_trainable_cls
from arguments import get_args


if __name__ == "__main__":
    # Load the hdf5 files into a global variable

    torch, nn = try_import_torch()
    args = get_args()

    #extract data from the config file
    if args.machine != "":
        with open(config.resource_file, 'r') as cfile:
            config_data = yaml.safe_load(cfile)

        #update the args in the config.py file
        args.num_workers, args.num_envs, args.num_gpus, args.gpus_worker, args.cpus_worker, args.roll_frags = config_data[args.machine]
    
    ray.init(local_mode=args.local_mode)

    
    #log directory
    str_logger = args.prefix + "_" + args.set + "_" + args.setting + "_" + args.expname + "_" + args.adapter + "_" + args.policy + "_" + args.temporal

    #training, once finished, save the logs
    

    #if the policy/adapter/policy+adapter is swappable


    #Before you start training. run evaluate to check how much reward can a random agent do
    #Towards the end evaluation on the trained games to see the final reward

    if args.train:

        #if the program is run for all the games independently
        #and so you run the training procedure independently across all the games
        if args.setting == 'eachgame':
            #get the list of train or test environments from args.trainset

            #for eachgame:
            for _ in range(len(configs.all_envs)):
                #baseline 1.a, 1.c, 1.d, 1.e, 1.f, 3.a.ft, 3.b.ft, 3.c.ft
                train.single_train(str_logger)



        #if the games are trained sequentially
        elif args.setting == 'seqgame':

            #baseline 2.a, 2.b, 2.C
            train.seq_train(str_logger)


        #if the model is trained on all the games
        else:

            #if you want to run the entire model e2e on multiple envs
            if args.prefix == "1.b" or args.prefix == "3.a.tr" or args.prefix == "3.b.tr":
                train.single_train(str_logger)
            
            elif args.prefix == "1.c" or args.prefix == "1.e" or args.prefix == "1.f" or args.prefix == "3.c.tr" or args.prefix == "1.d" or args.prefix == "3.d.tr" or args.prefix == "3.e.tr":
                train.train_multienv(str_logger)

