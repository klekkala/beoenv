import sys
from PIL import Image
from datetime import datetime
import tempfile
import yaml
import random
import numpy as np
import math, argparse, csv, copy, time, os
from pathlib import Path
#import graph_tool.all as gt
import argparse
from IPython import embed
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
import datetime


if __name__ == "__main__":
    # Load the hdf5 files into a global variable

    torch, nn = try_import_torch()
    args = get_args()


    ray.init(local_mode=args.local_mode)

    #if the env_name is beogym.. then temporal mode is always lstm
    if args.env_name == "beogym":
        args.temporal = "lstm"
    
    #log directory
    suffix = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    str_logger = args.temporal + "/" + args.prefix + "_" + args.set + "_" + args.setting + "_" + args.shared + "_" + args.backbone + "_" + args.policy + "_" + str(args.kl_coeff) + "_" + str(args.buffer_size) + "_" + str(args.batch_size) + "_" + args.temporal + "/" + suffix

    #training, once finished, save the logs
    

    #if the policy/adapter/policy+adapter is swappable


    #Before you start training. run evaluate to check how much reward can a random agent do
    if args.eval:
        print("Implement start eval")


    if args.train:

        #if the program is run for all the games independently
        #and so you run the training procedure independently across all the games
        if args.setting == 'singlegame':
            #get the list of train or test environments from args.trainset
            if args.env_name == 'beogym':
                train.beogym_single_train(str_logger,args.backbone)
            else:
                train.single_train(str_logger)



        #if the games are trained sequentially
        elif args.setting == 'seqgame':

            #baseline 2.a, 2.b, 2.C
            train.seq_train(str_logger)


        #if the model is trained on all the games
        else:

            #if you want to run the entire model e2e on multiple envs
            if args.prefix == "1.b" or args.prefix == "3.a.tr" or args.prefix == "3.b.tr":
                #train.single_train(str_logger)
                train.train_multienv(str_logger)
            
            elif args.prefix == "1.c" or args.prefix == "1.d" or args.prefix == "1.f" or args.prefix == "3.c.tr" or args.prefix == "1.d" or args.prefix == "3.d.tr" or args.prefix == "3.e.tr":
                train.train_multienv(str_logger)

    #Towards the end evaluation on the trained games to see the final reward
    if args.eval:
        print("Implement end eval")
