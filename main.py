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

    #update the args in the config.py file
    args.num_workers, args.num_envs, args.num_gpus, args.gpus_worker, args.cpus_worker, args.roll_frags = config_data[args.machine]
    
    ray.init(local_mode=args.local_mode)

    
    #log directory
    str_logger = args.env_name + "_" + args.model + "_" + str(args.stop_timesteps) + "_lr" + str(args.lr) + "_lam" + str(args.lambda_) + "_kl" + str(args.kl_coeff) + "_cli" + str(args.clip_param) + "_ent" + str(args.entropy_coeff) + "_gam" + str(args.gamma) + "_buf" + str(args.buffer_size) + "_bat" + str(args.batch_size) + "_num" + str(args.num_epoch)

    #to the config you need to add str_logger!!!!!!! '/' + str_logger

    #training, once finished, save the logs
    


    #construct the model

    #if gradients flow through the backbone or a pretrained backbone is loaded and frozen


    #if the policy/adapter/policy+adapter is swappable


    #what is the training paradigm.. how is it trained on what games


    #Before you start training. run evaluate to check how much reward can a random agent do
    #Towards the end evaluation on the trained games to see the final reward

    if args.train:

        #if the program is run for all the games independently
        #and so you run the training procedure independently across all the games
        if args.setting == 'eachgame':
            #get the list of train or test environments from args.trainset

            for each game:
                #env, prtr_model, str_logger
                #in this case you create a new adapter
                #policy,backbone stays fixed (pretrained/train)
                #if args.expname == 'adapter'
                #E2E PRETRAINED BACKBONE or FIXED PRETRAINED BACKBONE or FIXED BACKBONE AND POLICY
                #baseline 1.a
                #baseline 1.c, 1.d
                #baseline 1.e, 1.f
                #baseline 3.a.ft, 3.b.ft, 3.c.ft
                train.single_train(args.env_name, args.trainset, args.prtr, args.adapter, args.policy, args.expname, args.eval, str_logger)



        #if the games are trained sequentially
        #if args.setting is 'seq_game'.
        #Assumption: everytime its looped across all games
        elif args.setting == 'seqgame':

            #baseline 2.a
            #baseline 2.b (same )
            #baseline 2.c (our method)
            #expname == 'full', E2E through all envs
            train.seq_train(args.env_name, args.trainset, args.prtr, args.adapter, args.policy, args.expname, str_logger)



        #if its all games and the model is trained e2e
        #for now only the entire model trained on allgames or setgames is implemented
        else:

            #if you want to run the entire model e2e on multiple envs
            if ** == experiment:
                #baseline 1.b (allenvs)
                #baseline 3.a.tr (setenvs)
                train.single_train(envs, args.prtr, args.expname, str_logger)
            
            elif:
                #baseline 1.c, 1.e, 1.f, 3.c.tr: adapter+policy
                #baseline 1.d, 3.d.tr, 3.c.tr: policy
                train.multi_train()

