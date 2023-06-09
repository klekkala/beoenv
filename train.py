import sys
from PIL import Image
from datetime import datetime
import tempfile
import yaml
import random
import numpy as np
import math, argparse, csv, copy, time, os
from pathlib import Path
import envs
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
from typing import Dict, Tuple
import gym
from gym import spaces
from ray.rllib.policy.sample_batch import SampleBatch
import specs
from IPython import embed
from ray.rllib.algorithms.algorithm import Algorithm
from typing import List, Optional, Type, Union
from ray.rllib.utils.typing import AlgorithmConfigDict, ResultDict
from ray.tune.schedulers import PopulationBasedTraining, pb2
args = get_args()

#Only 4 functions
# each_env
# seq_env uses each_env in a for loop
# all_env: depends on the prtr, adapter, policy modules.. you train in a specific way.

#use_lstm needs to be incorporated here

class MultiPPO(PPO):
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":
            from multippo import PPOTorchPolicy
            return PPOTorchPolicy


def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

pbt_hyperparam_mutations = {
    "lambda": lambda: random.uniform(0.9, 1.0),
    "clip_param": lambda: random.uniform(0.01, 0.5),
    "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    "num_sgd_iter": lambda: random.randint(1, 30),
    "sgd_minibatch_size": lambda: random.randint(128, 16384),
    "train_batch_size": lambda: random.randint(2000, 160000),
    }

pb2_hyperparam_mutations = {
    "lambda": [0.9, 1.0],
    "clip_param": [0.01, 0.5],
    "lr": [1e-3, 1e-5],
    "num_sgd_iter": [1, 30],
    "sgd_minibatch_size": [128, 16384],
    "train_batch_size": [2000, 160000],
    }

pbt = PopulationBasedTraining(
    time_attr="time_total_s",
    perturbation_interval=120,
    resample_probability=0.25,
    # Specifies the mutations of these hyperparams
    hyperparam_mutations=pbt_hyperparam_mutations,
    custom_explore_fn=explore,
    )

pb2 = pb2.PB2(
    time_attr="time_total_s",
    perturbation_interval=50000,
    quantile_fraction=0.25,  # copy bottom % with top % (weights)
    # Specifies the hyperparam search space
    hyperparam_bounds=pb2_hyperparam_mutations
    )


def pick_config_env(str_env):
    # modify atari_config to incorporate the environments
    
    if args.env_name == 'atari':
        use_config = configs.atari_config
        use_env = envs.atari[str_env]
    elif args.env_name == 'beogym':
        use_config = configs.beogym_config
        use_env = envs.beogym[str_env]
    elif env_name == 'carla':
        use_config = configs.carla
        use_env = envs.carla[str_env]
    return use_config, use_env



#Generic train fucntion that is used across all the below setups

#list of envs, what is the backbone, 
#No Sequential transfer. Single task on all envs.
def rllib_loop(config, str_logger):

    #final modifications in the config
    if args.temporal == "lstm" or args.temporal == "attention":
        args.stop_timesteps = 75000000
    
    print("program running for, ", args.stop_timesteps)
    #load the config
    #extract data from the config file
    if args.machine != "":
        with open(configs.resource_file + '/' + args.env_name + '.yaml', 'r') as cfile:
            config_data = yaml.safe_load(cfile)

        #update the args in the config.py file
        print("updating resource parameters")
        args.num_workers, args.num_envs, args.num_gpus, args.gpus_worker, args.cpus_worker, _, args.data_path = config_data[args.machine]
        
    config.update(
                {"num_workers" : args.num_workers,
                "num_envs_per_worker" : args.num_envs,
                "num_gpus" : args.num_gpus, 
                "num_gpus_per_worker" : args.gpus_worker, 
                "num_cpus_per_worker": args.cpus_worker
                }
        )
    
    if args.env_name=='beogym':
        config['env_config']['data_path']=args.data_path
    
    print(config)


    """
    #Only train the backbone if backbone is e2e
    #get backbone and policy from setting.
    #you need to load the weights into the backbone or policy here!
    if args.policy != None:
        backbone_ckpt = Policy.from_checkpoint('/lab/kiran/ckpts/trained/' + args.backbone).get_weights()
        for params in plc.keys():
            #load the policy
            if 'mlp' in params:
                plc[i] = res_wts[i]

    #load the backbone
    if args.backbone != 'e2e':
        policy_ckpt = Policy.from_checkpoint(policy).get_weights()
        for params in plc.keys():
            #load the backbone
            if 'cnn' in params:
                plc[i] = res_wts[i]
    """

    if args.setting == 'seqgame' and config['env'] != configs.all_envs[0]:
        policy_ckpt = Policy.from_checkpoint(args.ckpt + "/" + args.prefix + "/checkpoint")
        plc.set_weights(policy_ckpt.get_weights())

  
    if args.no_tune:
        if "multiagent" in config:
            algo = MultiPPO(config=config)
            print("Using MultiPPO")
        else:
            algo = PPO(config=config)

        plc = algo.get_policy()

        # run manual training loop and print results after each iteration
        for _ in range(args.stop_timesteps):
            result = algo.train()
            print(pretty_print(result))
            
            # stop training of the target train steps or reward are reached
            #MAKE SURE YOU KEEP SAVING CHECKPOINTS
            if result["timesteps_total"] >= args.stop_timesteps:
                policy = algo.get_policy()
                policy.export_checkpoint(args.ckpt + "/" + str_logger + "/checkpoint")
                break
        algo.stop()

    """
    else:

        tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            scheduler=pbt,
            num_samples=4,
        ),
        param_space=config,
        run_config=air.RunConfig(stop={
            "timesteps_total": args.stop_timesteps,
            }),
        )
        results = tuner.fit()

    """
#ADD PREPROCESSOR TO STR_LOGGER
#DO THE RANDOM TRIALS


#Train singleenv.

#Generic train fucntion that is used across all the below setups
#what is the backbone and policy to be used if its not e2e
#No Sequential transfer. Single task on all envs.
#TECHNICALY, TRAINING THE ENTIRE MODEL ON ALL THE ENVIRONMENTS IS ALSO SINGLEENV
def single_train(str_logger, backbone='e2e', policy=None):

    # modify atari_config to incorporate the environments
    #construct the environment from envs.py based on the env_name 
    use_config, use_env = pick_config_env('single')
    env_config = {'env': args.set, 'framestack': args.temporal == '4stack'}

    if args.backbone == "e2e":
        args.train_backbone = True

    if args.env_name == 'atari':
        ModelCatalog.register_custom_model("model", SingleAtariModel)

    elif args.env_name == 'beogym':
        ModelCatalog.register_custom_model("model", SingleBeogymModel)

    #do all the config overwrites here
    use_config.update(
                {
                    "env" : use_env,
                    "env_config": env_config,
                    "logger_config" : {
                        "type": UnifiedLogger,
                        "logdir": os.path.expanduser(args.log + '/' + str_logger)
                    }
                }
            )
 



    #start the training loop
    rllib_loop(use_config, str_logger)

#SAVE CHECKPOINTS FOR 1.A PROPERLY!!!

#sequential learning
#this function reuses the train_singleenv function
def seq_train(str_logger):

    #get the base atari_config to incorporate the environments
    #construct the base env class from envs.py based on the env_name
    use_config, use_env = pick_config_env('single')

    #register the model
    ModelCatalog.register_custom_model("model", SingleAtariModel)
    
    #In the forloop base config and spec stays the same
    for eachenv in configs.all_envs: 
        #in the for loop set the previous models weights
        #adapter, policy, backbone
        #env_config consists of which games we use
        use_config.update(
            {"env" : eachenv, 
             "env_config" : {"full_action_space":True},
             "model": {"custom_model" : "model",
                        "vf_share_layers": True
             },
             "logger_config" : {
                "type": UnifiedLogger,
                "logdir": os.path.expanduser(args.log + '/' + str_logger)
                }
            }
        )

        rllib_loop(use_config)



#Multi task on all envs
def train_multienv(str_logger):

    #get the base atari_config to incorporate the environments
    #construct the base env class from envs.py based on the env_name
    use_config, use_env = pick_config_env('multi')
    env_config = {'envs': configs.all_envs}

    #construct the spec based on the environment/tasks
    #returns multipolicies and multimap
    #multistuff = specs.generate_specs()

    #register the model
    if args.shared == "full":
        mods = [SingleAtariModel]*len(configs.all_envs)
        ModelCatalog.register_custom_model("model", SingleAtariModel)
    
    elif "policy" in args.shared:
        mods = [SharedBackbonePolicyAtariModel]*len(configs.all_envs)
        for i in range(len(configs.all_envs)):
            ModelCatalog.register_custom_model("model_" + str(i), mods[i])

    elif "backbone" in args.shared:
        mods = [SharedBackboneAtariModel]*len(configs.all_envs)
        for i in range(len(configs.all_envs)):
            ModelCatalog.register_custom_model("model_" + str(i), mods[i])
    
    else:
        raise NotImplementedError
        
    if args.backbone == "e2e":
        args.train_backbone = True
    

    policies = {"policy_{}".format(i): specs.gen_policy(i) for i in range(len(configs.all_envs))}
    
    policy_ids = ["policy_{}".format(i) for i in range(len(configs.all_envs))]

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        pol_id = policy_ids[agent_id%len(mods)]
        return pol_id
    print(use_env)
    # modify atari_config to incorporate the current environment
    #do all the config overwrites here
    use_config.update(
            {
                    "env" : use_env,
                    "preprocessor_pref": "rllib",
                    "env_config": env_config,
                    "logger_config" : {
                        "type": UnifiedLogger,
                        "logdir": os.path.expanduser(args.log + '/' + str_logger)
                    },
                    "multiagent": {
                        "policies" : policies,
                        "policy_mapping_fn" : policy_mapping_fn,

                    },
                    "callbacks": envs.MultiCallbacks
            }
        )

    if args.shared == "full":
        use_config["multiagent"]["policies"] = {"model"}
        use_config["multiagent"]["policy_mapping_fn"] = (lambda agent_id, episode, **kwargs: "model")

    #multistuff is a tuple
    #adapter and policy is a list
    rllib_loop(use_config, str_logger)


def beogym_single_train(str_logger, backbone='e2e', policy=None):

    str_env = 'single' if args.setting == 'singlegame' else 'parellel'
    use_config, use_env = pick_config_env(str_env)
    env_config = {}

    if str_env == 'parellel': 
        env_config = {'envs': configs.all_envs, 'data_path':args.data_path}
    else:
        env_config = {'env': args.set,'data_path':args.data_path}


    # ModelCatalog.register_custom_model("model", SingleBeogymModel)

    use_config.update(
                {
                    "env" : envs.SingleBeoEnv,
                    'env_config':env_config,
                    "logger_config" : {
                        "type": UnifiedLogger,
                        "logdir": os.path.expanduser(args.log + '/' + str_logger)
                    },
                    # "model": {"custom_mod52el" : "model",
                    #           "vf_share_layers": True
                    # },
                }
            )

    #start the training loop
    rllib_loop(use_config, str_logger)


def beogym_seq_train(str_logger):

    #get the base atari_config to incorporate the environments
    #construct the base env class from envs.py based on the env_name
    use_config, use_env = pick_config_env('single')

    #register the model
    ModelCatalog.register_custom_model("model", SingleBeogymModel)
    
    #In the forloop base config and spec stays the same
    for eachenv in configs.all_envs: 
        #in the for loop set the previous models weights
        #adapter, policy, backbone
        #env_config consists of which games we use
        use_config.update(
            {"env" : envs.ParellelBeoEnv, 
             "env_config" : {"city":eachenv},
             "model": {"custom_model" : "model",
                        "vf_share_layers": True
             },
             "logger_config" : {
                "type": UnifiedLogger,
                "logdir": os.path.expanduser(args.log + '/' + str_logger)
                }
            }
        )

        rllib_loop(use_config)


def beogym_train_multienv(str_logger):

    #get the base atari_config to incorporate the environments
    #construct the base env class from envs.py based on the env_name
    use_config, use_env = pick_config_env('multi')

    #register the model
    if "backbone" in args.shared:
        mods = [SharedBackboneBeogymModel]*len(configs.all_envs)
    
    if "policy" in args.shared:
        mods = [SharedBackbonePolicyBeogymModel]*len(configs.all_envs)

    print(mods)
        
    for i in range(len(configs.all_envs)):
        ModelCatalog.register_custom_model("model_" + str(i), mods[i])
    

    policies = {"policy_{}".format(i): specs.gen_policy(i) for i in range(len(configs.all_envs))}
    #policy_ids = list(policies.keys())
    policy_ids = ["policy_{}".format(i) for i in range(len(configs.all_envs))]
    envs=len(mods)
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        pol_id = policy_ids[agent_id%envs]
        return pol_id

    # modify atari_config to incorporate the current environment
    #do all the config overwrites here
    use_config.update(
            {
                    "env" : use_env, 
                    "env_config" : {'envs': configs.all_envs},
                    "logger_config" : {
                        "type": UnifiedLogger,
                        "logdir": os.path.expanduser(args.log + '/' + str_logger)
                    },
                    "multiagent": {
                        "policies" : policies,
                        "policy_mapping_fn" : policy_mapping_fn,

                    },
                    #"callbacks": envs.MultiCallbacks
            }
        )


    #multistuff is a tuple
    #adapter and policy is a list
    rllib_loop(use_config)
