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
from models.atarimodels import SingleAtariModel, SharedBackboneAtariModel, SharedBackbonePolicyAtariModel, AtariCNNV2PlusRNNModel
from models.beogymmodels import SingleBeogymModel, BeogymCNNV2PlusRNNModel, FrozenBackboneModel, SingleImageModel
from ray.rllib.algorithms.ppo import PPOConfig
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
                "num_cpus_per_worker": args.cpus_worker,
                "train_batch_size": args.buffer_size,
                "sgd_minibatch_size": args.batch_size
                }
        )
    
    if args.env_name=='beogym':
        config['env_config']['data_path']=args.data_path
    
    print(config)

    #copy the current codebase to the log directory
    path = Path(args.log + "/" + args.env_name + "/" + str_logger)
    path = Path(args.log + "/" + args.env_name + "/" + str_logger + "/beoenv")
    path.mkdir(parents=True, exist_ok=True)
    distutils.dir_util.copy_tree("/lab/kiran/beoenv/", args.log + "/" + str_logger + "/beoenv/")

    ##Training starts
    #if args.no_tune:
    if "multiagent" in config:
        algo = MultiPPO(config=config)
        print("Using MultiPPO")
    else:
        algo = PPO(config=config)

    plc = algo.get_policy()

    #Only train the backbone if backbone is e2e
    #get backbone and policy from setting.
    #you need to load the weights into the backbone or policy here!

    
    if args.setting == 'seqgame' and config['env_config']['env'] != configs.all_envs[0]:
        policy_ckpt = Policy.from_checkpoint(args.log + "/" + args.env_name + "/" + str_logger.replace(config['env_config']['env'] + '/', '') + "/checkpoint")
        plc.set_weights(policy_ckpt.get_weights())

    #if args.policy != None:
    #    backbone_ckpt = Policy.from_checkpoint('/lab/kiran/ckpts/trained/' + args.backbone).get_weights()
    #    for params in plc.keys():
            #load the policy
    #        if 'mlp' in params:
    #            plc[i] = res_wts[i]

    #load the backbone
    if args.backbone != 'e2e' and 'e2e' in args.backbone:
        embed()
        load_ckpt = Policy.from_checkpoint(args.log + "/" + args.env_name + "/" + args.temporal + "/" + args.backbone + "/checkpoint").get_weights()
        embed()
        orig_wts = plc.get_weights()
        chng_wts = {}
        for params in load_ckpt.keys():
            if 'logits' not in params and 'value' not in params:
                print(params)
                chng_wts[params] = load_ckpt[params]
            else:
                chng_wts[params] = orig_wts[params]
        plc.set_weights(chng_wts)


    # run manual training loop and print results after each iteration
    for _ in range(args.stop_timesteps):
        result = algo.train()
        print(pretty_print(result))
        
        # stop training of the target train steps or reward are reached
        #MAKE SURE YOU KEEP SAVING CHECKPOINTS
        if result["timesteps_total"] >= args.stop_timesteps:
            algo.save(checkpoint_dir=args.log + "/" + args.env_name + "/" + str_logger.replace(config['env_config']['env'] + '/', '') + "/checkpoint/wholealgo")
            policy = algo.get_policy()
            policy.export_checkpoint(args.log + "/" + args.env_name + "/" + str_logger.replace(config['env_config']['env'] + '/', '') + "/checkpoint")
            break
    
    algo.stop()


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
    env_config = {'env': args.set, 'framestack': args.temporal == '4stack', 'full_action_space': False}

    if args.backbone == "e2e":
        args.train_backbone = True


    if "1CHAN" in args.backbone and args.temporal == "lstm":
        ModelCatalog.register_custom_model("model", AtariCNNV2PlusRNNModel)
        env_config['framestack'] = False
        use_config["model"]["use_lstm"] = False

    #include notemp case     
    else:
        ModelCatalog.register_custom_model("model", SingleAtariModel)

    #do all the config overwrites here
    use_config.update(
                {
                    "env" : use_env,
                    "env_config": env_config,
                    "logger_config" : {
                        "type": UnifiedLogger,
                        "logdir": os.path.expanduser(args.log + '/' + args.env_name + '/' + str_logger)
                    }
                }
            )

    #start the training loop
    rllib_loop(use_config, str_logger)



def beogym_single_train(str_logger, backbone='e2e', policy=None):

    use_config, use_env = pick_config_env('single')
    env_config = {'env': args.set,'data_path': args.data_path}

    if args.backbone == "e2e":
        args.train_backbone = True

    else:
        args.train_backbone = False
    """
    if args.temporal == "lstm":
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

    elif args.temporal == '2lstm':
        ModelCatalog.register_custom_model(
            "my_model", LSTM2Network
        )
        use_config.update(
                {
                    "model": {"custom_model": "my_model",
                              "vf_share_layers": True,
                              "conv_filters": [[16, 3, 2], [32, 3, 2], [64, 3, 2], [128, 3, 2], [256, 3, 2]]

                    },
                }
            )
    """
    #else:
    print("1chanlstm********************")
    ModelCatalog.register_custom_model("Single", SingleImageModel)
    #do all the config overwrites here
    use_config.update(
                {
                    "env" : use_env,
                    "env_config": env_config,
                    "logger_config" : {
                        "type": UnifiedLogger,
                        "logdir": os.path.expanduser(args.log + '/' + args.env_name + '/' + str_logger)
                    },
                    'model':{
                        "custom_model": "Single",
                        "custom_model_config" : {"backbone": args.backbone, "backbone_path": args.ckpt + args.env_name + "/" + args.backbone, "train_backbone": args.train_backbone, 'temporal': args.temporal, 'conv_filters': [[16, [8, 8], 4], [32, [4, 4], 2], [512, [11, 11], 1]]},
                        "framestack": False,
                        "use_lstm": False,
                        "vf_share_layers": True,
                        "conv_filters": [[16, [8, 8], 4], [32, [4, 4], 2], [512, [11, 11], 1]],
                    },
                }
            )

    #start the training loop
    rllib_loop(use_config, str_logger)



#SAVE CHECKPOINTS FOR 1.A PROPERLY!!!
#sequential learning
#this function reuses the train_singleenv function
def seq_train(str_logger):

    print("SEQUENTIAL MODE!")
    #get the base atari_config to incorporate the environments
    #construct the base env class from envs.py based on the env_name
    use_config, use_env = pick_config_env('single')

    if args.backbone == "e2e":
        args.train_backbone = True

    #register the model
    if args.env_name == "atari":
        ModelCatalog.register_custom_model("model", SingleAtariModel)
    
    else:
        ModelCatalog.register_custom_model("model", SingleBeogymModel)
    
    print(configs.all_envs)

    #In the forloop base config and spec stays the same
    for eachenv in configs.all_envs:
        #in the for loop set the previous models weights
        #adapter, policy, backbone
        #env_config consists of which games we use
        use_config.update(
            {"env" : use_env, 
             "env_config" : {'env': eachenv, "full_action_space": False, 'framestack': args.temporal == '4stack'},
             "logger_config" : {
                "type": UnifiedLogger,
                "logdir": os.path.expanduser(args.log + '/' + args.env_name + '/' + str_logger + "/" + eachenv + "/")
                }
            }
        )

        rllib_loop(use_config, str_logger)



#Multi task on all envs
def train_multienv(str_logger):

    #get the base atari_config to incorporate the environments
    #construct the base env class from envs.py based on the env_name
    use_config, use_env = pick_config_env('multi')
    env_config = {'envs': configs.all_envs, 'full_action_space': True, 'framestack': args.temporal == '4stack'}

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
        print("shared backbone")
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
                    "env_config": env_config,
                    "logger_config" : {
                        "type": UnifiedLogger,
                        "logdir": os.path.expanduser(args.log + '/' + args.env_name + '/' + str_logger)
                    },
                    "multiagent": {
                        "policies" : policies,
                        "policy_mapping_fn" : policy_mapping_fn,

                    },
                    "callbacks": envs.MultiCallbacks
            }
        )

    if args.shared == "full":
        print("full sharing")
        use_config["multiagent"]["policies"] = {"model"}
        use_config["multiagent"]["policy_mapping_fn"] = (lambda agent_id, episode, **kwargs: "model")

    #multistuff is a tuple
    #adapter and policy is a list
    rllib_loop(use_config, str_logger)




"""
## All the below functions are depricated

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
                        "logdir": os.path.expanduser(args.log + '/' + args.env_name + '/' + str_logger)
                    },
                    # "model": {"custom_mod52el" : "model",
                    #           "vf_share_layers": True
                    # },
                }
            )

    if backbone == '2lstm':
        ModelCatalog.register_custom_model(
            "my_model", LSTM2Network
        )
        use_config.update(
                {
                    "model": {"custom_model": "my_model",
                              "vf_share_layers": True,
                              "conv_filters": [[16, 3, 2], [32, 3, 2], [64, 3, 2], [128, 3, 2], [256, 3, 2]]

                    },
                }
            )

    #start the training loop
    rllib_loop(use_config, str_logger)



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

    print(configs.all_envs)
    # modify atari_config to incorporate the current environment
    #do all the config overwrites here
    use_config.update(
            {
                    "env" : use_env, 
                    "env_config" : {'envs': configs.all_envs, 'data_path':args.data_path},
                    "logger_config" : {
                        "type": UnifiedLogger,
                        "logdir": os.path.expanduser(args.log + '/' + args.env_name + '/' + str_logger)
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


"""
