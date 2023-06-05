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
from ray.rllib.utils.annotations import override
from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining
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
from models.atarimodels import SingleAtariModel, SharedBackboneAtariModel, SharedBackbonePolicyAtariModel
import specs

args = get_args()

#Only 4 functions
# each_env
# seq_env uses each_env in a for loop
# all_env: depends on the prtr, adapter, policy modules.. you train in a specific way.

#use_lstm needs to be incorporated here


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
def rllib_loop(config):

    #load the config
    algo = PPO(config=config)

    plc = algo.get_policy()

    """
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


    # run manual training loop and print results after each iteration
    for _ in range(args.stop_timesteps):
        result = algo.train()
        print(pretty_print(result))
        # stop training of the target train steps or reward are reached
        #MAKE SURE YOU KEEP SAVING CHECKPOINTS
        if result["timesteps_total"] >= args.stop_timesteps:
            policy = algo.get_policy()
            policy.export_checkpoint(args.ckpt + "/" + args.prefix + "/checkpoint")
            break
    algo.stop()






#Train singleenv.

#Generic train fucntion that is used across all the below setups
#what is the backbone and policy to be used if its not e2e
#No Sequential transfer. Single task on all envs.
#TECHNICALY, TRAINING THE ENTIRE MODEL ON ALL THE ENVIRONMENTS IS ALSO SINGLEENV
def single_train(str_logger, backbone='e2e', policy=None):

    # modify atari_config to incorporate the environments
    #construct the environment from envs.py based on the env_name
    str_env = 'single' if args.setting == 'eachgame' else 'parellel'
    use_config, use_env = pick_config_env(str_env)

    
    # modify atari_config to incorporate the current environment
    #do all the config overwrites here
    use_config.update(
                {
                    "env" : use_env, 
                    "env_config" : {'envs': configs.all_envs if str_env == 'parellel' else None}, 
                    "logger_config" : {
                        "type": UnifiedLogger,
                        "logdir": os.path.expanduser(args.log + '/' + str_logger)
                    }
                }
            )

    
    #start the training loop
    rllib_loop(use_config)



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
             "env_config" : {"full_action_space":True,},
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

    #construct the spec based on the environment/tasks
    #returns multipolicies and multimap
    #multistuff = specs.generate_specs()

    #register the model
    if "backbone" in args.expname:
        mods = [SharedBackboneAtariModel]*len(configs.all_envs)
    
    if "policy" in args.expname:
        mods = [SharedBackbonePolicyAtariModel]*len(configs.all_envs)
        
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
                    }
                    ,
                    "multiagent": {
                        "policies" : policies,
                        "policy_mapping_fn" : policy_mapping_fn,

                    }
            }
        )

    #multistuff is a tuple
    #adapter and policy is a list
    rllib_loop(use_config)


