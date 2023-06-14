import sys
from PIL import Image
from datetime import datetime
import tempfile
import yaml
import random

#import gymnasium as gym
#from gymnasium import spaces
#from gymnasium.utils import seeding


import numpy as np
import math, argparse, csv, copy, time, os
from pathlib import Path
import gym
import argparse
import ray
import cv2
import torch.nn as nn
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.annotations import override
from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.visionnet import VisionNetwork as Vision
from ray.rllib.policy.policy import Policy
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
#from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
# from backboneModel import FullyConnectedNetwork as TorchFC1
# from backboneModel import FullyConnectedNetwork2 as TorchFC2
#from backboneModel2 import VisionNetwork as Vision
from shared_model_cnn import TorchSharedWeightsModel as Vision
from typing import Dict, Tuple
#from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
#from Vo import FullyConnectedNetwork as TorchFC
#from ray.rllib.models.torch.visionnet import VisionNetwork as TorchFC
#from models.AtariModels import VaeNetwork as TorchVae
#from models.AtariModels import PreTrainedResNetwork as TorchPreTrainedRes
#from models.AtariModels import ResNetwork as TorchRes
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print, UnifiedLogger, Logger, LegacyLoggerCallback
from ray.tune.registry import get_trainable_cls
from ray.rllib.algorithms.callbacks import DefaultCallbacks
#from stable_baselines3.common.env_checker import check_env
#from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3.common.env_util import make_vec_env
#from stable_baselines3.common.evaluation import evaluate_policy

#from stable_baselines3 import PPO, A2C

from ray.rllib.examples.env.mock_env import MockEnv

if __name__ == "__main__":
    # Load the hdf5 files into a global variable



    torch, nn = try_import_torch()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["vae", "res", "fcnet","random", "imagenet", "voltron", "r3m", "value"],
        default="fcnet",
    )
    parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test: --stop-reward must "
             "be achieved within --stop-timesteps AND --stop-iters.",
    )
    
    parser.add_argument(
        "--machine", type=str, default="None", help="machine to be training"
    )
    parser.add_argument(
        "--config", type=str, default="/lab/kiran/hostfile.yaml", help="config file for resources"
    )
    parser.add_argument(
        "--log", type=str, default="/lab/kiran/logs/rllib/backbone", help="config file for resources"
    )
    parser.add_argument(
        "--stop_timesteps", type=int, default=10000000, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--lambda_", type=float, default=.95, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--kl_coeff", type=float, default=.5, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--clip_param", type=float, default=.1, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--entropy_coeff", type=float, default=.01, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--gamma", type=float, default=.95, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--vf_clip", type=float, default=10, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--buffer_size", type=int, default=5000, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--batch_size", type=int, default=500, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--num_epoch", type=int, default=10, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--num_workers", type=int, default=20, help="Number of GPUs each worker has"
    )
    
    parser.add_argument(
        "--num_envs", type=int, default=8, help="Number of envs each worker evaluates"
    )

    parser.add_argument(
        "--roll_frags", type=int, default=100, help="Rollout fragments"
    )
    
    parser.add_argument(
        "--num_gpus", type=float, default=1, help="Number of GPUs each worker has"
    )

    parser.add_argument(
        "--gpus_worker", type=float, default=.3, help="Number of GPUs each worker has"
    ) 

    parser.add_argument(
        "--cpus_worker", type=float, default=.5, help="Number of CPUs each worker has"
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Run with/without Tune using a manual train loop instead. If ran without tune, use PPO without grid search and no TensorBoard.",
    )

    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )


    class MyPrintLogger(Logger):
        """Logs results by simply printing out everything."""
        def _init(self):
            # Custom init function.
            print("Initializing ...")
            # Setting up our log-line prefix.
            self.prefix = self.config.get("logger_config").get("prefix")
        def on_result(self, result: dict):
            # Define, what should happen on receiving a `result` (dict).
            print(f"{self.prefix}: {result}")
        def close(self):
            # Releases all resources used by this logger.
            print("Closing")
        def flush(self):
            # Flushing all possible disk writes to permanent storage.
            print("Flushing ;)", flush=True)



    class VisionModel(TorchModelV2, nn.Module):

        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            TorchModelV2.__init__(
                self, obs_space, action_space, num_outputs, model_config, name
            )
            nn.Module.__init__(self)

            self.torch_sub_model = TorchFC1(
                obs_space, action_space, num_outputs, model_config, name
            )

        def forward(self, input_dict, state, seq_lens):
            # input_dict["obs"]["obs"] = input_dict["obs"]["obs"].float()
            fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
            return fc_out, []

        def value_function(self):
            return torch.reshape(self.torch_sub_model.value_function(), [-1])
   


    envs = ["AirRaidNoFrameskip-v4","AssaultNoFrameskip-v4","BeamRiderNoFrameskip-v4", "CarnivalNoFrameskip-v4","DemonAttackNoFrameskip-v4","NameThisGameNoFrameskip-v4","PhoenixNoFrameskip-v4","RiverraidNoFrameskip-v4","SpaceInvadersNoFrameskip-v4"]
    #envs = ["AirRaidNoFrameskip-v4","AssaultNoFrameskip-v4"]

    class MultiTaskEnv(gym.Env): 
        def __init__(self, env_config):
            for i in range(len(envs)):    
                if env_config.worker_index%9==i:
                    self.env = gym.make(envs[i], full_action_space=True)
                    self.name= envs[i]
            #self.env = wrap_deepmind(self.env)
            self.action_space = self.env.action_space
            self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8) #self.env.observation_space
            #if self.observation_space.shape[0]==214:
                #self.observation_space = gym.spaces.Box(0, 255, (210, 160, 3), np.uint8)

        def reset(self):
            temp = self.env.reset()
            if isinstance(temp, np.ndarray):
                return cv2.resize(temp, (84, 84))
            #if str(type(temp))!='tuple':
                #return cv2.resize(temp, (84, 84))
            temp=list(temp)
            temp[0] = cv2.resize(temp[0], (84, 84))
            #res = tuple((cv2.resize(temp[0], (84, 84)),temp[1]))
            return tuple(temp)
        
        def step(self, action):
            temp = self.env.step(action)
            if isinstance(temp, np.ndarray):
                return cv2.resize(temp, (84, 84))
            temp=list(temp)
            temp[0] = cv2.resize(temp[0], (84, 84))
            #res = tuple((cv2.resize(temp[0], (84, 84)),temp[1],temp[2],temp[3],temp[4]))
            return tuple(temp)


    class MultiSync(MultiAgentEnv):

        def __init__(self,num):
            self.agents=[]
            for i in range(len(envs)):
                self.agents.append(gym.make(envs[i], full_action_space=True))
            self.terminateds = set()
            self.truncateds = set()
            self.action_space = gym.spaces.Discrete(18)
            self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)
            self.resetted = False

        def reset(self, *, seed=None, options=None):
            res={}
            info={}
            self.resetted = True
            self.terminateds = set()
            self.truncateds = set()
            for i in range(len(envs)):
                temp,info = self.agents[i].reset()
                temp = cv2.resize(temp, (84, 84))
                res[i]=temp
                info[i] = info
            #print("reset", res, info)
            return res,info

        def step(self, action_dict):
            obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}
            for i, action in action_dict.items():
                temp = self.agents[i].step(action)
                temp=list(temp)
                temp[0] = cv2.resize(temp[0], (84, 84))
                obs[i], rew[i], terminated[i], truncated[i], info[i] = temp
                if terminated[i]:
                    self.terminateds.add(i)
                if truncated[i]:
                    self.truncateds.add(i)

            terminated["__all__"] = len(self.terminateds) == len(self.agents)
            truncated["__all__"] = len(self.truncateds) == len(self.agents)
            return obs, rew, terminated, truncated, info

    # class MultiSync(MultiAgentEnv):

    #     def __init__(self,num):
    #         self.agents=[]
    #         for i in range(len(envs)):
    #             self.agents.append(gym.make(envs[i], full_action_space=True))
    #         self.dones = set()
    #         self.action_space = gym.spaces.Discrete(18)
    #         self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)
    #         self.resetted = False

    #     def reset(self):
    #         res={}
    #         self.resetted = True
    #         self.dones = set()
    #         for i in range(len(envs)):
    #             temp = self.agents[i].reset()
    #             if isinstance(temp, np.ndarray):
    #                 temp = cv2.resize(temp, (84, 84))
    #             else:
    #                 temp=list(temp)
    #                 temp[0] = cv2.resize(temp[0], (84, 84))
    #             res[i]=temp 
    #         return res

    #     def step(self, action_dict):
    #         obs, rew, done, info = {}, {}, {}, {}
    #         for i, action in action_dict.items():
    #             temp = self.agents[i].step(action)
    #             if isinstance(temp, np.ndarray):
    #                 temp = cv2.resize(temp, (84, 84))
    #             else:
    #                 temp=list(temp)
    #                 temp[0] = cv2.resize(temp[0], (84, 84))
    #             obs[i], rew[i], done[i], info[i] = temp
    #             if done[i]:
    #                 self.dones.add(i)
    #         done["__all__"] = len(self.dones) == len(self.agents)
    #         return obs, rew, done, info


    class MyCallbacks(DefaultCallbacks):
        def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
        ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
            if worker.policy_config["batch_mode"] == "truncate_episodes":
            # Make sure this episode is really done.
                assert episode.batch_builder.policy_collectors["default_policy"].batches[
                    -1
                ]["dones"][-1], (
                    "ERROR: `on_episode_end()` should only be called "
                    "after episode is done!"
                )
            episode.custom_metrics[base_env.vector_env.envs[0].name] = episode.total_reward

    args = parser.parse_args()
    
    if args.tune:
        args.config = '/lab/kiran/beoenv/tune.yaml'

    #extract data from the config file
    if args.machine is not None:
        with open(args.config, 'r') as cfile:
            config_data = yaml.safe_load(cfile)

    args.num_workers, args.num_envs, args.num_gpus, args.gpus_worker, args.cpus_worker, args.roll_frags = config_data[args.machine]
    
    ray.init(local_mode=args.local_mode)

    if args.model=='vae':
        ModelCatalog.register_custom_model(
            "my_model", TorchVaeModel
        )
    elif args.model=='imagenet':
        ModelCatalog.register_custom_model(
            "my_model", TorchPreTrainedResModel
        )
    elif args.model=='res':
        ModelCatalog.register_custom_model(
            "my_model", TorchResModel
        )



    ModelCatalog.register_custom_model("my_model1", Vision)
    #ModelCatalog.register_custom_model("my_model2", TorchFC2)
    act_space = gym.spaces.Discrete(18)
    obs_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)

    def gen_policy(i):
        config = {
            "model": {
                "custom_model": "my_model1"
                #"custom_model": ["my_model1", "my_model2"][i % 2],
            }
        }
        return (None, obs_space, act_space, config)

    # Setup PPO with an ensemble of `num_policies` different policies
    policies = {
        "policy_{}".format(i): gen_policy(i)
        for i in range(9)
    }



    str_logger = str(args.stop_timesteps) + "_lr" + str(args.lr) + "_lam" + str(args.lambda_) + "_kl" + str(args.kl_coeff) + "_cli" + str(args.clip_param) + "_ent" + str(args.entropy_coeff) + "_gam" + str(args.gamma) + "_buf" + str(args.buffer_size) + "_bat" + str(args.batch_size) + "_num" + str(args.num_epoch)
    if args.model=='fcnet':
        config = {
        "env" : MultiSync,
        "disable_env_checking" : True,
        "clip_rewards" : True,
        "framework" : "torch",
        "logger_config": {
            "type": UnifiedLogger,
            "logdir": os.path.expanduser(args.log) + '/' + str_logger
            },
        "observation_filter":"NoFilter",
        "num_workers":args.num_workers,
        "rollout_fragment_length" : args.roll_frags,
        "num_envs_per_worker" : args.num_envs,
        "multiagent": {
                "policies": policies,
                "policy_mapping_fn": lambda agent_id: "policy_"+str(agent_id)
            },
        #"model":{
        #        "vf_share_layers" : True,
        #},
        #"lambda_" : args.lambda_,
        "kl_coeff" : args.kl_coeff,
        "clip_param" : args.clip_param,
        "entropy_coeff" : args.entropy_coeff,
        "gamma" : args.gamma,
        "vf_clip_param" : args.vf_clip,
        "train_batch_size":args.buffer_size,
        "sgd_minibatch_size":args.batch_size,
        "num_sgd_iter":args.num_epoch,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus":args.num_gpus,
        "num_gpus_per_worker" : args.gpus_worker,
        "num_cpus_per_worker":args.cpus_worker,
        #"callbacks": MyCallbacks
        }
    else:
        config = {
            "env" : env,
            "clip_rewards" : True,
            "framework" : "torch",
            "logger_config": {
                "type": UnifiedLogger,
                "logdir": os.path.expanduser(args.log) + '/' + str_logger
                },
            "observation_filter":"NoFilter",
            "num_workers":args.num_workers,
            "rollout_fragment_length" : args.roll_frags,
            "num_envs_per_worker" : args.num_envs,
            "model":{
                    "custom_model" : "my_model",
                    "vf_share_layers" : True,
            },
            #"lambda_" : args.lambda_,
            "kl_coeff" : args.kl_coeff,
            "clip_param" : args.clip_param,
            "entropy_coeff" : args.entropy_coeff,
            "gamma" : args.gamma,
            "vf_clip_param" : args.vf_clip,
            "train_batch_size":args.buffer_size,
            "sgd_minibatch_size":args.batch_size,
            "num_sgd_iter":args.num_epoch,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus":args.num_gpus,
            "num_gpus_per_worker" : args.gpus_worker,
            "num_cpus_per_worker":args.cpus_worker
            }



    stop = {
        "timesteps_total": args.stop_timesteps
    }

    if args.tune == False:
        # manual training with train loop using PPO and fixed learning rate
        #if args.run != "PPO":
        #    raise ValueError("Only support --run PPO with --tune.")
        #print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        #config.lr = 5e-4
        algo = PPO(config=config)
        # run manual training loop and print results after each iteration
        for _ in range(args.stop_timesteps):
            result = algo.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if result["timesteps_total"] >= args.stop_timesteps:
                #policy = algo.get_policy()
                path_to_checkpoint = algo.save()
                print("An Algorithm checkpoint has been created inside directory: "f"'{path_to_checkpoint}'.")
                #policy.export_checkpoint("./multiatari/atari_checkpoint")
                break
        algo.stop()
    else:

        hyperparam_mutations = {
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": lambda: random.randint(1, 30),
            "sgd_minibatch_size": lambda: random.randint(128, 16384),
            "train_batch_size": lambda: random.randint(2000, 160000),
        }

        pbt = PopulationBasedTraining(
            time_attr="time_total_s",
            perturbation_interval=120,
            resample_probability=0.25,
            # Specifies the mutations of these hyperparams
            hyperparam_mutations=hyperparam_mutations,
            custom_explore_fn=explore,
        )


        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        tuner = tune.Tuner(
            args.run,
            tune_config=tune.TuneConfig(
                metric="episode_reward_mean",
                mode="max",
                scheduler=pbt,
                num_samples = 2,
            ),
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop),
        )
        results = tuner.fit()

        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)
