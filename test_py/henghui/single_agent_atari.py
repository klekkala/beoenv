"""Simple example of setting up a multi-agent policy mapping.

Control the number of agents and policies via --num-agents and --num-policies.

This works with hundreds of agents and policies, but note that initializing
many TF policies will take some time.

Also, TF evals might slow down with large numbers of policies. To debug TF
execution, set the TF_TIMELINE_DIR environment variable.
"""

import argparse
import os
import random
import gym
#import gymnasium as gym
import cv2
import ray
import numpy as np
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig

from single_cnn import SingleTorchModel
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved

tf1, tf, tfv = try_import_tf()
from ray.rllib.policy.policy import PolicySpec

parser = argparse.ArgumentParser()

parser.add_argument("--num-agents", type=int, default=4)
parser.add_argument("--num-policies", type=int, default=2)
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=200, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=150.0, help="Reward at which we stop training."
)

'''
class SingleAtariEnv(gym.Env):
    def __init__(self,env_config):
        self.env = gym.make("BeamRiderNoFrameskip-v4", full_action_space=True)
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8) #self.env.observation_space

    def reset(self, seed=None, options=None):
        obs= self.env.reset()
        return cv2.resize(obs, (84, 84))

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        obs = cv2.resize(obs, (84, 84))
        return obs, reward, terminated, info


tune.register_env('SingleAtariEnv', lambda config: SingleAtariEnv(config))
'''

if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus or None)

    ModelCatalog.register_custom_model("mod", SingleTorchModel)

    act_space = gym.spaces.Discrete(18)
    obs_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)


    config = (
        PPOConfig()
        #.environment('SingleAtariEnv', env_config={})
        .environment("BeamRiderNoFrameskip-v4", env_config={'full_action_space':True})
        .framework(args.framework)
        .rollouts(num_rollout_workers=8)
        .training(model={
            "custom_model": "mod",
            "vf_share_layers" : True},
        lambda_=0.95,
        lr=0.0001,
        kl_coeff=0.5,
        clip_param=0.1,
        vf_clip_param=10.0,
        grad_clip=100.0,
        entropy_coeff=0.01,
        #entropy_coeff=0.9,
        train_batch_size=5000,
        sgd_minibatch_size=500,
        num_sgd_iter=10)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=.4,
            num_gpus_per_worker=.2,
            num_cpus_per_worker=1)
    )





    #stop = {
    #    "episode_reward_mean": args.stop_reward,
    #    "timesteps_total": args.stop_timesteps,
    #    "training_iteration": args.stop_iters,
    #}

    results = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(verbose=1),
    ).fit()

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()
