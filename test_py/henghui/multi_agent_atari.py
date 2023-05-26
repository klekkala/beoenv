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
import cv2
import ray
import numpy as np
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
from ray.rllib.examples.models.shared_weights_model import (
    SharedWeightsModel1,
    SharedWeightsModel2,
    TF2SharedWeightsModel,
)
from shared_model_cnn import TorchSharedWeightsModel
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
envs = ["ALE/AirRaid-v5", "ALE/BeamRider-v5"]

class MultiSync(MultiAgentEnv):

        def __init__(self,num):
            self.agents=[]
            for i in range(len(envs)):
                self.agents.append(gym.make(envs[i], full_action_space=True))
            self.dones = set()
            self.action_space = gym.spaces.Discrete(18)
            self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)
            self.resetted = False

        def reset(self):
            res={}
            self.resetted = True
            self.dones = set()
            for i in range(len(envs)):
                temp = self.agents[i].reset()
                if isinstance(temp, np.ndarray):
                    temp = cv2.resize(temp, (84, 84))
                else:
                    temp=list(temp)
                    temp[0] = cv2.resize(temp[0], (84, 84))
                    temp = temp[0]
                res[i]=temp
            return res

        def step(self, action_dict):
            obs, rew, done, info = {}, {}, {}, {}
            for i, action in action_dict.items():
                temp = self.agents[i].step(action)
                if isinstance(temp, np.ndarray):
                    temp = cv2.resize(temp, (84, 84))
                else:
                    temp=list(temp)
                    temp[0] = cv2.resize(temp[0], (84, 84))
                obs[i], rew[i], done[i], _, info[i] = temp
                if done[i]:
                    self.dones.add(i)
            done["__all__"] = len(self.dones) == len(self.agents)
            return obs, rew, done, info


tune.register_env('MultiSync', lambda config: MultiSync(config))


if __name__ == "__main__":
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus or None)

    # Register the models to use.
    if args.framework == "torch":
        mod1 = mod2 = TorchSharedWeightsModel
    elif args.framework == "tf2":
        mod1 = mod2 = TF2SharedWeightsModel
    else:
        mod1 = SharedWeightsModel1
        mod2 = SharedWeightsModel2
    ModelCatalog.register_custom_model("model1", mod1)
    ModelCatalog.register_custom_model("model2", mod2)

    act_space = gym.spaces.Discrete(18)
    obs_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)
    # Each policy can have a different configuration (including custom model).
    def gen_policy(i):
        config = {
            "model": {
                "custom_model": "model1"
            }
        }
        #return PolicySpec(config=config)
        return (None, obs_space, act_space, config)

        if bool(os.environ.get("RLLIB_ENABLE_RL_MODULE", False)):
            # just change the gammas between the two policies.
            # changing the module is not a critical part of this example.
            # the important part is that the policies are different.
            config = {
                "gamma": random.choice([0.95, 0.99]),
            }
        else:
            config = PPOConfig.overrides(
                model={
                    "custom_model": ["model1", "model2"][i % 2],
                },
                gamma=random.choice([0.95, 0.99]),
            )
        return PolicySpec(config=config)

    # Setup PPO with an ensemble of `num_policies` different policies.
    policies = {"policy_{}".format(i): gen_policy(i) for i in range(args.num_policies)}
    policy_ids = list(policies.keys())

    
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        pol_id = policy_ids[agent_id%len(envs)]
        return pol_id

    config = (
        PPOConfig()
        .environment(MultiSync, env_config={})
        .framework(args.framework)
        .rollouts(num_rollout_workers=6)
        .training(num_sgd_iter=10)
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=.25,
            num_gpus_per_worker=.25,
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
