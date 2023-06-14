from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.tf.visionnet import VisionNetwork
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.marl_module import (
    MultiAgentRLModuleConfig,
    MultiAgentRLModule,
)
from ray.rllib.utils.nested_dict import NestedDict

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray import tune

#from ray.rllib.models.models import Distribution

import gymnasium as gym

from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)

from pprint import pprint

from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec

import torch
import cv2
import numpy as np
import torch.nn as nn

#envs = ["AirRaidNoFrameskip-v4","AssaultNoFrameskip-v4"]
#envs = ["AirRaidNoFrameskip-v4","AssaultNoFrameskip-v4"]
envs = ["ALE/BeamRider-v5","ALE/SpaceInvaders-v5"]


#device = torch.device(0)



spec = SingleAgentRLModuleSpec(
    #module_class=DiscreteBCTorchModule,
    observation_space=gym.spaces.Box(0, 255, (84, 84, 4)),
    action_space=gym.spaces.Discrete(18),
    model_config_dict={"fcnet_hiddens": [64]},
)


#module = spec.build()
#marl_module = module.as_multi_agent()


config = (
    PPOConfig()
    .environment('ALE/BeamRider-v5', clip_rewards=True)
    .training(_enable_learner_api=True)
    .rl_module(
        _enable_rl_module_api=True,
        rl_module_spec=spec,
    )
    .training(model={"dim" : 84,
        "vf_share_layers": True,
        "fcnet_hiddens": [256, 256]},
        lambda_=0.95,
        kl_coeff=0.5,
        clip_param=0.1,
        vf_clip_param=10.0,
        entropy_coeff=0.01,
        train_batch_size=5000,
        sgd_minibatch_size=500,
        num_sgd_iter=10)
    .exploration(exploration_config={})
    .reporting(min_time_s_per_iteration=30)
    .rollouts(num_rollout_workers=1,
        num_envs_per_worker=1,
        rollout_fragment_length='auto')
    .resources(num_gpus=0,
        num_learner_workers=4,
        num_cpus_per_worker=.16,
        num_cpus_for_local_worker = 1,
        num_gpus_per_learner_worker = 1)
)

algorithm = config.build()

# run for some training steps
for i in range(2000000):
    result = algorithm.train()
    #if i%1000 == 0:
    pprint(result)

