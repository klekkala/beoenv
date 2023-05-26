from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.tf.visionnet import VisionNetwork
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.tune.registry import register_env

from ray.rllib.utils.nested_dict import NestedDict

from ray.rllib.core.rl_module.rl_module import RLModuleConfig
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
#envs = ["ALE/SpaceInvaders-v5","ALE/SpaceInvaders-v5"]

train_env = "SpaceInvadersNoFrameskip-v4"

class SingleTaskEnv(gym.Env): 
    def __init__(self, env_config):
        self.env = gym.make("SpaceInvadersNoFrameskip-v4", full_action_space=True)
        self.name= train_env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        temp = self.env.reset()
        #if isinstance(temp, np.ndarray):
        #    return cv2.resize(temp, (84, 84))
        #if str(type(temp))!='tuple':
            #return cv2.resize(temp, (84, 84))
        temp=list(temp)
        temp[0] = cv2.resize(temp[0], (84, 84))
        return tuple(temp)
    
    def step(self, action):
        temp = self.env.step(action)
        #if isinstance(temp, np.ndarray):
        #    return cv2.resize(temp, (84, 84))
        temp=list(temp)
        temp[0] = cv2.resize(temp[0], (84, 84))
        #res = tuple((cv2.resize(temp[0], (84, 84)),temp[1],temp[2],temp[3],temp[4]))
        return tuple(temp)


import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """
    def __init__(self, channel_in, channel_out, scale=2):
        super(ResDown, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = nn.AvgPool2d(scale, scale)
        
    def forward(self, x):
        skip = self.conv3(self.AvePool(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.AvePool(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """
    def __init__(self, channel_in, channel_out, scale=2):
        super(ResUp, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        
        self.UpNN = nn.Upsample(scale_factor = scale,mode = "nearest")
        
    def forward(self, x):
        skip = self.conv3(self.UpNN(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.UpNN(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x
    

class Encoder(nn.Module):
    """
    Encoder block
    Built for a 3x64x64 image and will result in a latent vector of size z x 1 x 1
    As the network is fully convolutional it will work for images LARGER than 64
    For images sized 64 * n where n is a power of 2, (1, 2, 4, 8 etc) the latent feature map size will be z x n x n

    When in .eval() the Encoder will not sample from the distribution and will instead output mu as the encoding vector
    and log_var will be None
    """
    def __init__(self, channels, ch=64, z=512):
        super(Encoder, self).__init__()
        self.conv1 = ResDown(channels, ch)  # 64
        self.conv2 = ResDown(ch, 2*ch)  # 32
        self.conv3 = ResDown(2*ch, 4*ch)  # 16
        self.conv4 = ResDown(4*ch, 8*ch)  # 8
        self.conv5 = ResDown(8*ch, 8*ch)  # 4
        self.conv_mu = nn.Conv2d(8*ch, z, 2, 2)  # 2
        self.conv_log_var = nn.Conv2d(8*ch, z, 2, 2)  # 2

    def sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x):
        x = x.float()
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.flatten(start_dim=1)

        """
        if self.training:
            mu = self.conv_mu(x)
            log_var = self.conv_log_var(x)
            x = self.sample(mu, log_var)
        else:
            mu = self.conv_mu(x)
            x = mu
            log_var = None
        """
        #return x, mu, log_var
        return x



tune.register_env('SingleTaskEnv', lambda config: SingleTaskEnv(config))



spec = SingleAgentRLModuleSpec(
    #module_class=DiscreteBCTorchModule,
    observation_space=gym.spaces.Box(0, 255, (84, 84, 3)),
    action_space=gym.spaces.Discrete(18),
    model_config_dict={"fcnet_hiddens": [64]},
)

#module = spec.build()
#marl_module = module.as_multi_agent()


config = (
    PPOConfig()
    .environment(SingleTaskEnv,
                disable_env_checking= True)
    .training(_enable_learner_api=True)
    .rl_module(
        _enable_rl_module_api=True,
        rl_module_spec=spec,
    )
    .training(model={"fcnet_hiddens": [256, 256]})
    .rollouts(num_envs_per_worker=5)
    .resources(num_gpus=0,
        num_cpus_for_local_worker = .5,
        num_gpus_per_learner_worker = .5)
)

algorithm = config.build()

# run for some training steps
for i in range(2000000):
    result = algorithm.train()
    #if i%1000 == 0:
    pprint(result)

