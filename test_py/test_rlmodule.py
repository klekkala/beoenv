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

layers = []
(w, h, in_channels) = (84,84,3)
filters = [[16,[8,8],4],[32,[4,4],2],[256,[11,11],1]]
activation='relu'
in_size = [w, h]
for out_channels, kernel, stride in filters[:-1]:
    padding, out_size = same_padding(in_size, kernel, stride)
    layers.append(
        SlimConv2d(
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            activation_fn=activation,
        )
    )
    in_channels = out_channels
    in_size = out_size

out_channels, kernel, stride = filters[-1]
layers.append(
    SlimConv2d(
        in_channels,
        out_channels,
        kernel,
        stride,
        None,  # padding=valid
        activation_fn=activation,
    )
)

layers.append(
    nn.Flatten()
)



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

tune.register_env('MultiSync', lambda config: MultiSync(config))

class BCTorchRLModuleWithSharedGlobalEncoder(TorchRLModule):
    """An RLModule with a shared encoder between agents for global observation."""

    def __init__(
        self, encoder: nn.Module, hidden_dim: int, action_dim: int
    ) -> None:
        super().__init__(config=None)

        self.encoder = encoder
        self.policy_head = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )


    def _forward_inference(self, batch):
        with torch.no_grad():
            return self._common_forward(batch)

    def _forward_exploration(self, batch):
        with torch.no_grad():
            return self._common_forward(batch)

    def _forward_train(self, batch):
        return self._train_forward(batch)

    def _common_forward(self, batch):
        obs = batch["obs"]
        with torch.no_grad():
            global_enc = self.encoder(obs).detach()
        policy_in = global_enc
        action_logits = self.policy_head(policy_in)

        return {"action_dist": torch.distributions.Categorical(logits=action_logits)}

class BCTorchMultiAgentModuleWithSharedEncoder(MultiAgentRLModule):
    def __init__(self, config: MultiAgentRLModuleConfig) -> None:
        super().__init__(config)

    def build(self):

        module_specs = self.config.modules
        module_spec = next(iter(module_specs.values()))
        hidden_dim = module_spec.model_config_dict["fcnet_hiddens"][0]
        shared_encoder = Encoder(channels = 3)

        rl_modules = {}
        for module_id, module_spec in module_specs.items():
            rl_modules[module_id] = BCTorchRLModuleWithSharedGlobalEncoder(
                encoder=shared_encoder,
                hidden_dim=hidden_dim,
                action_dim=module_spec.action_space.n,
            )

        self._rl_modules = rl_modules



spec = MultiAgentRLModuleSpec(
    marl_module_class=BCTorchMultiAgentModuleWithSharedEncoder,
    module_specs={
        "airraid": SingleAgentRLModuleSpec(
            observation_space=gym.spaces.Dict(
                {
                    "global": gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)
                }
            ),
            action_space=gym.spaces.Discrete(18),
            model_config_dict={"fcnet_hiddens": [64]},
        ),
        "assault": SingleAgentRLModuleSpec(
            observation_space=gym.spaces.Dict(
                {
                    "global": gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)
                }
            ),
            action_space=gym.spaces.Discrete(18),
            model_config_dict={"fcnet_hiddens": [64]},
        ),
    },
)

#module = spec.build()
#marl_module = module.as_multi_agent()


config = (
    PPOConfig()
    .environment(MultiSync, clip_rewards=True)
    .training(_enable_learner_api=True)
    .rl_module(
        _enable_rl_module_api=True,
        rl_module_spec=spec,
    )
    .training(model={"vf_share_layers": True,
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
    .rollouts(num_rollout_workers=24,
        num_envs_per_worker=1,
        rollout_fragment_length='auto')
    .resources(num_gpus=0,
        num_learner_workers=2,
        num_cpus_per_worker=1,
        num_cpus_for_local_worker = 1,
        num_gpus_per_learner_worker = 1)
)

algorithm = config.build()

# run for some training steps
for i in range(2000000):
    result = algorithm.train()
    #if i%1000 == 0:
    pprint(result)

