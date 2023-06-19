
#This file contains 2 models singletask, multitask


import functools
from typing import Optional

import numpy as np
import tree
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models.torch.model import TorchModel
from ray.rllib.models.base_model import RecurrentModel, Model, ModelIO
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC

from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.visionnet import VisionNetwork
#from vaemodel import SmallVAE as VAE
from atari_vae import VAE as VAE
from IPython import embed

from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)

torch, nn = try_import_torch()


# The global, shared layer to be used by both models.
# this model outputs a 512 latent dimension
ATARI_GLOBAL_SHARED_BACKBONE= VAE(channel_in=4, z=512)
#if using lstm this could be used:
#TORCH_GLOBAL_SHARED_BACKBONE= VAE(channel_in=1, z=512)

ATARI_GLOBAL_SHARED_POLICY = SlimFC(
    64,
    18,
    activation_fn=nn.ReLU,
    initializer=torch.nn.init.xavier_uniform_,
)

BEOGYM_GLOBAL_SHARED_BACKBONE= VAE(channel_in=4, z=512)
#if using lstm this could be used:
#TORCH_GLOBAL_SHARED_BACKBONE= VAE(channel_in=1, ch=32, z=512)

BEOGYM_GLOBAL_SHARED_POLICY = SlimFC(
    64,
    5,
    activation_fn=nn.ReLU,
    initializer=torch.nn.init.xavier_uniform_,
)

#this is class is used when we are working with a single game
class SingleAtariModel(VisionNetwork):


    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        if 'e2e' not in model_config['custom_model_config']['backbone'] and model_config['custom_model_config']['backbone'] != 'random':

            if '1channel' in model_config['custom_model_config']['backbone']:
                print("loading model 1channel lsjdflsf")
                self._convs = VAE(channel_in=1, z=512)
            elif '3channel' in model_config['custom_model_config']['backbone']:
                self._convs = VAE(channel_in=3, z=512)
            elif '4stack' in model_config['custom_model_config']['backbone']:
                self._convs = VAE(channel_in=4, z=512)
            else:
                raise NotImplementedError("vae model not implemented")
        
        if model_config['custom_model_config']['backbone_path'] != None:
            checkpoint = torch.load(model_config['custom_model_config']['backbone_path'], map_location="cpu")
            self._convs.load_state_dict(checkpoint['model_state_dict'])
        
        print(model_config, num_outputs)
        #from IPython import embed; embed()
        if model_config['custom_model_config']['freeze_backbone']:
            print("freezing backbone layers")
            self._convs.eval()
            for param in self._convs.parameters():
                param.requires_grad = False
        #from IPython import embed; embed()
        print(self._convs)
        print(self.num_outputs)
    
#this is class is reused for every game/city/town
#this is equivalent to a spec in rl_module api
class SharedBackboneAtariModel(SingleAtariModel):

    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        super().__init__(observation_space, action_space, num_outputs, model_config, name)
        self.backbone = ATARI_GLOBAL_SHARED_BACKBONE


#this is class is reused for every game/city/town
#this is equivalent to a spec in rl_module api
class SharedBackbonePolicyAtariModel(SingleAtariModel):

    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        super().__init__(observation_space, action_space, num_outputs, model_config, name)
        self.backbone = ATARI_GLOBAL_SHARED_BACKBONE
        self.pi = ATARI_GLOBAL_SHARED_POLICY





