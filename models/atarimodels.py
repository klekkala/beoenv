
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
from vaemodel import StackEncoder as StackEncoder
from atari_vae import VAE as VAE
from atari_vae import LSTMVAE as LSTMVAE
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
ATARI_GLOBAL_SHARED_BACKBONE= StackEncoder


#if using lstm this could be used:
ATARI_GLOBAL_SHARED_1CHANNEL_VAE = VAE(channel_in=1, z=512)

#if using 4stack this could be used:
ATARI_GLOBAL_SHARED_4STACK_VAE = VAE(channel_in=1, z=512)

ATARI_GLOBAL_SHARED_POLICY = nn.Sequential(
    nn.ZeroPad2d((0, 0, 0, 0)),
    nn.Conv2d(512, 18, kernel_size=(1, 1), stride=(1, 1))
    )


class SingleAtariModel(VisionNetwork):

    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        #if 'e2e' not in model_config['custom_model_config']['backbone'] and model_config['custom_model_config']['backbone'] != 'random':
        if 'e2e' not in model_config['custom_model_config']['backbone']:
            if model_config['custom_model_config']['backbone'] == 'random':
                self._convs = VAE(channel_in=observation_space.shape[-1], z=512)
            elif '1chan' in model_config['custom_model_config']['backbone']:
                self._convs = VAE(channel_in=1, z=512)
            elif '4stack' in model_config['custom_model_config']['backbone']:
                self._convs = VAE(channel_in=4, z=512)
            else:
                raise NotImplementedError("vae model not implemented")
        #elif model_config['custom_model_config']['backbone'] == 'e2e':
        #    
        #    if model_config['custom_model_config']['temporal'] == 'lstm' or model_config['custom_model_config']['temporal'] == 'attention':
        #        self._convs = LSTMVAE(channel_in=observation_space.shape[-1], z=512)
        #    else:
        #        self._convs = VAE(channel_in=observation_space.shape[-1], z=512)
        #else:
        #    raise NotImplementedError("lol")
        
        
        if model_config['custom_model_config']['backbone_path'] != None:
            print(model_config['custom_model_config']['backbone_path'])
            print("loading model weights")
            checkpoint = torch.load(model_config['custom_model_config']['backbone_path'], map_location="cpu")
            self._convs.load_state_dict(checkpoint['model_state_dict'])
        
        if not model_config['custom_model_config']['train_backbone']:
            print("freezing backbone layers")
            self._convs.eval()
            for param in self._convs.parameters():
                param.requires_grad = False
        print(self._convs)
        #embed()

#this is class is reused for every game/city/town
#this is equivalent to a spec in rl_module api
class SharedBackboneAtariModel(SingleAtariModel):

    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        super().__init__(observation_space, action_space, num_outputs, model_config, name)
        self._convs = ATARI_GLOBAL_SHARED_BACKBONE

#this is class is reused for every game/city/town
#this is equivalent to a spec in rl_module api

class SharedBackbonePolicyAtariModel(SingleAtariModel):

    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        super().__init__(observation_space, action_space, num_outputs, model_config, name)
        self._convs = ATARI_GLOBAL_SHARED_BACKBONE
        self._logits = ATARI_GLOBAL_SHARED_POLICY





