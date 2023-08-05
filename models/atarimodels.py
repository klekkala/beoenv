
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
from atari_vae import VAE
from atari_vae import Encoder, TEncoder
#from RES_VAE import Encoder as ResEncoder
from RES_VAE import TEncoder as TResEncoder
#from atari_vae import LSTMVAE as LSTMVAE
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
#ATARI_GLOBAL_SHARED_BACKBONE= StackEncoder


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
        if "RESNET" in model_config['custom_model_config']['backbone'] and "DUAL" in model_config['custom_model_config']['backbone']:            
            self._convs = ResEncoder(channel_in=4, ch=64, z=512)
        elif "RESNET" in model_config['custom_model_config']['backbone']:
            self._convs = TResEncoder(channel_in=4, ch=64, z=512)
        elif 'DUAL' in model_config['custom_model_config']['backbone']:
            self._convs = Encoder(channel_in=4, ch=32, z=512)
        elif '4STACK_CONT' in model_config['custom_model_config']['backbone']:
            self._convs = TEncoder(channel_in=4, ch=32, z=512)
        elif '4STACK_VAE' in model_config['custom_model_config']['backbone']:
            self._convs = VAE(channel_in=4, ch=32, z=512)


        #if 'e2e' not in model_config['custom_model_config']['backbone'] and model_config['custom_model_config']['backbone'] != 'random':
        #if 'e2e' not in model_config['custom_model_config']['backbone']:
            #if model_config['custom_model_config']['backbone'] == 'random':
            #    self._convs = VAE(channel_in=observation_space.shape[-1], z=512)

            
        #else:
            #in the case of e2e or random
        #    self._convs = backbone
        #    if model_config['custom_model_config']['temporal'] == 'lstm' or model_config['custom_model_config']['temporal'] == 'attention':
        #        self._convs = LSTMVAE(channel_in=observation_space.shape[-1], z=512)
        #    else:
        #        self._convs = VAE(channel_in=observation_space.shape[-1], z=512)
        #else:
        #    raise NotImplementedError("lol")
        
        
        if "e2e" not in model_config['custom_model_config']['backbone_path'] and "random" not in model_config['custom_model_config']['backbone_path']:
            print(model_config['custom_model_config']['backbone_path'])
            print("loading model weights")
            checkpoint = torch.load(model_config['custom_model_config']['backbone_path'], map_location="cpu")
            self._convs.load_state_dict(checkpoint['model_state_dict'])
        
        if not model_config['custom_model_config']['train_backbone']:
            print("freezing encoder layers")
            #freeze the entire backbone
            self._convs.eval()
            for param in self._convs.parameters():
                param.requires_grad = False
            #unfreeze the adapter
            #self._convs.conv_mu.train()
            #for param in self._convs.conv_mu.parameters():
            #    param.requires_grad = True

        #embed()
        #self.trainable_variables(True)

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





