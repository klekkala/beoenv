
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


#this is class is used when we are working with a single game
class SingleAtariModel(VisionNetwork):


    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        super().__init__(observation_space, action_space, num_outputs, model_config, name)



    
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





