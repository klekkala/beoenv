
#This file contains 2 models singletask, multitask


import functools
from typing import Optional

import numpy as np
import tree
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models.torch.model import TorchModel
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.base_model import RecurrentModel, Model, ModelIO
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC

from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from vaemodel import Encoder
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork

torch, nn = try_import_torch()


# The global, shared layer to be used by both models.
# this model outputs a 512 latent dimension

BEOGYM_GLOBAL_SHARED_BACKBONE= Encoder(channels=5, ch=32, z=512)
#if using lstm this could be used:
#TORCH_GLOBAL_SHARED_BACKBONE= VAE(channel_in=1, ch=32, z=512)

BEOGYM_GLOBAL_SHARED_POLICY = SlimFC(
    64,
    5,
    activation_fn=nn.ReLU,
    initializer=torch.nn.init.xavier_uniform_,
)

#this is class is used when we are working with a single game
class SingleBeogymModel(TorchModelV2, nn.Module):


    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)




class SharedBackboneAtariModel(ComplexInputNetwork):
        def __init__(self, observation_space, action_space, num_outputs, model_config, name):
            super().__init__(observation_space, action_space, num_outputs, model_config, name)
