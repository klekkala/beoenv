
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
from vaemodel import SmallVAE as VAE

torch, nn = try_import_torch()


# The global, shared layer to be used by both models.
# this model outputs a 512 latent dimension
TORCH_GLOBAL_SHARED_BACKBONE= VAE(channel_in=4, ch=32, z=512)

#if using lstm this could be used:
#TORCH_GLOBAL_SHARED_BACKBONE= VAE(channel_in=1, ch=32, z=512)

TORCH_GLOBAL_SHARED_POLICY = SlimFC(
    64,
    18,
    activation_fn=nn.ReLU,
    initializer=torch.nn.init.xavier_uniform_,
)

#this is class is used when we are working with a single game
class SingleAtariModel(TorchModelV2, nn.Module):


    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)


        self.backbone = VAE(channel_in=4, ch=32, z=512)

        # this is the adapter
        self.adapter = SlimFC(
            512,
            64,
            activation_fn=nn.ReLU,
            initializer=torch.nn.init.xavier_uniform_,
        )

        self.pi = SlimFC(
                    64,
                    18,
                    activation_fn=nn.ReLU,
                    initializer=torch.nn.init.xavier_uniform_)

        #value function is always non shared
        self.vf = SlimFC(
            64,
            1,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_)
        #print(self.backbone, self.adapter, self.pi, self.vf)

        self._output = None


    def forward(self, input_dict, state, seq_lens):

        out = self.backbone(input_dict["obs"])
        self._output = self.adapter(out)
        model_out = self.pi(self._output)
        return model_out, []

    def value_function(self):
        assert self._output is not None, "must call forward first!"
        return torch.reshape(self.vf(self._output), [-1])


#this is class is reused for every game/city/town
#this is equivalent to a spec in rl_module api
class SharedBackboneAtariModel(SingleAtariModel):

    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        super().__init__(observation_space, action_space, num_outputs, model_config, name)
        self.backbone = TORCH_GLOBAL_SHARED_BACKBONE


#this is class is reused for every game/city/town
#this is equivalent to a spec in rl_module api
class SharedBackbonePolicyAtariModel(SingleAtariModel):

    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        super().__init__(observation_space, action_space, num_outputs, model_config, name)
        self.backbone = TORCH_GLOBAL_SHARED_BACKBONE
        self.pi = TORCH_GLOBAL_SHARED_POLICY
