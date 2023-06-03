
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
TORCH_GLOBAL_SHARED_BACKBONE = VAE(channel_in=4, ch=32, z=512)


class TorchSharedWeightsModel(TorchModelV2, nn.Module):
    """Example of weight sharing between two different TorchModelV2s.

    The shared (single) layer is simply defined outside of the two Models,
    then used by both Models in their forward pass.
    """

    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        

        self._global_backbone = TORCH_GLOBAL_SHARED_BACKBONE
        #checkpoint = torch.load("/lab/kiran/shellrl/prtrencoder/Models/CONV_ATTARI_84.pt", map_location='cpu')
        #self._global_backbone.load_state_dict(checkpoint['model_state_dict'])
        #self._global_backbone.eval()
        #for param in self._global_backbone.parameters():
        #    param.requires_grad = False

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

        self._output = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):

        out = self._global_backbone(input_dict["obs"])
        self._output = self.adapter(out)
        model_out = self.pi(self._output)
        return model_out, []

    @override(ModelV2)
    def value_function(self):
        assert self._output is not None, "must call forward first!"
        return torch.reshape(self.vf(self._output), [-1])

