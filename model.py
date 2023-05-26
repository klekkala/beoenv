
#This file contains 2 models singletask, multitask


import functools
from typing import Optional

import gymnasium as gym
import numpy as np
import tree
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple

from ray.rllib.core.models.base import Encoder
from ray.rllib.core.models.base import ModelConfig
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.core.models.configs import (
    MLPEncoderConfig,
    RecurrentEncoderConfig,
)
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models.distributions import Distribution
from ray.rllib.models.utils import get_filter_config
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.models.torch.model import TorchModel
from ray.rllib.core.models.configs import CNNEncoderConfig

from ray.rllib.models.base_model import RecurrentModel, Model, ModelIO
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog

from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.torch.primitives import TorchMLP
from customtorch import ModTorchCNN
from ray.rllib.core.models.specs.specs_base import Spec
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.models.specs.specs_base import TensorSpec
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.framework import try_import_torch



# The global, shared layer to be used by both models.
#TORCH_GLOBAL_SHARED_BACKBONE= SlimFC(
#    64,
#    64,
#    activation_fn=nn.ReLU,
#    initializer=torch.nn.init.xavier_uniform_,
#)

TORCH_GLOBAL_SHARED_POLICY = SlimFC(
    64,
    64,
    activation_fn=nn.ReLU,
    initializer=torch.nn.init.xavier_uniform_,
)



#this is class is reused for every game/city/town
#this is equivalent to a spec in rl_module api
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




        # Non-shared initial layer.
        # this is the adapter
        self.adapter = SlimFC(
            int(np.product(observation_space.shape)),
            64,
            activation_fn=nn.ReLU,
            initializer=torch.nn.init.xavier_uniform_,
        )


        if args.pol:
            self.pi = TORCH_GLOBAL_SHARED_LAYER
        else:
            self.pi = SlimFC(
                        64,
                        64,
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
        out = self.first_layer(input_dict["obs"])
        self._output = self._global_shared_layer(out)
        model_out = self.last_layer(self._output)
        return model_out, []

    @override(ModelV2)
    def value_function(self):
        assert self._output is not None, "must call forward first!"
        return torch.reshape(self.vf(self._output), [-1])




#this is class is reused for every game/city/town
#this is equivalent to a spec in rl_module api
class TorchSingleWeightsModel(TorchModelV2, nn.Module):
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

        self.backbone = VAE(3)
        if args.prtr:
            self.backbone.load_weights(map_fun(args.prtr))
            self.backbone.eval()
            for all params:
                param.set_param = False


        # Non-shared initial layer.
        # this is the adapter
        self.adapter = SlimFC(
            int(np.product(observation_space.shape)),
            64,
            activation_fn=nn.ReLU,
            initializer=torch.nn.init.xavier_uniform_,
        )


        if args.pol:
            self.pi = TORCH_GLOBAL_SHARED_LAYER
        else:
            self.pi = SlimFC(
                        64,
                        64,
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
        out = self.first_layer(input_dict["obs"])
        self._output = self._global_shared_layer(out)
        model_out = self.last_layer(self._output)
        return model_out, []

    @override(ModelV2)
    def value_function(self):
        assert self._output is not None, "must call forward first!"
        return torch.reshape(self.vf(self._output), [-1])






