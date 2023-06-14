
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
from ray.rllib.core.models.torch.primitives import TorchCNN
from ray.rllib.core.models.specs.specs_base import Spec
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.models.specs.specs_base import TensorSpec
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

from RES_VAE import VAE
#from shellrl.prtrencoder.atari_vae import VAE
from ray.rllib.core.models.base import ENCODER_OUT

"""
class ModTorchMLPEncoder(TorchModel, Encoder):
    def __init__(self, config: MLPEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        # Create the neural network.
        self.net = TorchMLP(
            input_dim=config.input_dims[0],
            hidden_layer_dims=config.hidden_layer_dims,
            hidden_layer_activation=config.hidden_layer_activation,
            hidden_layer_use_layernorm=config.hidden_layer_use_layernorm,
            output_dim=config.output_dims[0],
            output_activation=config.output_activation,
            use_bias=config.use_bias,
        )

    @override(Model)
    def get_input_specs(self) -> Optional[Spec]:
        return SpecDict(
            {
                SampleBatch.OBS: TensorSpec(
                    "b, d", d=self.config.input_dims[0], framework="torch"
                ),
            }
        )

    @override(Model)
    def get_output_specs(self) -> Optional[Spec]:
        return SpecDict(
            {
                ENCODER_OUT: TensorSpec(
                    "b, d", d=self.config.output_dims[0], framework="torch"
                ),
            }
        )

    @override(Model)
    def _forward(self, inputs: dict, **kwargs) -> dict:
        return {ENCODER_OUT: self.net(inputs[SampleBatch.OBS])}
"""

class ModTorchCNNEncoder(TorchModel, Encoder):
    def __init__(self, config: CNNEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        layers = []
        # The bare-bones CNN (no flatten, no succeeding dense).
        #print(config.cnn_activation, config.cnn_use_layernorm, config.use_bias)
   
        cnn = TorchCNN(
            input_dims=config.input_dims,
            cnn_filter_specifiers=config.cnn_filter_specifiers,
            cnn_activation=config.cnn_activation,
            cnn_use_layernorm=config.cnn_use_layernorm,
            use_bias=config.use_bias,
        )

        #cnn = VAE()
        #cnn = VAE(channel_in=3, ch=32, z=512)
        #cnn = VAE(channel_in=3, ch=32, z=512)
        #print(self.VAE)
        #checkpoint = torch.load("/lab/kiran/shellrl/prtrencoder/Models/CONV_ATTARI_84.pt", map_location='cpu')
        #checkpoint = torch.load("/lab/kiran/ckpts/pretrained/atari/STL10_ATTARI_64.pt", map_location='cpu')
        #cnn.load_state_dict(checkpoint['model_state_dict'])
        #cnn.eval()
        #for param in cnn.parameters():
        #    param.requires_grad = False

        layers.append(cnn)

        # Add a flatten operation to move from 2/3D into 1D space.
        layers.append(nn.Flatten())

        # Add a final linear layer to make sure that the outputs have the correct
        # dimensionality (output_dims).
        layers.append(
            nn.Linear(
                #int(cnn.output_width) * int(cnn.output_height) * int(cnn.output_depth),        
                30976,
                #4608,
                #64,
                #512,
                config.output_dims[0]
            )
        )
        output_activation = get_activation_fn(
            config.output_activation, framework="torch"
        )
        if output_activation is not None:
            layers.append(output_activation())

        # Create the network from gathered layers.
        self.net = nn.Sequential(*layers)


    def get_input_specs(self) -> Optional[Spec]:
        return SpecDict(
            {
                SampleBatch.OBS: TensorSpec(
                    "b, w, h, c",
                    w=self.config.input_dims[0],
                    h=self.config.input_dims[1],
                    c=self.config.input_dims[2],
                    framework="torch",
                ),
            }
        )

    def get_output_specs(self) -> Optional[Spec]:
        return SpecDict(
            {
                ENCODER_OUT: TensorSpec(
                    "b, d", d=self.config.output_dims[0], framework="torch"
                ),
            }
        )

    def _forward(self, inputs: dict, **kwargs) -> dict:

        #return {ENCODER_OUT: self.net(inputs[SampleBatch.OBS].type(torch.float32).permute(0, 3, 1, 2))}
        return {ENCODER_OUT: self.net(inputs[SampleBatch.OBS])}



class ModCNNEncoderConfig(CNNEncoderConfig, ModelConfig):
    """Configuration for a convolutional network."""
    def build(self, framework: str = "torch") -> Model:
        self._validate()

        if framework == "torch":
            #from ray.rllib.core.models.torch.encoder import TorchCNNEncoder
            #return TorchCNNEncoder(self)
            return ModTorchCNNEncoder(self)
            #return ModTorchMLPEncoder(self)

        elif framework == "tf2":
            from ray.rllib.core.models.tf.encoder import TfCNNEncoder

            return TfCNNEncoder(self)






# Define a simple catalog that returns our custom distribution when
# get_action_dist_cls is called
class CustomPPOCatalog(PPOCatalog):
    def get_encoder_config(
        cls,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
        view_requirements=None,
    ) -> ModelConfig:
        """Returns an EncoderConfig for the given input_space and model_config_dict.

        Encoders are usually used in RLModules to transform the input space into a
        latent space that is then fed to the heads. The returned EncoderConfig
        objects correspond to the built-in Encoder classes in RLlib.
        For example, for a simple 1D-Box input_space, RLlib offers an
        MLPEncoder, hence this method returns the MLPEncoderConfig. You can overwrite
        this method to produce specific EncoderConfigs for your custom Models.

        The following input spaces lead to the following configs:
        - 1D-Box: MLPEncoderConfig
        - 3D-Box: CNNEncoderConfig
        # TODO (Artur): Support more spaces here
        # ...

        Args:
            observation_space: The observation space to use.
            model_config_dict: The model config to use.
            action_space: The action space to use if actions are to be encoded. This
                is commonly the case for LSTM models.
            view_requirements: The view requirements to use if anything else than
                observation_space or action_space is to be encoded. This signifies an
                advanced use case.

        Returns:
            The encoder config.
        """
        # TODO (Artur): Make it so that we don't work with complete MODEL_DEFAULTS


        # input_space is a 3D Box
    
        if (
            isinstance(observation_space, Box) and len(observation_space.shape) == 3
        ):
            if not model_config_dict.get("conv_filters"):
                model_config_dict["conv_filters"] = get_filter_config(
                    observation_space.shape
                )

            encoder_latent_dim = model_config_dict["encoder_latent_dim"]
            encoder_config = ModCNNEncoderConfig(
                input_dims=observation_space.shape,
                cnn_filter_specifiers=model_config_dict["conv_filters"],
                cnn_activation=model_config_dict["conv_activation"],
                cnn_use_layernorm=model_config_dict.get(
                    "conv_use_layernorm", False
                ),
                output_dims=[encoder_latent_dim],
                # TODO (sven): Setting this to None here helps with the existing
                #  APPO Pong benchmark (actually, leaving this at default=tanh does
                #  NOT learn at all!).
                #  We need to remove the last Dense layer from CNNEncoder in general
                #  AND establish proper ModelConfig objects (instead of hacking
                #  everything with the old default model config dict).
                output_activation=None,
            )
        """
        if model_config_dict["encoder_latent_dim"]:
            hidden_layer_dims = model_config_dict["fcnet_hiddens"]
        else:
            hidden_layer_dims = model_config_dict["fcnet_hiddens"][:-1]
        encoder_config = MLPEncoderConfig(
            input_dims=[observation_space.shape[0]],
            hidden_layer_dims=hidden_layer_dims,
            hidden_layer_activation=activation,
            output_dims=[encoder_latent_dim],
            output_activation=output_activation,
        )
        """

        return encoder_config







