import numpy as np
from typing import Dict, List
#import gymnasium as gym
import gym
from models.ResnetX import VAE as VAE
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import torchvision.transforms as transforms




torch, nn = try_import_torch()


class ImagenetNetwork(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):



        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)



        self._logits = None
        weights = ResNet18_Weights.IMAGENET1K_V1
        self._resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self._preprocess = weights.transforms()

        self._resnet.eval()


        layers=[]

        in_size = 1000

        layers.append(
            SlimFC(
                in_size=in_size,
                out_size=num_outputs,
                activation_fn=None,
                initializer=normc_initializer(1.0),
            )
        )

        self._logits = layers.pop()

        self._value_branch = SlimFC(
            in_size, 1, initializer=normc_initializer(0.01), activation_fn=None
        )


        for name, param in self._resnet.named_parameters():
            param.requires_grad = False


        self._features = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float()
        self._features = self._features.permute(0, 3, 1, 2)
        res_out = self._preprocess(self._features)
        res_out = self._resnet(res_out)
        self._features = res_out
        res_out = self._logits(res_out)
        return res_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return self._value_branch(self._features).squeeze(1)


class PreTrainedTorchResModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = TorchPreTrainedRes(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        # input_dict["obs"]["obs"] = input_dict["obs"]["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])