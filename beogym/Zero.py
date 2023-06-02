
import numpy as np
from typing import Dict, List
#import gymnasium as gym
import gym
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

torch, nn = try_import_torch()


class ZeroNetwork(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.num_outputs = int(np.product(self.obs_space.shape))
        self._last_batch_size = None
        self._features = None
        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [64])
        post_fcnet_hiddens = [64]
        self.pfh=post_fcnet_hiddens
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation"), framework="torch"
        )

        vf_share_layers = self.model_config.get("vf_share_layers")

        self.last_layer_is_flattened = False
        self._logits = None
        self._debug=0
        layers = []
        # (w, h, in_channels) = obs_space.shape
        (w, h, in_channels) = (208,416,5)

        in_size = [w, h]
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation,
                )
            )
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]

        layers.append(
            SlimConv2d(
                in_channels,
                out_channels,
                kernel,
                stride,
                None,  # padding=valid
                activation_fn=activation,
            )
        )

        num_outputs=5
        post_fcnet_hiddens=[64]
        
        if num_outputs and post_fcnet_hiddens:
            layers.append(nn.Flatten())
            
            self._convs = nn.Sequential(*layers)
            layers=[]
            in_size = out_channels+6
            # Add (optional) post-fc-stack after last Conv2D layer.
            for i, out_size in enumerate(post_fcnet_hiddens + [num_outputs]):
                layers.append(
                    SlimFC(
                        in_size=in_size,
                        out_size=out_size,
                        activation_fn="relu"
                        if i < len(post_fcnet_hiddens) - 1
                        else None,
                        initializer=normc_initializer(1.0),
                    )
                )
                in_size = out_size
            # Last layer is logits layer.
            self._logits = layers.pop()
            self._fcnet= nn.Sequential(*layers)
            # self._logits = nn.Sequential(*layers)
        else:
            raise ValueError(
                "Please set post_fcnet_hiddens"
            )


        #if self.num_outputs is None:
            # Create a B=1 dummy sample and push it through out conv-net.
        #    dummy_in = (
        #        torch.from_numpy(self.obs_space.sample())
        #        .permute(2, 0, 1)
        #        .unsqueeze(0)
        #        .float()
        #    )
        #    dummy_out = self._convs(dummy_in)
        #    self.num_outputs = dummy_out.shape[1]

        # Build the value layers
        self._value_branch_separate = self._value_branch = None
        vf_share_layers=True
        if vf_share_layers:
            self._value_branch = SlimFC(
                post_fcnet_hiddens[-1], 1, initializer=normc_initializer(0.01), activation_fn=None
            )
        # for name, param in self._convs.named_parameters():
        #     param.requires_grad = False

    
    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"]["obs"].float()
        # Permuate b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 3, 1, 2)
        conv_out = self._convs(self._features)
        #conv_out = nn.functional.normalize(conv_out, dim = 0)
        min_value = torch.min(conv_out)
        max_value = torch.max(conv_out)
        if max_value!=min_value:
            conv_out = (conv_out - min_value) / (max_value - min_value)
        #conv_out = (conv_out - torch.min(conv_out)) / (torch.max(conv_out) - torch.min(conv_out))
        # Store features to save forward pass when getting value_function out.
        self._features = conv_out
        self._aux=input_dict["obs"]["aux"].float()
        conv_out = torch.cat((conv_out, self._aux), dim=-1)
        conv_out = self._fcnet(conv_out)
        self._features = conv_out
        conv_out = self._logits(conv_out)
        #print(conv_out.shape)
        #logits = conv_out.squeeze(3)
        #logits = logits.squeeze(2)
        logits=conv_out
        return logits, state

    # @profile(precision=5)
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        features = self._features
        return self._value_branch(features).squeeze(1)

    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._convs(obs.permute(0, 3, 1, 2))  # switch to channel-major
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res


