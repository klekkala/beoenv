
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
from PIL import Image
from IPython import embed
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from vaemodel import Encoder
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.torch_utils import one_hot
from atari_vae import Encoder, TEncoder, TBeoEncoder
from typing import Dict, List, Tuple
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.policy.rnn_sequencing import add_time_dimension
import time
from ray.rllib.models.torch.visionnet import VisionNetwork
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import (
    normc_initializer as torch_normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)

from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN

torch, nn = try_import_torch()

# The global, shared layer to be used by both models.
# this model outputs a 512 latent dimension

#BEOGYM_GLOBAL_SHARED_BACKBONE= Encoder(channels=5, ch=32, z=512)
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
        embed()

class FrozenBackboneModel(ComplexInputNetwork):
    def _init_(self, observation_space, action_space, num_outputs, model_config, name):
        super()._init_(observation_space, action_space, num_outputs, model_config, name)
        print('frozen')
        self.cnns[1].eval()
        for param in self.cnns[1].parameters():
            param.requires_grad = False
        for name, param in self.cnns[1].named_parameters():
            param.requires_grad = False


    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(
                input_dict[SampleBatch.OBS], self.processed_obs_space, tensorlib="torch"
            )

        outs = []
        for i, component in enumerate(tree.flatten(orig_obs)):
            if i in self.cnns:
                cnn_out, _ = self.cnns[i](SampleBatch({SampleBatch.OBS: component}))
                outs.append(cnn_out)

            elif i in self.one_hot:
                if component.dtype in [
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                ]:
                    one_hot_in = {
                        SampleBatch.OBS: one_hot(
                            component, self.flattened_input_space[i]
                        )
                    }
                else:
                    one_hot_in = {SampleBatch.OBS: component}
                one_hot_out, _ = self.one_hot[i](SampleBatch(one_hot_in))
                outs.append(one_hot_out)
            else:
                nn_out, _ = self.flatten[i](
                    SampleBatch(
                        {
                            SampleBatch.OBS: torch.reshape(
                                component, [-1, self.flatten_dims[i]]
                            )
                        }
                    )
                )

                outs.append(nn_out)

        # Concat all outputs and the non-image inputs.
        out = torch.cat(outs, dim=1)
        # Push through (optional) FC-stack (this may be an empty stack).
        out, _ = self.post_fc_stack(SampleBatch({SampleBatch.OBS: out}))

        # No logits/value branches.
        if self.logits_layer is None:
            return out, []

        # Logits- and value branches.
        logits, values = self.logits_layer(out), self.value_layer(out)
        self._value_out = torch.reshape(values, [-1])
        return logits, []

#class SharedBackboneAtariModel(ComplexInputNetwork):
#        def __init__(self, observation_space, action_space, num_outputs, model_config, name):
#            super().__init__(observation_space, action_space, num_outputs, model_config, name)




class ComplexNet(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        self.original_space = (
            obs_space.original_space
            if hasattr(obs_space, "original_space")
            else obs_space
        )

        self.processed_obs_space = (
            self.original_space
            if model_config.get("_disable_preprocessor_api")
            else obs_space
        )

        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, self.original_space, action_space, num_outputs, model_config, name
        )

        self.flattened_input_space = flatten_space(self.original_space)

        # Atari type CNNs or IMPALA type CNNs (with residual layers)?
        # self.cnn_type = self.model_config["custom_model_config"].get(
        #     "conv_type", "atari")

        # Build the CNN(s) given obs_space's image components.
        self.cnns = {}
        self.one_hot = {}
        self.flatten_dims = {}
        self.flatten = {}
        concat_size = 0
        for i, component in enumerate(self.flattened_input_space):
            # Image space.
            if len(component.shape) == 3:

                config = {
                    "conv_filters": model_config["conv_filters"]
                    if "conv_filters" in model_config
                    else get_filter_config(component.shape),
                    "conv_activation": model_config.get("conv_activation"),
                    "post_fcnet_hiddens": [],
                    "vf_share_layers": True,
                    "conv_filters": [[16, [8, 8], 4], [32, [4, 4], 2], [512, [11, 11], 1]],
                }

                # if self.cnn_type == "atari":
                self.cnns[i] = ModelCatalog.get_model_v2(
                    component,
                    action_space,
                    num_outputs=None,
                    model_config=config,
                    framework="torch",
                    name="cnn_{}".format(i),
                )

                concat_size += self.cnns[i].num_outputs
                self.add_module("cnn_{}".format(i), self.cnns[i])
            # Discrete|MultiDiscrete inputs -> One-hot encode.
            elif isinstance(component, (Discrete, MultiDiscrete)):
                continue
            # Everything else (1D Box).
            else:
                concat_size+=component.shape[0]
                continue
        
        # Optional post-concat FC-stack.
        # post_fc_stack_config = {
        #     "fcnet_hiddens": model_config.get("post_fcnet_hiddens", []),
        #     "fcnet_activation": model_config.get("post_fcnet_activation", "relu"),
        # }
        # self.post_fc_stack = ModelCatalog.get_model_v2(
        #     Box(float("-inf"), float("inf"), shape=(concat_size,), dtype=np.float32),
        #     self.action_space,
        #     None,
        #     post_fc_stack_config,
        #     framework="torch",
        #     name="post_fc_stack",
        # )

        # Actions and value heads.
        self.logits_layer = None
        self.value_layer = None
        self._value_out = None

        if num_outputs:
            # Action-distribution head.
            self.final = SlimFC(
                in_size=concat_size,
                out_size=512,
                activation_fn=None,
                initializer=torch_normc_initializer(0.01),
            )
            self.logits_layer = SlimFC(
                in_size=512,
                out_size=num_outputs,
                activation_fn=None,
                initializer=torch_normc_initializer(0.01),
            )
            # Create the value branch model.
            self.value_layer = SlimFC(
                in_size=512,
                out_size=1,
                activation_fn=None,
                initializer=torch_normc_initializer(0.01),
            )
        else:
            self.num_outputs = concat_size

        if not model_config['custom_model_config']['train_backbone']:
            print("freezing encoder layers")
                #freeze the entire backbone
            self.cnns[1].eval()
            for param in self.cnns[1].parameters():
                param.requires_grad = False
            for name, param in self.cnns[1].named_parameters():
                param.requires_grad = False
            self.final.eval()
            for param in self.final.parameters():
                param.requires_grad = False
        else:
            print('not freezing')
        
        self.encoder = TBeoEncoder(channel_in=3, ch=32, z=512)

        if "e2e" not in model_config['custom_model_config']['backbone_path'] and "random" not in model_config['custom_model_config']['backbone_path']:
            print(model_config['custom_model_config']['backbone_path'])
            print("loading model weights")
            checkpoint = torch.load(model_config['custom_model_config']['backbone_path'], map_location="cpu")
            self.encoder.load_state_dict(checkpoint['model_state_dict'])

        if not model_config['custom_model_config']['train_backbone']:
            print("freezing encoder layers")
                #freeze the entire backbone
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
        else:
            print('not freezing')

        #embed()

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        if self.encoder is None:
            if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
                orig_obs = input_dict[SampleBatch.OBS]
            else:
                orig_obs = restore_original_dimensions(
                    input_dict[SampleBatch.OBS], self.processed_obs_space, tensorlib="torch"
                )
            # Push observations through the different components
            # (CNNs, one-hot + FC, etc..).
            outs = []
            for i, component in enumerate(tree.flatten(orig_obs)):
                if i in self.cnns:
                    cnn_out, _ = self.cnns[i](SampleBatch({SampleBatch.OBS: component}))
                    outs.append(cnn_out)
                elif i in self.one_hot:
                    if component.dtype in [
                        torch.int8,
                        torch.int16,
                        torch.int32,
                        torch.int64,
                        torch.uint8,
                    ]:
                        one_hot_in = {
                            SampleBatch.OBS: one_hot(
                                component, self.flattened_input_space[i]
                            )
                        }
                    else:
                        one_hot_in = {SampleBatch.OBS: component}
                    one_hot_out, _ = self.one_hot[i](SampleBatch(one_hot_in))
                    outs.append(one_hot_out)
                else:
                    nn_out = component
                    # nn_out, _ = self.flatten[i](
                    #     SampleBatch(
                    #         {
                    #             SampleBatch.OBS: torch.reshape(
                    #                 component, [-1, self.flatten_dims[i]]
                    #             )
                    #         }
                    #     )
                    # )
                    outs.append(nn_out)

            # Concat all outputs and the non-image inputs.
            
            out = torch.cat(outs, dim=1)
            # Push through (optional) FC-stack (this may be an empty stack).
            out = self.final(out)
        else:
            out = self.encoder(input_dict['obs']['obs'], input_dict['obs']['aux'])

        # No logits/value branches.
        if self.logits_layer is None:
            return out, []

        # Logits- and value branches.
        logits, values = self.logits_layer(out), self.value_layer(out)
        self._value_out = torch.reshape(values, [-1])
        return logits, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out






class BeogymCNNV2PlusRNNModel(TorchRNN, nn.Module):
    """A conv. + recurrent torch net example using a pre-trained MobileNet."""

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name
    ):

        TorchRNN.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.lstm_state_size = 514 # originally 16
        self.cnn_shape = [3, 84, 84]
        self.visual_size_in = self.cnn_shape[0] * self.cnn_shape[1] * self.cnn_shape[2]
        # MobileNetV2 has a flat output of (1000,).
        self.visual_size_out = 514


        # Load the MobileNetV2 from torch.hub.
        if "RESNET" in model_config['custom_model_config']['backbone'] and "DUAL" in model_config['custom_model_config']['backbone']:            
            self._convs = Encoder(channel_in=3, ch=64, z=512)
        elif "RESNET" in model_config['custom_model_config']['backbone']:
            self._convs = TEncoder(channel_in=3, ch=64, z=512)
        elif 'DUAL' in model_config['custom_model_config']['backbone']:
            self._convs = Encoder(channel_in=3, ch=32, z=512)
        else:
            #self._convs = TEncoder(channel_in=1, ch=32, z=512)
            self._convs = TEncoder(channel_in=3, ch=32, z=512, activation="elu")

        
        print(self._convs)
        self.lstm = nn.LSTM(
            self.visual_size_out, self.lstm_state_size, batch_first=True
        )

        # Postprocess LSTM output with another hidden layer and compute values.
        self.logits = SlimFC(self.lstm_state_size, self.num_outputs)
        self.value_branch = SlimFC(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

        if "e2e" not in model_config['custom_model_config']['backbone_path'] and "random" not in model_config['custom_model_config']['backbone_path']:
            print(model_config['custom_model_config']['backbone_path'])
            print("loading model weights")
            checkpoint = torch.load(model_config['custom_model_config']['backbone_path'], map_location="cpu")
            
            lstm_ckpt = {}
            convs_ckpt = {}
            #for eachkey in checkpoint['model_state_dict']:
                #if 'lstm' in eachkey:
                #    newkey = eachkey.replace('lstm.', '')
                #    lstm_ckpt[newkey] = checkpoint['model_state_dict'][eachkey]
                #else:
                #    if 'conv_mu' in eachkey:
                #        newkey = eachkey.replace('encoder.', '')
                #    else:
                #        newkey = eachkey.replace('encoder.encoder', 'encoder')
                #convs_ckpt[newkey] = checkpoint['model_state_dict'][eachkey]


            #for each in self._convs.named_parameters():
            #    print(each[0])

            #create cnn_modstdict
            #self._convs.load_state_dict(convs_ckpt)

            #create lstm_modstdict
            #self.lstm.load_state_dict(lstm_ckpt)

            self._convs.load_state_dict(checkpoint['model_state_dict'])

        if not model_config['custom_model_config']['train_backbone']:
            print("freezing encoder layers")
            #freeze the entire backbone
            self._convs.eval()
            for param in self._convs.parameters():
                param.requires_grad = False

            #self.lstm.eval()
            #for param in self.lstm.parameters():
            #    param.requires_grad = False

        
    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        # Create image dims.
        goal = torch.reshape(inputs[:, :, 21168:], (-1, 2))
        vision_in = torch.reshape(inputs[:, :, :21168], (-1, 84, 84, 3))
        vision_in = vision_in.permute(0, 3, 1, 2)

        vision_in = torch.reshape(vision_in, [-1] + self.cnn_shape)
        
        # Flatten.
        vision_out = torch.flatten(self._convs(vision_in), start_dim=1)
        vision_out = torch.cat((vision_out, goal), -1)

        vision_out_time_ranked = torch.reshape(
            vision_out, [inputs.shape[0], inputs.shape[1], vision_out.shape[-1]]
        )

        if len(state[0].shape) == 2:
            state[0] = state[0].unsqueeze(0)
            state[1] = state[1].unsqueeze(0)
        # Forward through LSTM.
        self._features, [h, c] = self.lstm(vision_out_time_ranked, state)
        # Forward LSTM out through logits layer and value layer.
        logits = self.logits(self._features)
        return logits, [h.squeeze(0), c.squeeze(0)]

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [
            list(self._convs.modules())[-1]
            .weight.new(1, self.lstm_state_size)
            .zero_()
            .squeeze(0),
            list(self._convs.modules())[-1]
            .weight.new(1, self.lstm_state_size)
            .zero_()
            .squeeze(0),
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])
    


### BELOW CLASS IS DEPRICATED!!!!!!!!!!!

class LSTM2Network(TorchRNN, nn.Module):
    """A conv. + recurrent torch net example using a pre-trained MobileNet."""

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name
    ):

        TorchRNN.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.lstm_state_size = 256
        self.cnn_shape = [84,84,3]
        # self.cnn_shape = list(cnn_shape)
        self.visual_size_in = self.cnn_shape[0] * self.cnn_shape[1] * self.cnn_shape[2]
        # MobileNetV2 has a flat output of (1000,).
        self.visual_size_out = 512

        layers = []
        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"
        filters = [[16, 3, 2], [32, 3, 2], [64, 3, 2], [128, 3, 2], [256, 3, 2], [512,3,1]]

        (w, h, in_channels) = (84,84,3)
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
                    activation_fn='relu',
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
                activation_fn='relu',
            )
        )

        self.cnn_model = nn.Sequential(*layers)



        self.lstm = nn.LSTM(
            self.visual_size_out+2, self.lstm_state_size, batch_first=True
        )


        self.final_lstm = nn.LSTM(
            258, self.lstm_state_size, batch_first=True
        )


        self.logits = SlimFC(
            in_size=self.lstm_state_size,
            out_size=self.num_outputs,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_,
        )
        self.value_branch = SlimFC(
            in_size=self.lstm_state_size,
            out_size=1,
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_,
        )


        # Holds the current "base" output (before logits layer).
        self._features = None
        self.view_requirements[SampleBatch.PREV_ACTIONS] = ViewRequirement(
                SampleBatch.ACTIONS, space=self.action_space, shift=-1
            )
        self.view_requirements[SampleBatch.PREV_REWARDS] = ViewRequirement(
                SampleBatch.REWARDS, shift=-1
            )

    @override(ModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
    
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""
        flat_inputs = torch.cat((input_dict["obs"]['obs'].view(input_dict["obs"]['obs'].shape[0], -1), input_dict["obs"]['aux'].view(input_dict["obs"]['aux'].shape[0], -1)), dim=1).float()
        # flat_inputs = input_dict["obs"]['obs'].view(input_dict["obs"]['obs'].shape[0], -1).float()
        # Note that max_seq_len != input_dict.max_seq_len != seq_lens.max()
        # as input_dict may have extra zero-padding beyond seq_lens.max().
        # Use add_time_dimension to handle this
        assert seq_lens is not None
        rew=torch.reshape(input_dict[SampleBatch.PREV_REWARDS].float(), [-1, 1])
        act=torch.reshape(input_dict[SampleBatch.PREV_ACTIONS].float(), [-1, 1])
        aux=torch.cat((rew,act), dim=1)
        flat_inputs = torch.cat((flat_inputs,aux), dim=1)
        self.time_major = self.model_config.get("_time_major", False)
        inputs = add_time_dimension(
            flat_inputs,
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.time_major,
        )
        output, new_state = self.forward_rnn(inputs, state, seq_lens,aux)
        output = torch.reshape(output, [-1, self.num_outputs])
        # print(input_dict['prev_rewards'])
        # print(input_dict[SampleBatch.PREV_REWARDS].unsqueeze(dim=1).shape)
        return output, new_state



    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens,aux):
        # Create image dims.
        info_in = inputs[:, :,84*84*3:84*84*3+2]
        ar_in = inputs[:, :,84*84*3+2:]
        vision_in = inputs[:, :, :84*84*3]
        vision_in = torch.reshape(vision_in, [-1] + self.cnn_shape)
        vision_in = vision_in.permute(0, 3, 1, 2)
        vision_out = self.cnn_model(vision_in)
        # Flatten.
        vision_out = vision_out.squeeze(2)
        vision_out = vision_out.squeeze(2)
        vision_out_time_ranked = torch.reshape(
            vision_out, [inputs.shape[0], inputs.shape[1], vision_out.shape[-1]]
        )
        if len(state[0].shape) == 2:
            state[0] = state[0].unsqueeze(0)
            state[1] = state[1].unsqueeze(0)
            state[2] = state[2].unsqueeze(0)
            state[3] = state[3].unsqueeze(0)
        # Forward through LSTM.
        vision_out_time_ranked = torch.cat((vision_out_time_ranked,info_in), dim=2)
        self._features, [h, c] = self.lstm(vision_out_time_ranked, [state[0], state[1]])
        # Forward LSTM out through logits layer and value layer.
        self._features = torch.cat((self._features,ar_in), dim=2)
        self._features, [fh, fc] = self.final_lstm(self._features, [state[2], state[3]])
        logits = self.logits(self._features)
        return logits, [h.squeeze(0), c.squeeze(0),fh.squeeze(0), fc.squeeze(0)]

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        linear = next(self.logits._model.children())
        h = [
            linear.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            linear.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            linear.weight.new(1, self.lstm_state_size).zero_().squeeze(0), 
            linear.weight.new(1, self.lstm_state_size).zero_().squeeze(0)
        ]
        return h


    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])


class SingleImageModel(VisionNetwork):

    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

    
        if  model_config['custom_model_config']['backbone'] == 'clip':
            #put the clip code here
            #self._convs = clipmodel
            self._model = 'clip'
            self._convs, self._pre = clip.load('RN50',device = 'cuda')
            print('its clip')
        
        else:
            self._model = 'not'
            chan_in = 3
            activation = 'relu'
        
            self._convs = TEncoder(channel_in=chan_in, ch=32, z=512, activation=activation)

            #embed()
            #model_config['custom_model_config']['train_backbone']=True
            
            if not model_config['custom_model_config']['train_backbone']:
                print("freezing encoder layers")
                #freeze the entire backbone
                self._convs.eval()
                for param in self._convs.parameters():
                    param.requires_grad = False
            else:
                print('not freezing')

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float()
        # Permuate b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 3, 1, 2)
        if self._model == 'clip':
            # for i in range(self._features.size(0)):
            #     print(self._pre(Image.fromarray(np.zeros((84,84,3)))))
            #     self._features[i] = self._pre(Image.fromarray(self._features[i].cpu().numpy())).to('cuda')
            # self._features = self._pre(self._features.cpu().numpy())
            print(self._features)
            conv_out = self._convs.encode_image(self._features)
        else:
            conv_out = self._convs(self._features)
        # Store features to save forward pass when getting value_function out.
        if not self._value_branch_separate:
            self._features = conv_out

        if not self.last_layer_is_flattened:
            if self._logits:
                conv_out = self._logits(conv_out)
            if len(conv_out.shape) == 4:
                if conv_out.shape[2] != 1 or conv_out.shape[3] != 1:
                    raise ValueError(
                        "Given `conv_filters` ({}) do not result in a [B, {} "
                        "(`num_outputs`), 1, 1] shape (but in {})! Please "
                        "adjust your Conv2D stack such that the last 2 dims "
                        "are both 1.".format(
                            self.model_config["conv_filters"],
                            self.num_outputs,
                            list(conv_out.shape),
                        )
                    )
                logits = conv_out.squeeze(3)
                logits = logits.squeeze(2)
            else:
                logits = conv_out
            return logits, state
        else:
            return conv_out, state
