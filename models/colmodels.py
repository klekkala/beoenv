
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


class TColEncoder(nn.Module):
    def __init__(self, channel_in=3, ch=16, z=64, h_dim=512):
        super(TColEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.ZeroPad2d((2, 2, 2, 2)),
            nn.Conv2d(channel_in, ch, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(ch, ch*2, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*32, kernel_size=(11, 11), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_mu = nn.Conv2d(ch*32, z, 1, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.conv_mu(x)
        x = torch.flatten(x, start_dim=1)
        
        return x




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



class TClipEncoder(nn.Module):
    def __init__(self, z=64):
        super(TClipEncoder, self).__init__()
        import clip
        self.encoder, self.preprocess = clip.load("RN50")
        # self.joint_layer = nn.Linear(in_features=1024+2, out_features=z, bias=True)
        self.joint_layer = nn.Linear(in_features=1024, out_features=z, bias=True)

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        x = self.encoder.encode_image(x).to(torch.float32)

        #concat aux
        # x = torch.concat((x, aux), axis=1)
        x = self.joint_layer(x)
        x = torch.flatten(x, start_dim=1)
        return x

class ColoClipEncoder(nn.Module):
    def __init__(self, z=64):
        super(ColoClipEncoder, self).__init__()
        self.encoder, self.preprocess = clip.load("RN50")
        self.joint_layer = nn.Linear(in_features=1024, out_features=z, bias=True)

    def forward(self, x):
        #x = x.permute(0, 3, 1, 2)
        #x = self.preprocess(x)
        x = self.encoder.encode_image(x).to(torch.float32)

        #concat aux
        # x = torch.concat((x, aux), axis=1)
        x = self.joint_layer(x)
        x = torch.flatten(x, start_dim=1)
        return x


class ColoR3MEncoder(nn.Module):
    def __init__(self, z=64):
        from r3m import load_r3m
        super(ColoR3MEncoder, self).__init__()
        self.encoder = load_r3m("resnet50")
        self.joint_layer = nn.Linear(in_features=2048, out_features=z, bias=True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.joint_layer(x)
        x = torch.flatten(x, start_dim=1)
        return x


class ColoMVPEncoder(nn.Module):
    def __init__(self, z=64):
        super(ColoMVPEncoder, self).__init__()
        import mvp
        self.encoder = mvp.load("vits-mae-hoi")
        self.encoder.freeze()
        self.joint_layer = nn.Linear(in_features=386, out_features=z, bias=True)

    def forward(self, x):
        #concat aux
        x = self.encoder(x)
        x = self.joint_layer(x)
        x = torch.flatten(x, start_dim=1)
        return x

class MVPEncoder(nn.Module):
    def __init__(self, z=64):
        super(MVPEncoder, self).__init__()
        self.encoder = mvp.load("vits-mae-hoi")
        self.encoder.freeze()
        self.joint_layer = nn.Linear(in_features=386, out_features=z, bias=True)

    def forward(self, x, aux):
        #concat aux
        x = self.encoder(x * 255.0)
        x = torch.concat((x, aux), axis=1)
        x = self.joint_layer(x)
        x = torch.flatten(x, start_dim=1)
        return x


class VC1Encoder(nn.Module):
    def __init__(self, z=64):
        super(VC1Encoder, self).__init__()
        import vc_models
        from vc_models.models.vit import model_utils
        self.encoder,embd_size,model_transforms,model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)
        self.joint_layer = nn.Linear(in_features=770, out_features=z, bias=True)

    def forward(self, x, aux):
        #concat aux
        x = self.encoder(x)
        x = torch.concat((x, aux), axis=1)
        x = self.joint_layer(x)
        x = torch.flatten(x, start_dim=1)
        return x

class ColoVC1Encoder(nn.Module):
    def __init__(self, z=64):
        super(ColoVC1Encoder, self).__init__()
        import vc_models
        from vc_models.models.vit import model_utils
        self.encoder,embd_size,model_transforms,model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)
        self.joint_layer = nn.Linear(in_features=770, out_features=z, bias=True)

    def forward(self, x):
        #concat aux
        x = self.encoder(x)
        x = self.joint_layer(x)
        x = torch.flatten(x, start_dim=1)
        return x


class SingleColModel(VisionNetwork):

    def __init__(
        self, observation_space, action_space, num_outputs, model_config, name
    ):
        super().__init__(observation_space, action_space, num_outputs, model_config, name)

        #chan_in = 1 if model_config['custom_model_config']['temporal'] == 'notemp' or model_config['custom_model_config']['temporal'] == 'lstm' else 4

        #activation = "elu" if model_config['custom_model_config']['temporal'] == 'lstm' else 'relu'
        
        self._convs = TColEncoder(channel_in=3, ch=32, z=512, activation="relu")

        #checkpoint = torch.load('/lab/kiran/ckpts/pretrained/atari/3CHAN_VEP_COLL_3CHAN_OBJCOLOR_TABTEXT_STANDARD_2.0_0.01_2_nsame_triplet_32_2_0.0001_1.pt', map_location="cpu")
        #self._convs.load_state_dict(checkpoint['model_state_dict'])
        
        print("freezing encoder layers")
        #freeze the entire backbone
        self._convs.eval()
        for param in self._convs.encoder.parameters():
            param.requires_grad = False

class ComplexNet(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        print("******######################", action_space, num_outputs)
        self.original_space = (
            obs_space.original_space
            if hasattr(obs_space, "original_space")
            else obs_space
        )
        self.div = model_config['custom_model_config']['div']
        print("division..........", self.div)
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
                # self.cnns[i] = ModelCatalog.get_model_v2(
                #     component,
                #     action_space,
                #     num_outputs=None,
                #     model_config=config,
                #     framework="torch",
                #     name="cnn_{}".format(i),
                # )

                # concat_size += self.cnns[i].num_outputs
                # self.add_module("cnn_{}".format(i), self.cnns[i])
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
            # self.logits_layer = SlimFC(
            #     in_size=512,
            #     out_size=num_outputs,
            #     activation_fn=None,
            #     initializer=torch_normc_initializer(0.01),
            # )

            #2 layer
            self.fc1 = SlimFC(
                in_size=512,
                out_size=32,  # Intermediate layer size, you can adjust this
                activation_fn= 'tanh',  # Add an activation function like ReLU
                initializer=torch_normc_initializer(0.01),
            )

            self.logits_layer = SlimFC(
                in_size=32,  # Input size should match the output size of the previous layer
                out_size=num_outputs,
                activation_fn= None,  # No activation for the final output layer
                initializer=torch_normc_initializer(0.01),
            )
            print("****************num_outputs is***************", num_outputs)

            # Create the value branch model.
            self.value_layer = SlimFC(
                in_size=512,
                out_size=1,
                activation_fn=None,
                initializer=torch_normc_initializer(0.01),
            )
        else:
            self.num_outputs = concat_size

        #if not model_config['custom_model_config']['train_backbone']:
        #    print("freezing encoder layers")
                #freeze the entire backbone
            # self.cnns[1].eval()
            # for param in self.cnns[1].parameters():
            #     param.requires_grad = False
            # for name, param in self.cnns[1].named_parameters():
            #     param.requires_grad = False
        #    self.final.eval()
        #    for param in self.final.parameters():
        #        param.requires_grad = False
        #else:
        #    print('not freezing')
        
        # self.encoder = TBeoEncoder(channel_in=3, ch=32, z=512, div=model_config['custom_model_config']['div'])


        # hardcode for colosseum
        if "clip" in model_config['custom_model_config']['backbone_path']:
            print("clip")
            self.encoder = ColoClipEncoder(512)
        elif "r3m" in model_config['custom_model_config']['backbone_path']:
            print("r3m")
            self.encoder = ColoR3MEncoder(512)
        elif "mvp" in model_config['custom_model_config']['backbone_path']:
            print("mvp")
            self.encoder = ColoMVPEncoder(512)
        elif "vc1" in model_config['custom_model_config']['backbone_path']:
            print("vc1")
            self.encoder = ColoVC1Encoder(512)
        else:
            self.encoder = TColEncoder(channel_in=3, ch=32, z=512)
            checkpoint_path = model_config['custom_model_config']['backbone_path']
        # elif "e2e" not in model_config['custom_model_config']['backbone_path'] and "random" not in model_config['custom_model_config']['backbone_path']:
        #     print(model_config['custom_model_config']['backbone_path'])
        #     print("loading model weights")
        #     if "RESNET" in model_config['custom_model_config']['backbone_path']:
        #         self.encoder = TBeoEncoder(channel_in=3, ch=64, z=512)
        #     checkpoint = torch.load(model_config['custom_model_config']['backbone_path'], map_location="cpu")
        #     self.encoder.load_state_dict(checkpoint['model_state_dict'])
        # self.encoder = TClipEncoder(512)

        
        if 'random' not in model_config['custom_model_config']['backbone_path'] and 'r3m' not in model_config['custom_model_config']['backbone_path'] and 'mvp' not in model_config['custom_model_config']['backbone_path'] and 'vc1' not in model_config['custom_model_config']['backbone_path'] and 'clip' not in model_config['custom_model_config']['backbone_path']:
            print("loading weights")
            state_dict = torch.load(checkpoint_path)
            self.encoder.load_state_dict(state_dict['model_state_dict'])
        else:
            print(model_config['custom_model_config']['backbone_path'])
        #self.linear = SlimFC(in_size=18432, out_size=512, activation_fn= 'tanh', initializer=torch_normc_initializer(0.01))


        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

        # if not model_config['custom_model_config']['train_backbone']:
        #     print("freezing encoder layers")
        #         #freeze the entire backbone
        #     self.encoder.eval()
        #     for param in self.encoder.parameters():
        #         param.requires_grad = False
        #     for name, param in self.encoder.named_parameters():
        #         param.requires_grad = False
        # else:
        #     print('not freezing')

        
        #dd = self.trainable_variables(True)    
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
            #out = self.encoder(input_dict['obs']['obs']/255.0, input_dict['obs']['aux'])
            # out = self.encoder(input_dict['obs']['obs'], input_dict['obs']['aux'])
            # out = self.encoder(input_dict['obs']['front_rgb']/255.0)
            #out = self.encoder(input_dict['obs']['front_rgb'])
            #out = self.encoder(input_dict['obs']/1.0)
            out = self.encoder(input_dict['obs']/self.div)
            #out = self.linear(out)

        # No logits/value branches.
        if self.logits_layer is None:
            return out, []

        # Logits- and value branches.
        logits = self.logits_layer(self.fc1(out))
        values = self.value_layer(out)
        # print(out.shape, logits.shape, values.shape)
        self._value_out = torch.reshape(values, [-1])
        return logits, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out

