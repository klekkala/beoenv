import gymnasium as gym
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from torchvision.models import resnet18, ResNet18_Weights
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from typing import Mapping, Any
import cv2
import numpy as np
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from pprint import pprint
from ray.rllib.policy.sample_batch import SampleBatch
#env = gym.make("ALE/BeamRider-v5")
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from beogym import BeoGym
from ray.rllib.core.models.catalog import Catalog

from typing import Mapping, Any

from ray.rllib.algorithms.ppo.ppo_rl_module import PPORLModule

from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT, STATE_IN, STATE_OUT
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.nested_dict import NestedDict

torch, nn = try_import_torch()
from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule

#from RES_VAE import VAE
#from customppo import CustomPPOCatalog
from customppo import CustomPPOCatalog
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule



class CustomPPOTorchRLModule(PPORLModule, TorchRLModule):
    framework: str = "torch"
    def __init__(self, *args, **kwargs) -> None:
        TorchRLModule.__init__(self, *args, **kwargs)
        PPORLModule.__init__(self, *args, **kwargs)

        print(self.encoder)
        #print(self.pi)
        #print(self.vf)


    # def output_specs_inference(self) -> SpecType:
    #     """Returns the output specs of the forward_inference method.

    #     Override this method to customize the output specs of the inference call.
    #     The default implementation requires the forward_inference to reutn a dict that
    #     has `action_dist` key and its value is an instance of `Distribution`.
    #     This assumption must always hold.
    #     """
    #     for i in range(100):
    #         print(1)
    #     return {"action_dist": Distribution}

    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        output = {}
        encoder_outs = self.encoder(batch)
        if STATE_OUT in encoder_outs:
            output[STATE_OUT] = encoder_outs[STATE_OUT]

        # Actions
        action_logits = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        output[SampleBatch.ACTION_DIST_INPUTS] = action_logits

        return output

    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        with torch.no_grad():
            return self._common_forward(batch)

    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        return self._common_forward(batch)

    def _common_forward(self, batch: NestedDict) -> Mapping[str, Any]:
        output = {}

        # Shared encoder
        #batch['obs'] = self.backbone(batch['obs'])
        obs=batch['obs']
        
        aux = obs[:, -6:]
        obs = obs[:, :-6]
        obs = obs.reshape((obs.shape[0], 208, 416, 3))
        data={'obs':obs,'aux':aux}

        encoder_outs = self.encoder(data)

        # for i in range(100):
        #     print(encoder_outs.keys())
        #     print(global_enc['encoder_out']['actor'].shape)
        #     print(global_enc['encoder_out']['critic'].shape)
        # encoder_outs = torch.cat([global_enc['encoder_out'], aux], dim=-1)

        #encoder_outs = self.encoder(batch)
        if STATE_OUT in encoder_outs:
            output[STATE_OUT] = encoder_outs[STATE_OUT]

        # Value head
        vf_out = self.vf(encoder_outs[ENCODER_OUT][CRITIC])
        output[SampleBatch.VF_PREDS] = vf_out.squeeze(-1)

        # Policy head
        action_logits = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        output[SampleBatch.ACTION_DIST_INPUTS] = action_logits

        return output





spec = SingleAgentRLModuleSpec(
    module_class=CustomPPOTorchRLModule,#DiscreteBCTorchModule,#PPOTorchRLModule,
    observation_space= gym.spaces.Dict({"obs": gym.spaces.Box(low=0, high=255, shape=(208, 416, 3), dtype=np.uint8), "aux": gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)}),
    action_space=gym.spaces.Discrete(5),
    model_config_dict={
        "vf_share_layers": True,
        #"use_lstm" : True, 
        "conv_filters": [[16, [3, 5], [1,2]], [32, [5, 5], 2], [64, [5, 5], 3], [128, [5, 5], 4], [256, [9, 9], 1]],
        "fcnet_hiddens": [64],
        "encoder_latent_dim":256,
        },
    catalog_class=CustomPPOCatalog,
)

config = (
    PPOConfig()
    .environment(BeoGym, env_config={})
    .training(_enable_learner_api=True)
    .rl_module(
        _enable_rl_module_api=True,
        rl_module_spec=spec,
    )
    .training(
        lambda_=0.95,
        lr=0.0001,
        kl_coeff=0.5,
        clip_param=0.1,
        vf_clip_param=10.0,
        grad_clip=100.0,
        entropy_coeff=0.01,
        #entropy_coeff=0.9,
        train_batch_size=1000,
        sgd_minibatch_size=100,
        num_sgd_iter=10,
        )
        #.exploration(exploration_config={})
    .reporting(min_time_s_per_iteration=30)
    .rollouts(num_rollout_workers=1,
        num_envs_per_worker=1,
        rollout_fragment_length='auto'
        )
    .resources(
        num_gpus_per_learner_worker = 2,
        num_gpus=0,
        num_learner_workers=1,
        num_cpus_per_worker=.5,
        num_gpus_per_worker=0)
        #num_cpus_for_local_worker = 1,
)

algorithm = config.build()


# run for 2 training steps
for _ in range(2000000):
    result = algorithm.train()
    pprint(result)
