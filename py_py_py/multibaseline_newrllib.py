import gymnasium as gym
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from torchvision.models import resnet18, ResNet18_Weights
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from typing import Mapping, Any
import cv2
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.tf.visionnet import VisionNetwork
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.marl_module import (
    MultiAgentRLModuleConfig,
    MultiAgentRLModule,
)
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
import numpy as np
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from pprint import pprint
from ray.rllib.policy.sample_batch import SampleBatch
#env = gym.make("ALE/BeamRider-v5")
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from RES_VAE import VAE

from ray import tune
from typing import Mapping, Any

from ray.rllib.algorithms.ppo.ppo_rl_module import PPORLModule

from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT, STATE_IN
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.nested_dict import NestedDict

from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.tf.visionnet import VisionNetwork
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.marl_module import (
    MultiAgentRLModuleConfig,
    MultiAgentRLModule,
)
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.env.multi_agent_env import MultiAgentEnv

torch, nn = try_import_torch()

#envs = ["ALE/AirRaid-v5", "ALE/Assault-v5", "ALE/BeamRider-v5", "ALE/Carnival-v5", "ALE/DemonAttack-v5", "ALE/NameThisGame-v5", "ALE/Phoenix-v5", "ALE/Riverraid-v5", "ALE/SpaceInvaders-v5"]
envs = ["ALE/Assault-v5", "ALE/BeamRider-v5"]
#emap = ["assault", "beamrider"]

class MultiSync(MultiAgentEnv):

    def __init__(self,num):
        self.agents=[]
        for i in range(len(envs)):
            self.agents.append(gym.make(envs[i], full_action_space=True))
        self.terminateds = set()
        self.truncateds = set()
        self.action_space = gym.spaces.Discrete(18)
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)
        self.resetted = False

    def reset(self, *, seed=None, options=None):
        res={}
        info={}
        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()
        for i in range(len(envs)):
            temp,info = self.agents[i].reset()
            temp = cv2.resize(temp, (84, 84))
            res[i]=temp
            info[i] = info
        #print("reset", res, info)
        return res,info

    def step(self, action_dict):
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}
        for i, action in action_dict.items():
            temp = self.agents[i].step(action)
            temp=list(temp)
            temp[0] = cv2.resize(temp[0], (84, 84))
            obs[i], rew[i], terminated[i], truncated[i], info[i] = temp
            if terminated[i]:
                self.terminateds.add(i)
            if truncated[i]:
                self.truncateds.add(i)

        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = len(self.truncateds) == len(self.agents)
        #print("obs", obs, rew)
        return obs, rew, terminated, truncated, info


#from customppo import CustomPPOCatalog
class CustomPPOTorchRLModule(PPORLModule, TorchRLModule):
    framework: str = "torch"
    def __init__(self, *args, **kwargs) -> None:
        #backbone = kwargs['backbone']
        #if 'policy' in kwargs:
        #    self.pi = kwargs['policy']
        
        #kwargs = {}
        TorchRLModule.__init__(self, *args, **kwargs)
        PPORLModule.__init__(self, *args, **kwargs)


        #self.encoder = backbone

        
        #print(self.encoder)
        #print(self.pi)
        #print(self.vf)

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
        """PPO forward pass during exploration.
        Besides the action distribution, this method also returns the parameters of the
        policy distribution to be used for computing KL divergence between the old
        policy and the new policy during training.
         """
        with torch.no_grad():
            return self._common_forward(batch)

    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        return self._common_forward(batch)

    def _common_forward(self, batch: NestedDict) -> Mapping[str, Any]:
        output = {}

        # Shared encoder
        #batch['obs'] = self.backbone(batch['obs'])
        encoder_outs = self.encoder(batch)
        if STATE_OUT in encoder_outs:
            output[STATE_OUT] = encoder_outs[STATE_OUT]

        # Value head
        vf_out = self.vf(encoder_outs[ENCODER_OUT][CRITIC])
        output[SampleBatch.VF_PREDS] = vf_out.squeeze(-1)

        # Policy head
        action_logits = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        output[SampleBatch.ACTION_DIST_INPUTS] = action_logits

        return output

tune.register_env('MultiSync', lambda config: MultiSync(config))

def get_expected_module_config(
    model_config_dict: dict,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    catalog_class = PPOCatalog,
) -> RLModuleConfig:
    """Get a PPOModuleConfig that we would expect from the catalog otherwise.

    Args:
        env: Environment for which we build the model later
        model_config_dict: Model config to use for the catalog
        observation_space: Observation space to use for the catalog.

    Returns:
         A PPOModuleConfig containing the relevant configs to build PPORLModule
    """
    config = RLModuleConfig(
        observation_space=observation_space,
        action_space=action_space,
        model_config_dict=model_config_dict,
        catalog_class=catalog_class,
    )

    return config


class MultiPPOAtari(MultiAgentRLModule):
    def __init__(self, config: MultiAgentRLModuleConfig) -> None:
        super().__init__(config)
    
    def setup(self):
        module_specs = self.config.modules
        #print(self.config.modules)
        module_spec = next(iter(module_specs.values()))
        module_dict = module_spec.model_config_dict

        #specify special instructions for building encoder
        #catalog = CustomPPOCatalog(gym.spaces.Box(0, 255, (84, 84, 3), np.uint8),
        #                            gym.spaces.Discrete(18),
        #                            model_config_dict={"vf_share_layers": True, 
        #                                                "encoder_latent_dim": 32})

        #shared_encoder = catalog.build_actor_critic_encoder(framework="torch")

        rl_modules = {}
        for module_id, module_spec in module_specs.items():
            print(module_id)
            #print(module_spec)
            config = get_expected_module_config(
                model_config_dict=module_dict, observation_space=module_spec.observation_space,
                action_space=module_spec.action_space,
                #catalog_class=module_spec.catalog_class
            )

            rl_modules[module_id] = module_spec.module_class(
                config,
                #backbone=shared_encoder,
            )

        self._rl_modules = rl_modules



multispec = MultiAgentRLModuleSpec(
    marl_module_class=MultiPPOAtari,
    module_specs={
        "0" : SingleAgentRLModuleSpec(
    module_class=CustomPPOTorchRLModule,
    #see how to remove the below 2 lines
    observation_space=gym.spaces.Box(0, 255, (84, 84, 3), np.uint8),
    action_space=gym.spaces.Discrete(18),
    model_config_dict={"vf_share_layers": True, "encoder_latent_dim": 32,
        "conv_filters": None,
        "fcnet_hiddens": [128, 32]},
    #catalog_class=CustomPPOCatalog
),
        "1" : SingleAgentRLModuleSpec(
    module_class=CustomPPOTorchRLModule,
    observation_space=gym.spaces.Box(0, 255, (84, 84, 3), np.uint8),
    action_space=gym.spaces.Discrete(18),
    model_config_dict={"vf_share_layers": True, "encoder_latent_dim": 32,
        "conv_filters": None,
        "fcnet_hiddens": [128, 32]},
    #catalog_class=CustomPPOCatalog
),
        #"default_policy" : SingleAgentRLModuleSpec(
    #module_class=CustomPPOTorchRLModule,
    #observation_space=gym.spaces.Box(0, 255, (84, 84, 3), np.uint8),
    #action_space=gym.spaces.Discrete(18),
    #model_config_dict={"vf_share_layers": True, "encoder_latent_dim": 32,
    #    "conv_filters": None,
    #    "fcnet_hiddens": [128, 32]},
    #catalog_class=CustomPPOCatalog
#),
    }
)

#import sys
#sys.setrecursionlimit(1 << 25)

#import threading
#threading.stack_size(1 << 34)

def select_policy(agent_id, episode, worker, **kwargs):
    print(type(agent_id), agent_id, agent_id==0, agent_id==1)
    print(agent_id, episode.episode_id, worker)
    #towards the end the int value is agent_id
    if agent_id == 0:
        return "0"
    #return '1'
    elif agent_id == 1:
        return "1"
    #return "1"




config = (
    PPOConfig()
    .environment(MultiSync, env_config={
        "frameskip": 1,
        "full_action_space": False,
        "repeat_action_probability": 0.0
        })
    .training(_enable_learner_api=True)
    .rl_module(
        _enable_rl_module_api=True,
        #rl_module_spec=marl_module,
        rl_module_spec=multispec,
    )
    .training(lambda_=0.95,
        lr=0.0001,
        kl_coeff=0.5,
        clip_param=0.1,
        vf_clip_param=10.0,
        grad_clip=100.0,
        entropy_coeff=0.01,
        train_batch_size=5000,
        sgd_minibatch_size=500,
        num_sgd_iter=10)
    .multi_agent(
        policies={
            "0",
            "1"#,
            #"default_policy"
        },
        #policies_to_train=["0"],
        #policy_mapping_fn=select_policy
    )
    .exploration(exploration_config={})
    .reporting(min_time_s_per_iteration=30)
    .rollouts(num_rollout_workers=1,
        num_envs_per_worker=1,
        rollout_fragment_length='auto')
    .resources(
        num_gpus_per_learner_worker = 1,
        num_gpus=0,
        num_learner_workers=1,
        num_cpus_per_worker=.5,
        num_gpus_per_worker=0)
        #num_cpus_for_local_worker = 1)
)

algorithm = config.build()


# run for 2 training steps
for _ in range(2000000):
    result = algorithm.train()
    pprint(result)
