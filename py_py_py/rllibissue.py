import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
import numpy as np
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from pprint import pprint
import cv2
from typing import Iterator, Mapping, Any, Union, Dict, Optional, Type, Set
from ray import tune
from typing import Mapping, Any
from ray.rllib.algorithms.ppo.ppo_rl_module import PPORLModule
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT, STATE_IN
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module.marl_module import (
    MultiAgentRLModuleConfig,
    MultiAgentRLModule,
)
torch, nn = try_import_torch()
ModuleID = str
#envs = ["AirRaidNoFrameskip-v4", "BeamRiderNoFrameskip-v4"]
envs = ["ALE/AirRaid-v5", "ALE/BeamRider-v5"]

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
        return obs, rew, terminated, truncated, info

tune.register_env('MultiSync', lambda config: MultiSync(config))

from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule

class CustomPPOTorchRLModule(PPORLModule, TorchRLModule):
    framework: str = "torch"
    def __init__(self, *args, **kwargs) -> None:
        TorchRLModule.__init__(self, *args, **kwargs)
        PPORLModule.__init__(self, *args, **kwargs)

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


def select_policy(agent_id, episode, worker, **kwargs):
    if agent_id == 0:
        print("airraid")
        return "airraid"
    elif agent_id == 1:
        print("beamrider")
        return "beamrider"

class MultiPPOAtari(MultiAgentRLModule):
    def __init__(self, config: MultiAgentRLModuleConfig) -> None:
        super().__init__(config)
    def _forward_train(
        self, batch: MultiAgentBatch, **kwargs
    ) -> Union[Mapping[str, Any], Dict[ModuleID, Mapping[str, Any]]]:
        """Runs the forward_train pass.

        TODO(avnishn, kourosh): Review type hints for forward methods.

        Args:
            batch: The batch of multi-agent data (i.e. mapping from module ids to
                SampleBaches).

        Returns:
            The output of the forward_train pass the specified modules.
        """
        print("blaslhjdfjskdfj")
        module_ids = list(batch.shallow_keys())
        print(module_ids)
        return self._run_forward_pass("_forward_train", batch, **kwargs)

    @override(RLModule)
    def _forward_inference(
        self, batch: MultiAgentBatch, **kwargs
    ) -> Union[Mapping[str, Any], Dict[ModuleID, Mapping[str, Any]]]:
        """Runs the forward_inference pass.

        TODO(avnishn, kourosh): Review type hints for forward methods.

        Args:
            batch: The batch of multi-agent data (i.e. mapping from module ids to
                SampleBaches).

        Returns:
            The output of the forward_inference pass the specified modules.
        """
        return self._run_forward_pass("_forward_inference", batch, **kwargs)

    @override(RLModule)
    def _forward_exploration(
        self, batch: MultiAgentBatch, **kwargs
    ) -> Union[Mapping[str, Any], Dict[ModuleID, Mapping[str, Any]]]:
        """Runs the forward_exploration pass.

        TODO(avnishn, kourosh): Review type hints for forward methods.

        Args:
            batch: The batch of multi-agent data (i.e. mapping from module ids to
                SampleBaches).

        Returns:
            The output of the forward_exploration pass the specified modules.
        """
        return self._run_forward_pass("_forward_exploration", batch, **kwargs)


multispec = MultiAgentRLModuleSpec(
    marl_module_class=MultiPPOAtari,
    module_specs={
        "airraid" : SingleAgentRLModuleSpec(
    module_class=PPOTorchRLModule,
    observation_space=gym.spaces.Box(0, 255, (84, 84, 3), np.uint8),
    action_space=gym.spaces.Discrete(18),
    model_config_dict={"vf_share_layers": True, "encoder_latent_dim": 32,
                    "fcnet_hiddens": [128, 32]},
    catalog_class = PPOCatalog
    ),
        "beamrider" : SingleAgentRLModuleSpec(
    module_class=PPOTorchRLModule,
    observation_space=gym.spaces.Box(0, 255, (84, 84, 3), np.uint8),
    action_space=gym.spaces.Discrete(18),
    model_config_dict={"vf_share_layers": True, "encoder_latent_dim": 32,
                    "fcnet_hiddens": [128, 32]},
    catalog_class = PPOCatalog
    )
    }
)

config = (
    PPOConfig()
    .environment(MultiSync)
    .training(_enable_learner_api=True)
    .rl_module(
        _enable_rl_module_api=True,
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
            "airraid",
            "beamrider"
        },
        policy_mapping_fn=select_policy
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
for _ in range(20):
    result = algorithm.train()
    pprint(result)
