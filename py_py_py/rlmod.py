import gymnasium as gym
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec



from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.marl_module import (
    MultiAgentRLModuleConfig,
    MultiAgentRLModule,
)
from ray.rllib.utils.nested_dict import NestedDict

from ray.rllib.examples.env.multi_agent import MultiAgentCartPole

import torch
import torch.nn as nn


class BCTorchRLModuleWithSharedGlobalEncoder(TorchRLModule):
    """An RLModule with a shared encoder between agents for global observation."""

    def __init__(
        self, encoder: nn.Module, local_dim: int, hidden_dim: int, action_dim: int
    ) -> None:
        super().__init__(config=None)

        self.encoder = encoder
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + local_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def _forward_inference(self, batch):
        with torch.no_grad():
            return self._common_forward(batch)

    def _forward_exploration(self, batch):
        with torch.no_grad():
            return self._common_forward(batch)

    def _forward_train(self, batch):
        return self._common_forward(batch)

    def _common_forward(self, batch):
        obs = batch["obs"]
        global_enc = self.encoder(obs["global"])
        policy_in = torch.cat([global_enc, obs["local"]], dim=-1)
        action_logits = self.policy_head(policy_in)

        return {"action_dist": torch.distributions.Categorical(logits=action_logits)}


class BCTorchMultiAgentModuleWithSharedEncoder(MultiAgentRLModule):
    def __init__(self, config: MultiAgentRLModuleConfig) -> None:
        super().__init__(config)
    
    def build(self):

        module_specs = self.config.modules
        module_spec = next(iter(module_specs.values()))
        global_dim = module_spec.observation_space["global"].shape[0]
        hidden_dim = module_spec.model_config_dict["fcnet_hiddens"][0]
        shared_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        rl_modules = {}
        for module_id, module_spec in module_specs.items():
            rl_modules[module_id] = BCTorchRLModuleWithSharedGlobalEncoder(
                encoder=shared_encoder,
                local_dim=module_spec.observation_space["local"].shape[0],
                hidden_dim=hidden_dim,
                action_dim=module_spec.action_space.n,
            )

        self._rl_modules = rl_modules




spec = MultiAgentRLModuleSpec(
    marl_module_class=BCTorchMultiAgentModuleWithSharedEncoder,
    module_specs={
        "local_2d": SingleAgentRLModuleSpec(
            observation_space=gym.spaces.Dict(
                {
                    "global": gym.spaces.Box(low=-1, high=1, shape=(2,)),
                    "local": gym.spaces.Box(low=-1, high=1, shape=(2,)),
                }
            ),
            action_space=gym.spaces.Discrete(2),
            model_config_dict={"fcnet_hiddens": [64]},
        ),
        "local_5d": SingleAgentRLModuleSpec(
            observation_space=gym.spaces.Dict(
                {
                    "global": gym.spaces.Box(low=-1, high=1, shape=(2,)),
                    "local": gym.spaces.Box(low=-1, high=1, shape=(5,)),
                }
            ),
            action_space=gym.spaces.Discrete(5),
            model_config_dict={"fcnet_hiddens": [64]},
        ),
    },
)

module = spec.build()

import torch
from pprint import pprint

from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .framework("torch")
    .environment(MultiAgentCartPole)
    .rl_module(_enable_rl_module_api=True)
)

algorithm = config.build()

# run for 2 training steps
for _ in range(2):
    result = algorithm.train()
    pprint(result)



    
config = (
    PPOConfig()
    .environment(MultiSync)
    .rl_module(
        _enable_rl_module_api=True,
        rl_module_spec=MultiAgentRLModuleSpec(
            module_specs=SingleAgentRLModuleSpec(module_class=DiscreteBCTorchModule)
        ),
    )
    .training(model={"fcnet_hiddens": [256, 256]})
)

algorithm = config.build()
