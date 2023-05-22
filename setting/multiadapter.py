class BCTorchRLModuleWithSharedGlobalEncoder(TorchRLModule):
    """An RLModule with a shared encoder between agents for global observation."""

    def __init__(
        self, encoder: nn.Module, hidden_dim: int, action_dim: int
    ) -> None:
        super().__init__(config=None)

        #encoder
        self.encoder = encoder
        
        #adapter
        #load adpater from the ckpt
        
        #policy
        self.policy_head = nn.Sequential(
            nn.Linear(2048, hidden_dim),
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
        return self._train_forward(batch)

    def _common_forward(self, batch):
        obs = batch["obs"]
        global_enc = self.encoder(obs)
        policy_in = global_enc
        action_logits = self.policy_head(policy_in)

        return {"action_dist": torch.distributions.Categorical(logits=action_logits)}

class BCTorchMultiAgentModuleWithSharedEncoder(MultiAgentRLModule):
    def __init__(self, config: MultiAgentRLModuleConfig) -> None:
        super().__init__(config)

    def build(self):

        module_specs = self.config.modules
        module_spec = next(iter(module_specs.values()))
        hidden_dim = module_spec.model_config_dict["fcnet_hiddens"][0]
        
        #declare all common modules here
        shared_encoder = Encoder(channels = 3)
        #shared_adapter
        #shared_policy
        
        rl_modules = {}
        for module_id, module_spec in module_specs.items():
            rl_modules[module_id] = BCTorchRLModuleWithSharedGlobalEncoder(
                encoder=shared_encoder,
                hidden_dim=hidden_dim,
                action_dim=module_spec.action_space.n,
            )

        self._rl_modules = rl_modules



spec = MultiAgentRLModuleSpec(
    marl_module_class=BCTorchMultiAgentModuleWithSharedEncoder,
    module_specs={
        "airraid": SingleAgentRLModuleSpec(
            observation_space=gym.spaces.Dict(
                {
                    "global": gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)
                }
            ),
            action_space=gym.spaces.Discrete(18),
            model_config_dict={"fcnet_hiddens": [64]},
        ),
        "assault": SingleAgentRLModuleSpec(
            observation_space=gym.spaces.Dict(
                {
                    "global": gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)
                }
            ),
            action_space=gym.spaces.Discrete(18),
            model_config_dict={"fcnet_hiddens": [64]},
        ),
    },
)