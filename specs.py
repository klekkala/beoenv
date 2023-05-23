



def generate_spec(envs, ):

    if len(envs) == 1:
        spec = SingleAgentRLModuleSpec(
            module_class=PPOTorchRLModule,
            #observation_space=gym.spaces.Box(0, 255, (84, 84, 3), np.uint8),
            #action_space=gym.spaces.Discrete(18),
            model_config_dict={"vf_share_layers": True, "encoder_latent_dim": 32,
                "conv_filters": None,
                "fcnet_hiddens": [128, 32]},
            catalog_class=CustomPPOCatalog
        return spec



    else:
        raise NotImplementedError
        """
        module_specs = {}
        env_list = []    
        for i in _:
            module_specs[i] = SingleAgentRLModuleSpec(
            module_class=PPOTorchRLModule,
            #observation_space=gym.spaces.Box(0, 255, (84, 84, 3), np.uint8),
            #action_space=gym.spaces.Discrete(18),
            model_config_dict={"vf_share_layers": True, "encoder_latent_dim": 32,
                "conv_filters": None,
                "fcnet_hiddens": [128, 32]},
            catalog_class=CustomPPOCatalog
        )
    
        return module_specs 
        """


        """
        module_specs={
            "0" : SingleAgentRLModuleSpec(
        module_class=CustomPPOTorchRLModule,
        #see how to remove the below 2 lines
        observation_space=gym.spaces.Box(0, 255, (84, 84, 3), np.uint8),
        action_space=gym.spaces.Discrete(18),
        model_config_dict={"vf_share_layers": True, "encoder_latent_dim": 32,
            "conv_filters": None,
            "fcnet_hiddens": [128, 32]},
        catalog_class=CustomPPOCatalog
        ),
            "1" : SingleAgentRLModuleSpec(
        module_class=CustomPPOTorchRLModule,
        observation_space=gym.spaces.Box(0, 255, (84, 84, 3), np.uint8),
        action_space=gym.spaces.Discrete(18),
        model_config_dict={"vf_share_layers": True, "encoder_latent_dim": 32,
            "conv_filters": None,
            "fcnet_hiddens": [128, 32]},
        catalog_class=CustomPPOCatalog
        ),


        multispec = MultiAgentRLModuleSpec(
        marl_module_class=MultiPPOAtari,
        module_specs = {}
        }
        """




def select_policy(agent_id, episode, worker, **kwargs):
    print(type(agent_id), agent_id, agent_id==0, agent_id==1)
    print(agent_id, episode.episode_id, worker)
    #towards the end the int value is agent_id
    if agent_id == 0:
        return "0"
    elif agent_id == 1:
        return "1"
    #return "1"
