

#This file is used when we have multiple models and games (1 model/game)

from ppotrainer import AtariPPOTorchRLModule, BeoGymPPOTorchRLModule




def generate_spec(envs):

    #If its a single environment
    if len(envs) == 1:
        spec = SingleAgentRLModuleSpec(
            module_class=AtariPPOTorchRLModule,
            #observation_space=gym.spaces.Box(0, 255, (84, 84, 3), np.uint8),
            #action_space=gym.spaces.Discrete(18),
            model_config_dict={"vf_share_layers": True, "encoder_latent_dim": 32,
                "conv_filters": None,
                "fcnet_hiddens": [128, 32]},
            catalog_class=CustomPPOCatalog
        return spec



    else:
        #just like gen_policies
        for i in _:
            ModelCatalog.register_custom_model("model1", mod1)
            ModelCatalog.register_custom_model("model2", mod2)






def select_policy(agent_id, episode, worker, **kwargs):
    print(type(agent_id), agent_id, agent_id==0, agent_id==1)
    print(agent_id, episode.episode_id, worker)
    #towards the end the int value is agent_id
    if agent_id == 0:
        return "0"
    elif agent_id == 1:
        return "1"
    #return "1"
