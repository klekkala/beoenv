

#This file is used when we have multiple models and games (1 model/game)
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
"""
import configs
from ray.rllib.models import ModelCatalog
import gym
import numpy as np
from arguments import get_args
args = get_args()

#THIS IS ONLY TESTED FOR ATARI. IT WONT WORK FOR BEOGYM!!!
def gen_policy(i):
    if args.temporal == '4stack':
        obs_space = gym.spaces.Box(0, 255, (84, 84, 4), np.uint8)
    else:
        obs_space = gym.spaces.Box(0, 255, (84, 84, 1), np.uint8)
    
    if args.backbone == "e2e":
        args.train_backbone = True

    #6D action space
    act_space = gym.spaces.Discrete(6)
    config = {
        "model": {
            "custom_model": "model_" + str(i),
            "vf_share_layers": True,
            "conv_filters": [[16, [8, 8], 4], [32, [4, 4], 2], [512, [11, 11], 1],],
            "conv_activation" : "relu" if args.temporal == '4stack' else "elu",
            "custom_model_config" : {"backbone": args.backbone, "backbone_path": configs.map_models[args.backbone], "train_backbone": args.train_backbone},
            "framestack": args.temporal == '4stack',
            "use_lstm": args.temporal == 'lstm',
            "use_attention": args.temporal == 'attention',    
        }
    }
    #return PolicySpec(config=config)
    return (None, obs_space, act_space, config)

