

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


def gen_policy(i):
    obs_space = gym.spaces.Box(0, 255, (84, 84, 4), np.uint8)
    act_space = gym.spaces.Discrete(18)
    config = {
        "model": {
            "custom_model": "model_" + str(i)
        }
    }
    #return PolicySpec(config=config)
    return (None, obs_space, act_space, config)

