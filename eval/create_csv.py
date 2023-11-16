import gym
import numpy as np
from ray.rllib.models import ModelCatalog
import envs
import torch
from envs import SingleAtariEnv
from ray.rllib.models.torch.visionnet import VisionNetwork
from atari_vae import Encoder, TEncoder
from ray.rllib.policy.policy import Policy
from models.atarimodels import SingleAtariModel
import random
import sys,os,random
import pandas as pd
import copy
from IPython import embed

ModelCatalog.register_custom_model("model", SingleAtariModel)



if sys.argv[1] == 'atari':
    games = {'DemonAttackNoFrameskip-v4': 'DA', 'SpaceInvadersNoFrameskip-v4': 'SA', 'CarnivalNoFrameskip-v4': 'CA', 'AirRaidNoFrameskip-v4': 'AR', 'NameThisGameNoFrameskip-v4': 'NG'}
    from atari_checkpts import models
    f = open('atari.csv', 'r')

else:
    games = {'Wall_Street': 'WS', 'Union_Square': 'US', 'Hudson_River': 'HR', 'CMU': 'CMU', 'Shore_Street': 'SS', 'Allegheny': 'AG'}
    from beogym_checkpts import models
    f = open('beogym.csv', 'r')




f = open('atari.csv', 'w')
for modeldict in models:
    # f.write(f'{modeldict}' + ', ')
    #iterate through all the models
    
    for key, value in modeldict.items():
        
        #iterate through all games
        for gamekey, gamevalue in games.items():
            f.write(key + ', ')
            f.write(gamekey)
            f.write('\n')

f.close()

f = open('atari.csv', 'r')
all_dict = {}
for line in f:
    components = line.strip().split(', ')

    category = components[0]
    game = components[1]
    values = list(map(float, components[2:]))

    if category not in all_dict:
        all_dict[category] = {}

    all_dict[category][game] = values

# exist_models=all_dict.keys()
# models = [i for i in models if i not in exist_models]
print(all_dict)
#iterate through all the game_specific models
f.close()


