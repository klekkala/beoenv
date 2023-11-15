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





#transfer from policy_model_path to backbone_model_path
#env_name has to be that of policy_model_path
def eval_adapter(env_name, model_path):

    print(env_name, model_path)
    
    model = Policy.from_checkpoint(model_path)
        
    if sys.argv[1] == 'atari':
        env = SingleAtariEnv({'env': env_name, 'full_action_space': False, 'framestack': False})
    else:
        env = SingleBeoEnv({'env': env_name, 'data_path': '/home6/tmp/kiran/'})
    
    obs = env.reset()
    env.seed()

    total = 0.0
    horizon = 4650 if sys.argv[1] == 'atari' else 700

    for _ in range(horizon):
        if sys.argv[1] == 'beogym':
            obs['obs'] = obs['obs'].astype(np.float32)
        action = model.compute_single_action(obs)[0]
        obs, reward, done, _ = env.step(action)
        total += reward
        if done:
            break
    return total

if sys.argv[1] == 'atari':
    games = {'DemonAttackNoFrameskip-v4': 'DA', 'SpaceInvadersNoFrameskip-v4': 'SA', 'CarnivalNoFrameskip-v4': 'CA'}
else:
    games = {'Wall_Street': 'WS', 'Union_Square': 'US'}


if sys.argv[1] == 'beogym':
    from beogym_checkpts import models
else:
    from atari_checkpts import models





# rf = open('atari.csv', 'r')
# f = open('atari.csv', 'a+')
# all_lines = rf.readlines()

# filelines = []

# for each_line in all_lines:
#     gamename, modelname, _ = all_lines[0].split(', ')
#     total_value = eval_adapter(gamename, modelname)
#     filelines.append(str(total_value))

# f.writelines('\n')
# f.close()


with open('atari.csv', 'r') as f:
    all_lines = f.readlines()
new_lines=[]
for each_line in all_lines:
    modelname, gamename = each_line.strip().split(', ')[:2]
    model_path=''
    for model in models:
        if model.get(modelname, None):
            model_path = model[modelname]
    assert model_path!=''
    total_value = eval_adapter(gamename, model_path)
    new_lines.append(each_line.strip()+', '+str(total_value)+'\n')

with open('atari.csv', 'w') as f:
    f.writelines(new_lines)

# for modeldict in models:
#     f.write(f'{modeldict}' + ', ')
#     #iterate through all the models
    
#     for key, value in modeldict.items():
        
#         if all_dict.get(f'{modeldict}',None):
#             if all_dict[modeldict].get(f'{key}',None):
#                 continue
#         #iterate through all games
#         f.write(key + ', ')
#         for gamekey, gamevalue in games.items():

#             total_value = eval_adapter(gamekey, modeldict[key])
#             f.write(str(total_value) + ', ')
#     f.write('\n')

# f.close()