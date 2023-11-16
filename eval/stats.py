import gym
import numpy as np
from ray.rllib.models import ModelCatalog
import envs
import torch
from IPython import embed
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
import json

ModelCatalog.register_custom_model("model", SingleAtariModel)


if sys.argv[1] == 'atari':
    games = {'DemonAttackNoFrameskip-v4': 'DA', 'SpaceInvadersNoFrameskip-v4': 'SA', 'CarnivalNoFrameskip-v4': 'CA', 'AirRaidNoFrameskip-v4': 'AR', 'NameThisGameNoFrameskip-v4': 'NG'}
    from atari_checkpts import models
    file_path = 'atari.csv'

else:
    games = {'Wall_Street': 'WS', 'Union_Square': 'US', 'Hudson_River': 'HR', 'CMU': 'CMU', 'Shore_Street': 'SS', 'Allegheny': 'AG'}
    from beogym_checkpts import models
    file_path = 'beogym.csv'


def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/n # in Python 2 use sum(data)/float(n)

def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss

def stddev(data, ddof=0):
    """Calculates the population standard deviation
    by default; specify ddof=1 to compute the sample
    standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/(n-ddof)
    return pvar**0.5


def load_csv():
    lis = []
    return lis

def seed_everything(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    env.seed(seed)
    return env
    

def process_file(file_path):
    category_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            components = line.strip().split(',')

            category = components[0]
            game = components[1]
            values = list(map(float, components[2:]))

            if category not in category_dict:
                category_dict[category] = {}

            category_dict[category][game] = values

    return category_dict


    
outdict = process_file(file_path)

result = {}
#return 

#iterate through all the game_specific models
for modelkey, _ in outdict.items():
    result[modelkey] = []

    #iterate through all the games
    for key, values in outdict[modelkey].items():
        #iterate through all games
        result[modelkey].append(str(round(mean(values),1)) + '±' + str(round(stddev(values),1)) + str(' '))

df = pd.DataFrame(result).T

final_cols = []
for gamekey, gamevalue in games.items():
    final_cols.append(f'{gamevalue} ')

df.columns = final_cols
df.to_csv(sys.argv[1] + '_evaluation.csv')
 

#random_generated_int = random.randint(0, 2**31-1)
#env = seed_everything(random_generated_int, env)