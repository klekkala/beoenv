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
ModelCatalog.register_custom_model("model", SingleAtariModel)


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



def seed_everything(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    env.seed(seed)
    return env


#transfer from policy_model_path to backbone_model_path
#env_name has to be that of policy_model_path
def eval_adapter(env_name, backbone_model_path, policy_model_path, rounds):
    print(env_name, policy_model_path)
    backbone_model = Policy.from_checkpoint(backbone_model_path)
    policy_model = Policy.from_checkpoint(policy_model_path)
    policy_weights = policy_model.get_weights()
    backbone_weights = backbone_model.get_weights()
    for key in policy_weights.keys():
        if 'logits' in key:
            backbone_weights[key] = policy_weights[key]

    #backbone_weights_copy = copy.deepcopy(backbone_weights)
    #for key in backbone_weights_copy.keys():
    #    if 'logits' in key:
    #        backbone_weights_copy[key] = np.random.rand(*backbone_weights_copy[key].shape)
    policy_model.set_weights(policy_weights)
    backbone_model.set_weights(backbone_weights)
    #backbone_model.set_weights(backbone_weights_copy)
    res=[]

    for i in range(rounds):
        env = SingleAtariEnv({'env': env_name, 'full_action_space': False, 'framestack': False})
        random_generated_int = random.randint(0, 2**31-1)
        env = seed_everything(random_generated_int, env)
        obs = env.reset()
        env.seed(random_generated_int)

        total = 0.0
        for _ in range(4650):
            action = policy_model.compute_single_action(obs)[0]
            obs, reward, done, _ = env.step(action)
            total += reward
            if done:
                break
        res.append(total)
        #print("lsjf;lasjlkdfjlk;asjdflk;saj;fdjslak;dff", res1, policy_model_path)
        
    
    return str(round(mean(res),1)) + '±' + str(round(stddev(res),1))

g3='CarnivalNoFrameskip-v4'
g4='AirRaidNoFrameskip-v4'
g5='NameThisGameNoFrameskip-v4'
g6='PhoenixNoFrameskip-v4'

from model_checkpts import *

#these models are trained on g1
#demon
models1 = {'E2E ':g1_e2e, 'RANDOM ':g1_random, 'SOM ':g1_SOM, 'TCN ':g1_TCN,'VIP ':g1_VIP, 'VEP ':g1_VEP}

#these models are trained on g2
models2 = {'E2E ':g2_e2e, 'RANDOM ':g2_random, 'SOM ':g2_SOM, 'TCN ':g2_TCN, 'VIP ':g2_VIP, 'VEP ':g2_VEP}



res = {}
numrounds = 10


for key,value in models1.items():
    g3_g1 = eval_adapter(g3, value, value, numrounds)
    g3_g2 = eval_adapter(g3, models2[key], models2[key], numrounds)
    
    
    g4_g1 = eval_adapter(g4, value, value, numrounds)
    g4_g2 = eval_adapter(g4, models2[key], models2[key], numrounds)
    

    g5_g1 = eval_adapter(g5, value, value, numrounds)
    g5_g2 = eval_adapter(g5, models2[key], models2[key], numrounds)
    
    
    res[key] = [g3_g1 + ' | ' + g3_g2, g4_g1 + ' | ' + g4_g2, g5_g1 + ' | ' + g5_g2]

df = pd.DataFrame(res).T
df.columns = [f'{g3}', f' {g4}', f' {g5}']
df.to_csv('newgame_evaluation.csv')
 
