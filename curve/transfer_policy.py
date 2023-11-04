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
    assert(env_name in policy_model_path)
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
    res1=[]
    res2=[]

    for i in range(rounds):
        
        env1 = SingleAtariEnv({'env': env_name, 'full_action_space': False, 'framestack': False})
        env2 = SingleAtariEnv({'env': env_name, 'full_action_space': False, 'framestack': False})
        
        random_generated_int = random.randint(0, 2**31-1)
        env1 = seed_everything(random_generated_int, env1)
        obs1 = env1.reset()
        env1.seed(random_generated_int)

        total1 = 0.0
        total2 = 0.0
        for _ in range(4650):
            action = policy_model.compute_single_action(obs1)[0]
            obs1, reward, done, _ = env1.step(action)
            total1 += reward
            if done:
                break
        res1.append(total1)
        #print("lsjf;lasjlkdfjlk;asjdflk;saj;fdjslak;dff", res1, policy_model_path)
        
        if backbone_model_path != policy_model_path:
            random_generated_int = random.randint(0, 2**31-1)
            env2 = seed_everything(random_generated_int, env2)
            obs2 = env2.reset()
            env2.seed(random_generated_int)

            for _ in range(4650):
                action = backbone_model.compute_single_action(obs2)[0]
                obs2, reward, done, _ = env2.step(action)
                total2 += reward
                if done:
                    break
            res2.append(total2)
    
    
    if backbone_model_path != policy_model_path:
        return str(round(mean(res1),1)) + '±' + str(round(stddev(res1),1)) + ' / ' + str(round(mean(res2),1)) + '±' + str(round(stddev(res2),1))
    else:
        return str(round(mean(res1),1)) + '±' + str(round(stddev(res1),1))

g1='DemonAttackNoFrameskip-v4'
g2='SpaceInvadersNoFrameskip-v4'

from model_checkpts import *

#these models are trained on g1
#demon
models1 = {'E2E ':g1_e2e, 'RANDOM ':g1_random, 'SOM ':g1_SOM, 'TCN ':g1_TCN,'VIP ':g1_VIP, 'VEP ':g1_VEP, 'MVEP ':g1_MVEP}

#these models are trained on g2
models2 = {'E2E ':g2_e2e, 'RANDOM ':g2_random, 'SOM ':g2_SOM, 'TCN ':g2_TCN, 'VIP ':g2_VIP, 'VEP ':g2_VEP, 'MVEP ':g2_MVEP}



res = {}
numrounds = 15
#first measure the best performance on each game using end to end training (no transfer)
res[''] = [eval_adapter(g1, g1_e2e, g1_e2e, numrounds), eval_adapter(g2, g2_e2e, g2_e2e, numrounds)]


for key,value in models1.items():
    g1g2 = eval_adapter(g1, models2[key], value, numrounds)
    g2g1 = eval_adapter(g2, value, models2[key], numrounds)
    res[key] = [g1g2, g2g1]

df = pd.DataFrame(res).T
df.columns = [f'{g1}', f' {g2}']
df.to_csv('policy_evaluation.csv')
 
