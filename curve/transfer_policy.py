import gym
import numpy as np
from ray.rllib.models import ModelCatalog
import envs
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
        obs1 = env1.reset()
        obs2 = env2.reset()
        random_generated_int = random.randint(0, 2**31-1)
        env1.seed(random_generated_int)
        env2.seed(random_generated_int)

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

g1_e2e = '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_e2e_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_11_15_10_22/checkpoint/'
g1_SOM = '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_SOM_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_150_0.1_32_32_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_12_14_10_18/checkpoint/'
g1_TCN = '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_TCN_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_4.0_0.01_32_32_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_12_11_47_47/checkpoint/'
g1_VIP = '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_VIP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_50.0_15.0_0.01_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_16_22_29_43/checkpoint/'
g1_VEP = '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_VEP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_1.0_0.1_1.0_same_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_19_18_54_18/checkpoint/'
g1_random = '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_random_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_17_18_27_30/checkpoint/'



g2_e2e = '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_e2e_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_11_15_12_08/checkpoint/'
g2_SOM = '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_1CHAN_SOM_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_150_0.1_32_32_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_12_14_09_57/checkpoint/'
g2_TCN = '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_1CHAN_TCN_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_4.0_0.01_32_32_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_12_11_48_05/checkpoint/'
g2_VIP = '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_1CHAN_VIP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_50.0_15.0_0.01_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_16_22_29_56/checkpoint/'
g2_VEP = '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_1CHAN_VEP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_1.0_0.1_1.0_same_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_19_18_53_54/checkpoint/'
g2_random = '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_random_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_17_13_36_39/checkpoint/'


#these models are trained on g1
#demon
models1 = {'e2e':g1_e2e, 'random':g1_random, 'SOM':g1_SOM, 'TCN':g1_TCN,'VIP':g1_VIP, 'VEP':g1_VEP}

#these models are trained on g2
models2 = {'e2e':g2_e2e, 'random':g2_random, 'SOM':g2_SOM, 'TCN':g2_TCN, 'VIP':g2_VIP, 'VEP':g2_VEP}



res = {}
numrounds = 10
#first measure the best performance on each game using end to end training (no transfer)
res[''] = [eval_adapter(g1, g1_e2e, g1_e2e, numrounds), eval_adapter(g2, g2_e2e, g2_e2e, numrounds)]


for key,value in models1.items():
    g1g2 = eval_adapter(g1, models2[key], value, numrounds)
    g2g1 = eval_adapter(g2, value, models2[key], numrounds)
    res[key] = [g1g2, g2g1]

df = pd.DataFrame(res).T
df.columns = [f'{g1}', f' {g2}']
df.to_csv('policy_evaluation.csv')
 
