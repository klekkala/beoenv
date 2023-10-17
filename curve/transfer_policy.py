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


def eval_adapter(env_name, backbone_model_path, policy_model_path, rounds):

    backbone_model = Policy.from_checkpoint(backbone_model_path)
    policy_model = Policy.from_checkpoint(policy_model_path)
    policy_weights = policy_model.get_weights()
    backbone_weights = backbone_model.get_weights()
    for key in policy_weights.keys():
        if 'convs' in key:
            policy_weights[key] = backbone_weights[key]

    backbone_weights_copy = copy.deepcopy(backbone_weights)
    for key in backbone_weights_copy.keys():
        if 'logits' in key:
            backbone_weights_copy[key] = np.random.rand(*backbone_weights_copy[key].shape)
    policy_model.set_weights(policy_weights)
    backbone_model.set_weights(backbone_weights_copy)
    res1=[]
    res2=[]
    env1 = SingleAtariEnv({'env': env_name, 'full_action_space': True, 'framestack': False})
    env2 = SingleAtariEnv({'env': env_name, 'full_action_space': True, 'framestack': False})

    for i in range(rounds):
        obs1 = env1.reset()
        obs2 = env2.reset()
        random_generated_int = random.randint(0, 2**31-1)
        env1.seed(random_generated_int)
        env2.seed(random_generated_int)

        total1 = 0.0
        total2 = 0.0
        for _ in range(5000):
            action = policy_model.compute_single_action(obs1)[0]
                
            obs1, reward, done, _ = env1.step(action)
                
            total1 += reward
            if done:
                break
        for _ in range(5000):
            action = backbone_model.compute_single_action(obs2)[0]
                
            obs2, reward, done, _ = env2.step(action)
                
            total2 += reward
            if done:
                break

        res1.append(total1)
        res2.append(total2)
    
    
    return str(sum(res2)/len(res2))+'/'+str(sum(res1)/len(res1))

g1='DemonAttackNoFrameskip-v4'
g2='SpaceInvadersNoFrameskip-v4'

g1_e2e = '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_e2e_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_11_15_10_22/checkpoint/'
g1_SOM = '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_SOM_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_150_0.1_32_32_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_12_14_10_18/checkpoint/'
g1_TCN = '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_TCN_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_4.0_0.01_32_32_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_12_11_47_47/checkpoint/'
# g1_VIP = '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_VIP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_40.0_0.01_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_12_15_22_15/checkpoint/'
g1_VEP = '/lab/kiran/logs/rllib/atari/notemp/1.a_DemonAttackNoFrameskip-v4_singlegame_full_1CHAN_VEP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_10.0_0.1_0.01_32_32_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_13_18_31_21/checkpoint/'

g2_e2e = '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_e2e_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_11_15_12_08/checkpoint/'
g2_SOM = '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_1CHAN_SOM_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_150_0.1_32_32_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_12_14_09_57/checkpoint/'
g2_TCN = '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_1CHAN_TCN_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_4.0_0.01_32_32_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_12_11_48_05/checkpoint/'
# g2_VIP = Policy.from_checkpoint('/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_1CHAN_VIP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_40.0_0.01_32_0_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_12_15_15_46/checkpoint/')
g2_VEP = '/lab/kiran/logs/rllib/atari/notemp/1.a_SpaceInvadersNoFrameskip-v4_singlegame_full_1CHAN_VEP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_10.0_0.1_0.01_32_32_0.0001_0.pt_PolicyNotLoaded_0.0_20000_2000_notemp/23_10_13_18_31_46/checkpoint/'



models1 = {'e2e':g1_e2e, 'SOM':g1_SOM, 'TCN':g1_TCN, 'VEP':g1_VEP}
models2 = {'e2e':g2_e2e, 'SOM':g2_SOM, 'TCN':g2_TCN, 'VEP':g2_VEP}


res = {}
for key,value in models1.items():
    g1g2 = eval_adapter(g2, value, models2[key], 10)
    g2g1 = eval_adapter(g1, models2[key], value, 10)
    res[key] = [g1g2, g2g1]

df = pd.DataFrame(res).T
df.columns = [f'{g1} encoder {g2} policy on {g2}', f'{g1} encoder {g2} policy on {g1}']
df.to_csv('VEP.csv')
    
