import os
import time
import ray
import numpy as np
import train
from evaluation import evaluate

@ray.remote
def use_gpu():
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

if __name__ == "__main__":
    
    ray.init(address='auto')
    ray.cluster_resources()
    start_time = time.time()
    
    base_path = "/lab/kiran/models/trained/sess0/ppo/AirRaidNoFrameskip-v4.pt"
   
    res_envs = [train.atari.remote('AirRaidNoFrameskip-v4', base_path), train.atari.remote('AssaultNoFrameskip-v4', base_path), train.atari.remote('BeamRiderNoFrameskip-v4', base_path), train.atari.remote('CarnivalNoFrameskip-v4', base_path), train.atari.remote('DemonAttackNoFrameskip-v4', base_path), train.atari.remote('NameThisGameNoFrameskip-v4', base_path), train.atari.remote('PooyanNoFrameskip-v4', base_path), train.atari.remote('PhoenixNoFrameskip-v4', base_path), train.atari.remote('RiverraidNoFrameskip-v4', base_path), train.atari.remote('SpaceInvadersNoFrameskip-v4', base_path)]
    
    #res_envs = [train.atari.remote('AirRaidNoFrameskip-v4', base_path), train.atari.remote('AssaultNoFrameskip-v4', base_path), train.atari.remote('BeamRiderNoFrameskip-v4', base_path)]
   
    #res_envs = [train.atari.remote('AirRaidNoFrameskip-v4'),  train.atari.remote('AssaultNoFrameskip-v4'), train.atari.remote('BeamRiderNoFrameskip-v4')]
    ray.get(res_envs)
