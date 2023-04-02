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

   
    res_envs = [train.atari.remote('AirRaidNoFrameskip-v4'), train.atari.remote('AssaultNoFrameskip-v4'), train.atari.remote('BeamRiderNoFrameskip-v4'), train.atari.remote('CarnivalNoFrameskip-v4'), train.atari.remote('DemonAttackNoFrameskip-v4'), train.atari.remote('NameThisGameNoFrameskip-v4'), train.atari.remote('PooyanNoFrameskip-v4'), train.atari.remote('PhoenixNoFrameskip-v4'), train.atari.remote('RiverraidNoFrameskip-v4'), train.atari.remote('SolarisNoFrameskip-v4'), train.atari.remote('SpaceInvadersNoFrameskip-v4')]
   
    #res_envs = [train.atari.remote('AirRaidNoFrameskip-v4'),  train.atari.remote('AssaultNoFrameskip-v4'), train.atari.remote('BeamRiderNoFrameskip-v4')]
    ray.get(res_envs)
