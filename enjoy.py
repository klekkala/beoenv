import gym
import numpy as np
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.algorithm import Algorithm
import cv2 
my_restored_policy = Policy.from_checkpoint("./testatari/atari_checkpoint/")


class MultiTaskEnv(gym.Env):
        def __init__(self, env_config):
            self.env = gym.make("NameThisGameNoFrameskip-v4", full_action_space=True)
            self.name= "NameThisGameNoFrameskip-v4"
            #self.env = wrap_deepmind(self.env)
            self.action_space = self.env.action_space
            self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8) #self.env.observation_space
            #if self.observation_space.shape[0]==214:
                #self.observation_space = gym.spaces.Box(0, 255, (210, 160, 3), np.uint8)

        def reset(self):
            temp = self.env.reset()
            if isinstance(temp, np.ndarray):
                return cv2.resize(temp, (84, 84))
            #if str(type(temp))!='tuple':
                #return cv2.resize(temp, (84, 84))
            temp=list(temp)
            temp[0] = cv2.resize(temp[0], (84, 84))
            #res = tuple((cv2.resize(temp[0], (84, 84)),temp[1]))
            return tuple(temp)

        def step(self, action):
            temp = self.env.step(action)
            if isinstance(temp, np.ndarray):
                return cv2.resize(temp, (84, 84))
            temp=list(temp)
            temp[0] = cv2.resize(temp[0], (84, 84))
            return tuple(temp)


def reset(env):
    temp = env.reset()
    if isinstance(temp, np.ndarray):
        return cv2.resize(temp, (84, 84))
            #if str(type(temp))!='tuple':
                #return cv2.resize(temp, (84, 84))
    temp=list(temp)
    temp[0] = cv2.resize(temp[0], (84, 84))
            #res = tuple((cv2.resize(temp[0], (84, 84)),temp[1]))
    return 

res=[]

env = gym.make("NameThisGameNoFrameskip-v4", full_action_space=True)
rounds=50
for i in range(rounds):
    total=0
    obs = env.reset()
    obs = cv2.resize(obs, (84, 84))
    for q in range(1000):
        action = my_restored_policy.compute_single_action(obs)[0]
        obs, reward, done, _ = env.step(action)
        obs = cv2.resize(obs, (84, 84))
        total+=reward
        if done:
            break
    res.append(total)
average = sum(res) / len(res)
print(average)
with open('Name.txt','w') as f:
    f.write(str(res))

