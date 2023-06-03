import gym
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.env import BaseEnv
from typing import Dict, Tuple
from ray.rllib.policy.policy import Policy
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
from ray import air, tune
import numpy as np
import cv2
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
##SingleTask, MultiTask, MultiEnv classes and their related classes/functions

class SingleAtariEnv(gym.Env):
    def __init__(self, env):
        self.env = wrap_deepmind(gym.make("ALE/BeamRider-v5", full_action_space=True))
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed=None, options=None):
        return self.env.reset()
        obs = self.env.reset()
        return cv2.resize(obs, (84, 84))

    def step(self, action):
        return self.env.step(action)
        obs, reward, done, info = self.env.step(action)
        obs = cv2.resize(obs, (84, 84))
        #cv2.imwrite('obs.png', obs)
        return obs, reward, done, info




class ParellelAtariEnv(gym.Env): 
    def __init__(self, envs):
        for i in range(len(envs['envs'])):
            if envs.worker_index%len(envs['envs'])==i:
                self.env = wrap_deepmind(gym.make(envs['envs'][i], full_action_space=True))
                self.name= envs['envs'][i]
        #self.env = wrap_deepmind(self.env)
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 4), np.uint8) #self.env.observation_space
        #if self.observation_space.shape[0]==214:
            #self.observation_space = gym.spaces.Box(0, 255, (210, 160, 3), np.uint8)

    def reset(self):
        return self.env.reset()
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
        return self.env.step(action)
        temp = self.env.step(action)
        if isinstance(temp, np.ndarray):
            return cv2.resize(temp, (84, 84))
        temp=list(temp)
        temp[0] = cv2.resize(temp[0], (84, 84))
        #res = tuple((cv2.resize(temp[0], (84, 84)),temp[1],temp[2],temp[3],temp[4]))
        return tuple(temp)



# class MultiAtariEnv(MultiAgentEnv):

#     def __init__(self, envs):
#         self.agents=[]
#         self.envs = envs['envs']
#         for i in range(len(self.envs)):
#             self.agents.append(gym.make(self.envs[i], full_action_space=True))
#         self.done = set()
#         self.action_space = gym.spaces.Discrete(18)
#         self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)
#         self.resetted = False

#     def reset(self, *, seed=None, options=None):
#         res={}
#         self.resetted = True
#         self.done = set()
#         for i in range(len(self.envs)):
#             temp = self.agents[i].reset()
#             temp = cv2.resize(temp, (84, 84))
#             res[i]=temp
#         return res

#     def step(self, action_dict):
#         obs, rew, done, info = {}, {}, {}, {}
#         for i, action in action_dict.items():
#             temp = self.agents[i].step(action)
#             temp=list(temp)
#             temp[0] = cv2.resize(temp[0], (84, 84))
#             obs[i], rew[i], done[i], info[i] = temp
#             if done[i]:
#                 self.done.add(i)

#         done["__all__"] = len(self.done) == len(self.agents)
#         return obs, rew, done, info

class MultiAtariEnv(MultiAgentEnv):

        def __init__(self,envs):
            self.agents=[]
            self.envs = envs['envs']
            for i in range(len(envs['envs'])):
                env=wrap_deepmind(gym.make(envs['envs'][i], full_action_space=True))
                self.agents.append(env)
            self.dones = set()
            self.action_space = gym.spaces.Discrete(18)
            self.observation_space = gym.spaces.Box(0, 255, (84, 84, 4), np.uint8)
            self.resetted = False

        def reset(self):
            res={}
            self.resetted = True
            self.dones = set()
            for i in range(len(self.envs)):
                temp = self.agents[i].reset()
                res[i]=temp 
            return res

        def step(self, action_dict):
            obs, rew, done, info = {}, {}, {}, {}
            for i, action in action_dict.items():
                temp = self.agents[i].step(action)
                obs[i], rew[i], done[i], info[i] = temp
                if done[i]:
                    self.dones.add(i)
            done["__all__"] = len(self.dones) == len(self.agents)
            return obs, rew, done, info


tune.register_env('SingleAtariEnv', lambda config: SingleAtariEnv(config))
tune.register_env('ParellelAtariEnv', lambda config: ParellelAtariEnv(config))
tune.register_env('MultiAtariEnv', lambda config: MultiAtariEnv(config))

atari = {'single': SingleAtariEnv, 'parellel': ParellelAtariEnv, 'multi': MultiAtariEnv}

#Henghui Todo
#class SingleBeoEnv((gym.Env))
class SingleBeoEnv(gym.Env):
    def __init__(self,env_config):
        self.env = BeoGym({})
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, seed=None, options=None):
        return self.env.reset()
        obs,info = self.env.reset()
        return cv2.resize(obs, (84, 84)), info

    def step(self, action):
        return self.env.step(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = cv2.resize(obs, (84, 84))
        #cv2.imwrite('obs.png', obs)
        return obs, reward, terminated, truncated, info

class ParellelBeoEnv(gym.Env): 
    def __init__(self, envs):
        for i in range(len(envs)):    
            if env_config.worker_index%9==i:
                self.env = BeoGym({})
                self.name= envs[i]
        #self.env = wrap_deepmind(self.env)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        #if self.observation_space.shape[0]==214:
            #self.observation_space = gym.spaces.Box(0, 255, (210, 160, 3), np.uint8)

    def reset(self):
        return self.env.reset()
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
        return self.env.step(action)
        temp = self.env.step(action)
        if isinstance(temp, np.ndarray):
            return cv2.resize(temp, (84, 84))
        temp=list(temp)
        temp[0] = cv2.resize(temp[0], (84, 84))
        #res = tuple((cv2.resize(temp[0], (84, 84)),temp[1],temp[2],temp[3],temp[4]))
        return tuple(temp)

class MultiBeoEnv(MultiAgentEnv):

    def __init__(self, envs):
        self.agents=[]
        self.envs = envs
        for i in range(len(self.envs)):
            self.agents.append(BeoGym({}))
        self.done = set()
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = self.agents[0].observation_space
        self.resetted = False

    def reset(self, *, seed=None, options=None):
        res={}
        info={}
        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()
        for i in range(len(envs)):
            temp = self.agents[i].reset()
            #temp = cv2.resize(temp, (84, 84))
            res[i]=temp
        return res

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {},  {}
        for i, action in action_dict.items():
            temp = self.agents[i].step(action)
            #temp=list(temp)
            #temp[0] = cv2.resize(temp[0], (84, 84))
            obs[i], rew[i], done[i],  info[i] = temp
            if done[i]:
                self.done.add(i)

        done["__all__"] = len(self.done) == len(self.agents)
        return obs, rew, done, info
#class ParellelBeoEnv(gym.Env)
#class MultiBeoEnv(MultiAgentEnv)





"""
Henghui: Why is this class used?
class MyCallbacks(DefaultCallbacks):
    def on_episode_end(
    self,
    *,
    worker: RolloutWorker,
    base_env: BaseEnv,
    policies: Dict[str, Policy],
    episode: Episode,
    env_index: int,
    **kwargs
    ):
    # Check if there are multiple episodes in a batch, i.e.
    # "batch_mode": "truncate_episodes".
        if worker.policy_config["batch_mode"] == "truncate_episodes":
        # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        episode.custom_metrics[base_env.vector_env.envs[0].name] = episode.total_reward
"""