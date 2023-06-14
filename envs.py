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
import random
from ray.rllib.env.wrappers.atari_wrappers import FrameStack, WarpFrame, NoopResetEnv, MonitorEnv, MaxAndSkipEnv, FireResetEnv
#from beogym.beogym import BeoGym
##SingleTask, MultiTask, MultiEnv classes and their related classes/functions


def wrap_custom(env, dim=84, framestack=True):
    """Configure environment for DeepMind-style Atari.

    Note that we assume reward clipping is done outside the wrapper.

    Args:
        env: The env object to wrap.
        dim: Dimension to resize observations to (dim x dim).
        framestack: Whether to framestack observations.
    """
    env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    if env.spec is not None and "NoFrameskip" in env.spec.id:
        env = MaxAndSkipEnv(env, skip=4)

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, dim)
    # env = ScaledFloatFrame(env)  # TODO: use for dqn?
    # env = ClipRewardEnv(env)  # reward clipping is handled by policy eval
    # 4x image framestacking.
    if framestack is True:
        env = FrameStack(env, 4)
    return env




class ParellelCallbacks(DefaultCallbacks):
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



class MultiCallbacks(DefaultCallbacks):
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


class SingleAtariEnv(gym.Env):
    def __init__(self, env_config):
        print("I'm usoing single env lksjdflkasklfdkaskdlfj", env_config['env'])
        self.env = wrap_custom(gym.make(env_config['env'], full_action_space=True), framestack=True)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        print(self.observation_space)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

class ParellelAtariEnv(gym.Env): 
    def __init__(self, env_config):
        self.atari_rewards={"AirRaidNoFrameskip-v4":27275, "AssaultNoFrameskip-v4": 724,"BeamRiderNoFrameskip-v4": 1588, "CarnivalNoFrameskip-v4": 3500,"DemonAttackNoFrameskip-v4": 2080,"NameThisGameNoFrameskip-v4": 7250,"PhoenixNoFrameskip-v4":6010,"RiverraidNoFrameskip-v4": 8600,"SpaceInvadersNoFrameskip-v4":1145}
        for i in range(len(env_config['envs'])):
            print(env_config.worker_index, env_config['envs'])
            if env_config.worker_index%len(env_config['envs'])==i:
                self.env = wrap_custom(gym.make(env_config['envs'][i], full_action_space=True))
                self.name= env_config['envs'][i]
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        print(self.observation_space)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


class MultiAtariEnv(MultiAgentEnv):

        def __init__(self, envs):
            self.agents=[]
            self.envs = envs['envs']
            for i in range(len(envs['envs'])):
                print(envs['envs'][i])
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


    def step(self, action):
        return self.env.step(action)


class ParellelBeoEnv(gym.Env): 
    def __init__(self, envs):
        for i in range(len(envs)):    
            if env_config.worker_index%len(envs)==i:
                self.env = BeoGym({'city':[envs[i]]})
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
            self.agents.append(BeoGym({'city':[self.envs[i]]}))
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


beogym = {'single': SingleBeoEnv, 'parellel': ParellelBeoEnv, 'multi': MultiBeoEnv}



