import gym
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.env import BaseEnv
from typing import Dict, Tuple
from ray.rllib.policy.policy import Policy

##SingleTask, MultiTask, MultiEnv classes and their related classes/functions

class CusEnv(gym.Env):
    def __init__(self,env_config):
        self.env = gym.make("ALE/BeamRider-v5", full_action_space=True)
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8) #self.env.observation_space

    def reset(self, seed=None, options=None):
        obs,info = self.env.reset()
        return cv2.resize(obs, (84, 84)), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = cv2.resize(obs, (84, 84))
        #cv2.imwrite('obs.png', obs)
        return obs, reward, terminated, truncated, info




class MultiTaskEnv(gym.Env): 
    def __init__(self, env_config):
        for i in range(len(envs)):    
            if env_config.worker_index%9==i:
                self.env = gym.make(envs[i], full_action_space=True)
                self.name= envs[i]
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
        #res = tuple((cv2.resize(temp[0], (84, 84)),temp[1],temp[2],temp[3],temp[4]))
        return tuple(temp)



class MultiSync(MultiAgentEnv):

    def __init__(self,num):
        self.agents=[]
        for i in range(len(envs)):
            self.agents.append(gym.make(envs[i], full_action_space=True))
        self.terminateds = set()
        self.truncateds = set()
        self.action_space = gym.spaces.Discrete(18)
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)
        self.resetted = False

    def reset(self, *, seed=None, options=None):
        res={}
        info={}
        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()
        for i in range(len(envs)):
            temp,info = self.agents[i].reset()
            temp = cv2.resize(temp, (84, 84))
            res[i]=temp
            info[i] = info
        return res,info

    def step(self, action_dict):
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}
        for i, action in action_dict.items():
            temp = self.agents[i].step(action)
            temp=list(temp)
            temp[0] = cv2.resize(temp[0], (84, 84))
            obs[i], rew[i], terminated[i], truncated[i], info[i] = temp
            if terminated[i]:
                self.terminateds.add(i)
            if truncated[i]:
                self.truncateds.add(i)

        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = len(self.truncateds) == len(self.agents)
        return obs, rew, terminated, truncated, info




class Beogym(MultiAgentEnv):

    def __init__(self,num):
        self.agents=[]
        for i in range(len(envs)):
            self.agents.append(gym.make(envs[i], full_action_space=True))
        self.terminateds = set()
        self.truncateds = set()
        self.action_space = gym.spaces.Discrete(18)
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), np.uint8)
        self.resetted = False

    def reset(self, *, seed=None, options=None):
        res={}
        info={}
        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()
        for i in range(len(envs)):
            temp,info = self.agents[i].reset()
            temp = cv2.resize(temp, (84, 84))
            res[i]=temp
            info[i] = info
        return res,info

    def step(self, action_dict):
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}
        for i, action in action_dict.items():
            temp = self.agents[i].step(action)
            temp=list(temp)
            temp[0] = cv2.resize(temp[0], (84, 84))
            obs[i], rew[i], terminated[i], truncated[i], info[i] = temp
            if terminated[i]:
                self.terminateds.add(i)
            if truncated[i]:
                self.truncateds.add(i)

        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = len(self.truncateds) == len(self.agents)
        return obs, rew, terminated, truncated, info




"""
class CarlaSingle(gym.Env)
class CarlaMulti(gym.Env)
class CarlaMultiSync(MultiAgentEnv)
"""




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