import gym
import rlbench
import gymnasium
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
import string
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.wrappers.atari_wrappers import FrameStack, ScaledFloatFrame, WarpFrame, NoopResetEnv, MonitorEnv, MaxAndSkipEnv, FireResetEnv
import ray
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from IPython import embed
#import graph_tool.all as gt
from ray import air, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from gym import spaces
from ray.rllib.utils.images import rgb2gray, resize
# import vc_models
# from vc_models.models.vit import model_utils
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

#from beogym.beogym import BeoGym
##SingleTask, MultiTask, MultiEnv classes and their related classes/functions


def _convert_image_to_rgb(image):
    return image.convert("RGB")

_transform = Compose([
            Resize(224, interpolation=BICUBIC),
            CenterCrop(224),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

import torchvision.transforms as T
_transforms = T.Compose([T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()])


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

    #env = ScaledFloatFrame(env)  # TODO: use for dqn?
    # env = ClipRewardEnv(env)  # reward clipping is handled by policy eval
    # 4x image framestacking.
    if framestack is True:
        env = FrameStack(env, 4)
    else:
        env = FrameStack(env, 1)
    return env



from ray.rllib.utils.annotations import override

atari_rewards={"AirRaidNoFrameskip-v4": 8000, "AssaultNoFrameskip-v4": 883,"BeamRiderNoFrameskip-v4": 1400, "CarnivalNoFrameskip-v4": 4384,"DemonAttackNoFrameskip-v4": 415, "NameThisGameNoFrameskip-v4": 6000,"PhoenixNoFrameskip-v4":4900,"RiverraidNoFrameskip-v4": 8400,"SpaceInvadersNoFrameskip-v4":500}
atari_envs = ["AirRaidNoFrameskip-v4", "AssaultNoFrameskip-v4", "BeamRiderNoFrameskip-v4", "CarnivalNoFrameskip-v4", "DemonAttackNoFrameskip-v4", "NameThisGameNoFrameskip-v4", "PhoenixNoFrameskip-v4", "RiverraidNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4"]



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
        env_keys = list(episode.agent_rewards.keys())
        for each_id in range(len(env_keys)):
            episode.custom_metrics[base_env.envs[0].envs[env_keys[each_id][0]]] = episode.agent_rewards[(env_keys[each_id][0], env_keys[each_id][1])]

from PIL import Image

class SingleAtariEnv(gym.Env):
    def __init__(self, env_config):

        self.env = wrap_custom(gym.make(env_config['env'], full_action_space=env_config['full_action_space']), framestack=env_config['framestack'])

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    """
    def step(self, action):
        res = ''.join(random.choices(string.ascii_lowercase +
                             string.digits, k=7))
        ab = self.env.step(action)
        obs = ab[0][:,:,0]
        im = Image.fromarray(obs)
        im.save("/lab/kiran/beamrider_rllib_imgs/" + res + ".png")
        return ab
    """



class MultiAtariEnv(MultiAgentEnv):

        def __init__(self, env_config):
            self.agents=[]
            self.envs = env_config['envs']
            for i in range(len(env_config['envs'])):
                print(env_config['envs'][i])
                env=wrap_custom(gym.make(env_config['envs'][i], full_action_space=False))
                self.agents.append(env)
            self.dones = set()
            #This is a bad habbit. change it.
            self.action_space = self.agents[-1].action_space
            self.observation_space = self.agents[-1].observation_space
            print(self.observation_space)
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



atari = {'single': SingleAtariEnv, 'multi': MultiAtariEnv}

#Henghui Todo
#class SingleBeoEnv((gym.Env))
class SingleBeoEnv(gym.Env):
    def __init__(self,env_config):
        import graph_tool.all as gt
        from beogym.beogym import BeoGym
        self.env = BeoGym({'city':env_config['env'], 'data_path':env_config['data_path']})
        self.action_space = self.env.action_space
        self.e_model = env_config['e_model']
        self.observation_space = self.env.observation_space
        self.tran=None
        if "clip" in self.e_model:
            self.observation_space = gym.spaces.Dict(
                {"obs": gym.spaces.Box(low=-5.0, high=5.0, shape=(3, 224, 224), dtype=np.float32),
                "aux": gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)})
        elif 'r3m' in self.e_model or 'mvp' in self.e_model:
            self.observation_space = gym.spaces.Dict(
                {"obs": gym.spaces.Box(low=-5.0, high=5.0, shape=(3, 224, 224), dtype=np.float32),
                "aux": gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)})
        elif 'vc1' in self.e_model:
            _,_,self.tran,_ = model_utils.load_model(model_utils.VC1_BASE_NAME)
            self.observation_space = gym.spaces.Dict(
                {"obs": gym.spaces.Box(low=-5.0, high=5.0, shape=(3, 224, 224), dtype=np.float32),
                "aux": gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)})
    
    def reset(self, seed=None, options=None):
        step_output = self.env.reset()
        if "clip" in self.e_model:
            step_output['obs'] = _transform(Image.fromarray(step_output['obs'])).numpy()
        elif 'r3m' in self.e_model or 'mvp' in self.e_model:
            step_output['obs'] = _transforms(Image.fromarray(step_output['obs'])).reshape(3, 224, 224).numpy()
        elif 'vc1' in self.e_model:
            step_output['obs'] = self.tran(Image.fromarray(step_output['obs'])).reshape(3, 224, 224).numpy()
        return step_output


    def step(self, action):
        step_output = self.env.step(action)
        if "clip" in self.e_model:
            step_output[0]['obs'] = _transform(Image.fromarray(step_output[0]['obs'])).numpy()
        elif 'r3m' in self.e_model or 'mvp' in self.e_model:
            step_output[0]['obs'] = _transforms(Image.fromarray(step_output[0]['obs'])).reshape(3, 224, 224).numpy()
        elif 'vc1' in self.e_model or 'mvp' in self.e_model:
            step_output[0]['obs'] = self.tran(Image.fromarray(step_output[0]['obs'])).reshape(3, 224, 224).numpy()
        # elif 'vc-1' in self.e_model:
        #     step_output['obs'] = vc_transform(Image.fromarray(step_output['obs'])).reshape(3, 224, 224).numpy()
        return step_output


class MultiBeoEnv(MultiAgentEnv):

    def __init__(self, envs):
        self.agents=[]
        self.envs = envs
        for i in range(len(self.envs)):
            print(self.envs)
            self.agents.append(SingleBeoEnv({'city':[self.envs[i]], 'data_path':env_config['data_path']}))
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



beogym = {'single': SingleBeoEnv, 'multi': MultiBeoEnv}



class SingleColoEnv(gym.Env):
    def __init__(self, env_config):
        self.step_count = 0
        self.env = gymnasium.make(env_config['env'], render_mode="rgb_array")
        #self.action_space = gym.spaces.Box(-1.0, 1.0, (8,), np.float32)
        self.action_space = gym.spaces.Box(
            np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.0]),
            np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.04]),
            dtype=np.float32)
        # self.observation_space = self.env.observation_space
        if env_config['env'] == 'rlbench/slide_block_to_target-vision-v0':
            self.observation_space = gym.spaces.Box(0, 255, (3, 84, 84), np.uint8)
            #self.observation_space = gym.spaces.Dict({'front_rgb': gym.spaces.Box(0, 255, (3, 84, 84), np.uint8), 'gripper_joint_positions': gym.spaces.Box(-np.inf, np.inf, (2,), np.float32), 'gripper_open': gym.spaces.Box(-np.inf, np.inf, (1,), np.float32), 'gripper_pose': gym.spaces.Box(-np.inf, np.inf, (7,), np.float32), 'gripper_touch_forces': gym.spaces.Box(-np.inf, np.inf, (6,), np.float32), 'joint_forces': gym.spaces.Box(-np.inf, np.inf, (7,), np.float32), 'joint_positions': gym.spaces.Box(-np.inf, np.inf, (7,), np.float32), 'joint_velocities': gym.spaces.Box(-np.inf, np.inf, (7,), np.float32), 'left_shoulder_rgb': gym.spaces.Box(0, 255, (128, 128, 3), np.uint8), 'right_shoulder_rgb': gym.spaces.Box(0, 255, (128, 128, 3), np.uint8), 'task_low_dim_state': gym.spaces.Box(-np.inf, np.inf, (6,), np.float32), 'wrist_rgb': gym.spaces.Box(0, 255, (128, 128, 3), np.uint8)})
    
        # self.observation_space = gym.spaces.Dict({'front_rgb': gym.spaces.Box(0, 255, (3, 128, 128), np.uint8), 'gripper_joint_positions': gym.spaces.Box(-np.inf, np.inf, (2,), np.float32), 'gripper_open': gym.spaces.Box(-np.inf, np.inf, (1,), np.float32), 'gripper_pose': gym.spaces.Box(-np.inf, np.inf, (7,), np.float32), 'gripper_touch_forces': gym.spaces.Box(-np.inf, np.inf, (6,), np.float32), 'joint_forces': gym.spaces.Box(-np.inf, np.inf, (7,), np.float32), 'joint_positions': gym.spaces.Box(-np.inf, np.inf, (7,), np.float32), 'joint_velocities': gym.spaces.Box(-np.inf, np.inf, (7,), np.float32), 'left_shoulder_rgb': gym.spaces.Box(0, 255, (128, 128, 3), np.uint8), 'right_shoulder_rgb': gym.spaces.Box(0, 255, (128, 128, 3), np.uint8), 'task_low_dim_state': gym.spaces.Box(-np.inf, np.inf, (6,), np.float32), 'wrist_rgb': gym.spaces.Box(0, 255, (128, 128, 3), np.uint8)})
        print(env_config['env'])
        if env_config['env'] == 'rlbench/open_drawer-vision-v0':
            self.observation_space = gym.spaces.Box(0, 255, (3, 84, 84), np.uint8)
            #self.observation_space = gym.spaces.Dict({'front_rgb': gym.spaces.Box(0, 255, (3, 84, 84), np.uint8), 'gripper_joint_positions': gym.spaces.Box(-np.inf, np.inf, (2,), np.float32), 'gripper_open': gym.spaces.Box(-np.inf, np.inf, (1,), np.float32), 'gripper_pose': gym.spaces.Box(-np.inf, np.inf, (7,), np.float32), 'gripper_touch_forces': gym.spaces.Box(-np.inf, np.inf, (6,), np.float32), 'joint_forces': gym.spaces.Box(-np.inf, np.inf, (7,), np.float32), 'joint_positions': gym.spaces.Box(-np.inf, np.inf, (7,), np.float32), 'joint_velocities': gym.spaces.Box(-np.inf, np.inf, (7,), np.float32), 'left_shoulder_rgb': gym.spaces.Box(0, 255, (128, 128, 3), np.uint8), 'right_shoulder_rgb': gym.spaces.Box(0, 255, (128, 128, 3), np.uint8), 'task_low_dim_state': gym.spaces.Box(-np.inf, np.inf, (57,), np.float32), 'wrist_rgb': gym.spaces.Box(0, 255, (128, 128, 3), np.uint8)})
        if env_config['env'] == 'rlbench/reach_target-vision-v0':
            
            self.observation_space = gym.spaces.Box(0, 255, (3, 84, 84), np.uint8)
            print("*****************Observtion****", self.observation_space)
            print("**********************action********", self.action_space)
            #self.observation_space = gym.spaces.Dict({'front_rgb': gym.spaces.Box(0, 255, (3, 84, 84), np.uint8), 'gripper_joint_positions': gym.spaces.Box(-np.inf, np.inf, (2,), np.float32), 'gripper_open': gym.spaces.Box(-np.inf, np.inf, (1,), np.float32), 'gripper_pose': gym.spaces.Box(-np.inf, np.inf, (7,), np.float32), 'gripper_touch_forces': gym.spaces.Box(-np.inf, np.inf, (6,), np.float32), 'joint_forces': gym.spaces.Box(-np.inf, np.inf, (7,), np.float32), 'joint_positions': gym.spaces.Box(-np.inf, np.inf, (7,), np.float32), 'joint_velocities': gym.spaces.Box(-np.inf, np.inf, (7,), np.float32), 'left_shoulder_rgb': gym.spaces.Box(0, 255, (128, 128, 3), np.uint8), 'right_shoulder_rgb': gym.spaces.Box(0, 255, (128, 128, 3), np.uint8), 'task_low_dim_state': gym.spaces.Box(-np.inf, np.inf, (3,), np.float32), 'wrist_rgb': gym.spaces.Box(0, 255, (128, 128, 3), np.uint8)})
    
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        #obs['front_rgb'] = obs['front_rgb'].reshape(3, 84, 84)
        #obs['front_rgb'] = np.array(Image.fromarray(obs['front_rgb'].astype(np.uint8)).resize((84, 84)))
        obs['front_rgb'] = np.moveaxis(obs['front_rgb'], -1, 0)
        #obs['front_rgb'] = _transforms(Image.fromarray(obs['front_rgb'])).reshape(3, 224, 224).numpy()
        self.step_count = 0
        return obs['front_rgb']

    def step(self, action):
        obs, reward, terminate, _, _ = self.env.step(action)
        #print(obs)
        #from PIL import Image
        #im = Image.fromarray(obs['front_rgb'])
        #im.save("test.jpeg")
        
        #obs['front_rgb'] = np.array(Image.fromarray(obs['front_rgb'].astype(np.uint8)).resize((84, 84)))
        
        self.step_count += 1
        #print(reward)
        
        if self.step_count > 200:
            terminate = True
        


        # obs['front_rgb'] = _transforms(Image.fromarray(obs['front_rgb'])).reshape(3, 224, 224).numpy()
        obs['front_rgb'] = np.moveaxis(obs['front_rgb'], -1, 0)
        return obs['front_rgb'], reward, terminate, _

colosseum = {'single': SingleColoEnv}
