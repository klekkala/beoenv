import numpy as np
from ray.rllib.models import ModelCatalog
import gym
from ray.rllib.policy.policy import Policy
from envs import wrap_custom, SingleBeoEnv
from models.testmodels import SingleAtariModel, SharedBackboneAtariModel, SharedBackbonePolicyAtariModel
from ray.rllib.algorithms.algorithm import Algorithm
ModelCatalog.register_custom_model("model", SingleAtariModel)
# Use the `from_checkpoint` utility of the Policy class:
# my_restored_policy = Policy.from_checkpoint("/lab/kiran/ckpts/trained/1.a_AirRaidNoFrameskip-v4_singlegame_full_e2e_PolicyNotLoaded_4stack/23_06_14_08_52_25/checkpoint")
# print(my_restored_policy)
beogym_path = "/lab/kiran/ray_results/PPO/PPO_BeoGym_d210d_00000_0_2023-06-23_22-20-23/checkpoint_003000/policies/default_policy"
# my_restored_policy = Algorithm.from_checkpoint(beogym_path)

# Algorithm
def eval(model_path, env_name, num_trials):
    env= wrap_custom(gym.make(env_name, full_action_space=True), framestack=True)
    my_restored_policy = Policy.from_checkpoint(model_path)
    for i in range(num_trials):
        episode_reward = 0
        done = False
        obs = env.reset()
        while not done:
            action,state,_ = my_restored_policy.compute_single_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        print(episode_reward)
            

def beogym_eval(model_path, env_name, num_trials):
    env_config={'env':'Wall_Street','data_path':'/home/tmp/kiran/'}
    env= SingleBeoEnv(env_config)
    my_restored_policy = Policy.from_checkpoint(model_path)
    # init_state = state = [np.zeros([256], np.float32) for _ in range(2)]
    init_state = state = my_restored_policy.get_initial_state()
    for i in range(num_trials):
        # episode_reward = 0
        # done = False
        # obs = env.reset()
        # state = my_restored_policy.get_initial_state()
        # reward = 0.0
        # action = 0
        # while not done:
        #     action, state, _ = my_restored_policy.compute_single_action(observation=obs, state=state, prev_action=action,prev_reward=reward, full_fetch=True)
        #     obs, reward, done, info = env.step(action)
        # print(info)
        # episode_reward += reward
        # if done:
        #     obs = env.reset()
        #     state = agent.get_policy().get_initial_state()
        #     reward = 0.0
        #     action = 0



        episode_reward = 0
        done = False
        obs = env.reset()
        a=0
        reward=0.0
        while True:
            a, state_out, _ = my_restored_policy.compute_single_action(obs, state, prev_action=a,prev_reward=reward)
            obs, reward, done, _ = env.step(a)
            episode_reward+=reward
            if done:
                obs = env.reset()
                state = init_state
                break
            else:
                state = state_out
        print(episode_reward)

# path="/lab/kiran/ckpts/trained/1.a_AirRaidNoFrameskip-v4_singlegame_full_e2e_PolicyNotLoaded_4stack/23_06_14_08_52_25/checkpoint"
# eval(path, 'AirRaidNoFrameskip-v4', 2)

beogym_eval(beogym_path, '', 10)
