import time
import sys
from PIL import Image

from RES_VAE import VAE as VAE

from agent import Agent
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from MplCanvas import MplCanvas
from data_helper import dataHelper, coord_to_sect, coord_to_filename


import numpy as np
import config as app_config
import math, cv2, h5py, argparse, csv, copy, time, os, shutil
from pathlib import Path

import argparse
import ray
from ray.rllib.utils.annotations import override
from ray import air, tune
from ray.rllib.algorithms import ppo
from ray.tune.registry import register_env
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
#from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
#from Vo import FullyConnectedNetwork as TorchFC
#from ray.rllib.models.torch.visionnet import VisionNetwork as TorchFC
from Zero import ZeroNetwork as TorchZero
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
#from stable_baselines3.common.env_checker import check_env
#from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3.common.env_util import make_vec_env
#from stable_baselines3.common.evaluation import evaluate_policy

#from stable_baselines3 import PPO, A2C


class BeoGym(gym.Env):
    
    def __init__(self, env_config=None):
        # Test comment by Antonio Revilla
        turning_range = 30 if app_config.PANO_IMAGE_MODE else 60

        super(BeoGym, self).__init__()

        self.dh = dataHelper("data/test.csv", app_config.PANO_HOV)
        self.agent = Agent(self.dh, turning_range, app_config.PANO_IMAGE_RESOLUTION, app_config.PANO_IMAGE_MODE)

        self.action_space = spaces.Discrete(4)
        #self.observation_space = spaces.Box(low = 0, high = 255, shape = (*self.agent.view_res, 3), dtype= np.uint8)
        self.observation_space = gym.spaces.Dict({"obs": spaces.Box(low = 0, high = 255, shape = (208, 416, 3), dtype= np.uint8), "aux": spaces.Box(low = -1.0, high = 1.0,shape = (5,), dtype= np.float32)})
        self.seed(1)

        # Settings:
        self.game = 'courier'
        self.max_steps = 1000
        self.curr_step = 0
        self.min_radius_meters = 5 # The radius around the goal that can be considered the goal itself.
        self.max_radius_meters = 20 # The outer radius around the goal where the rewards kicks in.
        self.min_distance_reached = 15 # The closest distance the agent has been so far to the goal.
        self.goal_reward = 100
        #self.courier_goal = (6.787983881738484, -39.73080249032006)
        # Sample goal:
        # if 'a' is not None:
        #     self.courier_goal = (6.787983881738484, -39.73080249032006)
        # else:
        #     # Pick a random goal.
        while True:
            self.courier_goal = self.dh.sample_location()
            self.initial_distance_to_goal = self.dh.distance_to_goal(self.agent.agent_pos_curr, self.courier_goal)
            # Make sure the goal is not within the max_radius_meters to the agent's current position. Also, the sampled goal
            # should not be the same as the agent's current position:
            if (self.initial_distance_to_goal > self.max_radius_meters and self.courier_goal != self.agent.agent_pos_curr):
                break
        #
        #     print("Goal is None. Sampled a location as goal: ", self.courier_goal)

        # Logging: to be implemented.

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        info = {}
        aux=[2*(self.agent.agent_pos_curr[0] +100)/200 - 1,2*(self.agent.agent_pos_curr[1] +100)/200 - 1, self.agent.curr_angle/360,2*(self.courier_goal[0] +100)/200 - 1,2*(self.courier_goal[1] +100)/200 - 1]
        return {'obs':self.agent.reset(),'aux':np.array(aux)}, info

    
    def step(self, action):
        done = False
        info = {}
        print("Step with action: ", action)
        self.agent.take_action(action)
        # Keep track of the number of steps in an episode:
        self.curr_step += 1
        
        if app_config.SAVE_IMAGE_PATH:
            # Image
            action_name = ""
            if action == 0:
                action_name = "straight"
            elif action == 1:
                action_name = "back"
            else:
                angle = self.agent.action_angle_map[action]
                action_name = f"turn_{angle}_degrees"

            filename = f"step_{self.curr_step}_action_{action_name}.{app_config.IMAGE_SOURCE_EXTENSION}"
            cv2.imwrite(f"{app_config.IMAGE_PATH_DIR}/{filename}", self.agent.curr_view)

        if (self.curr_step >= self. max_steps):
            
            done = True
            info['time_limit_reached'] = True

        # print("comparison: ", self.game == 'courier')

        # Three different type of games: https://arxiv.org/pdf/1903.01292.pdf
        if self.game == 'courier':
            reward, terminated = self.courier_reward_fn()
        elif self.game == 'coin':
            reward, terminated = self.coin_reward_fn()
        elif self.game == 'instruction':
            reward, terminated = self.instruction_reward_fn()
        aux=[2*(self.agent.agent_pos_curr[0] +100)/200 - 1,2*(self.agent.agent_pos_curr[1] +100)/200 - 1, self.agent.curr_angle/360,2*(self.courier_goal[0] +100)/200 - 1,2*(self.courier_goal[1] +100)/200 - 1]
        return {'obs':self.agent.curr_view,'aux':np.array(aux)}, reward, terminated, done, info

    def render(self, mode='human', steps=100):

        # Renders initial plot
        window_name = 'window'
        # print("POS", self.agent.agent_pos_curr, "PREV", self.agent.agent_pos_prev, "CUR_ANGLE", self.agent.curr_angle)
        if mode != 'random':
            self.dh.update_plot(self.agent.agent_pos_curr, self.courier_goal)


        # img = cv2.imshow(self.agent.curr_view)
            cv2.imshow(window_name, self.agent.curr_view)
        # A loop of random actions taken for the amount of steps specified
        if mode == "random":
            self.random_mode(steps, window_name=window_name)

        if mode == "spplanner":
            self.shortest_path_mode(window_name=window_name)

        # This infinite loop is in place to allow keyboard inputs until the program is manually terminated
        while mode == "human":
            # print("POS", self.agent.agent_pos_curr, "PREV", self.agent.agent_pos_prev, "CUR_ANGLE",
            #       self.agent.curr_angle)
            # Wait for keyboard input
            key = cv2.waitKeyEx(0)
            # Map keyboard inputs to your actions
            if key == app_config.KeyBoardActions.LEFT:
                print("Left")
                self.agent.go_left()
            elif key == app_config.KeyBoardActions.RIGHT:
                print("Right")
                self.agent.go_right()
            elif key == app_config.KeyBoardActions.FORWARD:
                print("Forward")
                self.agent.go_straight()
            elif key == app_config.KeyBoardActions.REVERSE:
                print("Reverse")
                self.agent.go_back()
            elif key == app_config.KeyBoardActions.KILL_PROGRAM:
                print("Killing Program")
                cv2.destroyAllWindows()
                break

            # If input key is an action related to moving in a certain direction, update the plot based on the action
            # already taken and update the window.

            if key in app_config.DIRECTION_ACTIONS:
                self.dh.update_plot(self.agent.agent_pos_curr, self.courier_goal)
                # img = cv2.imshow(self.agent.curr_view)
                cv2.imshow(window_name, self.agent.curr_view)


            # Update Bird's-Eye-View graph
            graph = self.dh.bird_eye_view(self.agent.agent_pos_curr, r)
            if graph is None:
                raise EnvironmentError("Graph is None")
            self.dh.draw_bird_eye_view(self.agent.agent_pos_curr, r, graph, self.agent.curr_angle)

    def shortest_path_mode(self, window_name=None):
        print("Traversing shortest path to goal")
        shortest_path = self.dh.getShortestPathNodes(self.agent.agent_pos_curr, self.courier_goal)
        print(shortest_path)

        if window_name:
            cv2.imshow(window_name, self.agent.curr_view)
        last_angle = 0
        for i in range(len(shortest_path)):
            angle = 0
            minAngle = 500
            if i != len(shortest_path) - 1:
                agl = self.dh.fix_angle(self.dh.get_angle(shortest_path[i], shortest_path[i + 1]))
                for j in self.dh.camera_angle_map.values():
                    if abs(self.dh.get_distance(agl, j)) < minAngle:
                        angle = j
                        minAngle = abs(self.dh.get_distance(agl, j))
            self.agent.update_agent(shortest_path[i], self.agent.agent_pos_curr, last_angle)
            # self.agent.update_agent(shortest_path[i], self.agent.agent_pos_curr, self.agent.curr_angle)
            self.dh.update_plot(self.agent.agent_pos_curr, self.courier_goal)
            if window_name:
                cv2.imshow(window_name, self.agent.curr_view)
                key = cv2.waitKey(1000)
            while last_angle != angle:
                dis = self.dh.get_distance(last_angle, angle)
                if abs(dis)<=15:
                    last_angle = angle
                else:
                    if abs(self.dh.get_distance(last_angle+15, angle)) < abs(self.dh.get_distance(last_angle-15, angle)):
                        last_angle += 15
                    else:
                        last_angle -= 15
                self.agent.update_agent(self.agent.agent_pos_curr, self.agent.agent_pos_curr, last_angle)
                if window_name:
                    cv2.imshow(window_name, self.agent.curr_view)
                    key = cv2.waitKey(1000)
                last_angle = angle

    def random_mode(self, steps, window_name=None):
        start = time.time()
        for i in range(steps):
            # 7 since there are 6 actions that can be taken. (0-7 since 0 is inclusive and 7 is exclusive )
            self.step(np.random.randint(low=0, high=7))
            #self.dh.update_plot(self.agent.agent_pos_curr, self.agent.agent_pos_prev, self.agent.curr_angle)
            #if window_name:
                #cv2.imshow(window_name, self.agent.curr_view)
                #time.sleep(2)
        end = time.time()
        print(f"----Start: {start}. End: {end}. Difference: {end - start}----")


    # Headless version of the render method
    def render_headless(self, mode="human", steps=0):
        # Renders initial plot
        window_name = 'window'
        # print("POS", self.agent.agent_pos_curr, "PREV", self.agent.agent_pos_prev, "CUR_ANGLE", self.agent.curr_angle)
        self.dh.update_plot(self.agent.agent_pos_curr, self.courier_goal)

        # A loop of random actions taken for the amount of steps specified
        if mode == "random":
            self.random_mode(steps, window_name=None)

        if mode == "spplanner":
            self.shortest_path_mode(window_name=None)

        if mode == "comp":
            action = None
            action_choices = [x for x in range(0, 7)]
            action_choices.append('k')
            # Action is None check is for the initial start of the mode
            while action != 'k' or action is None:
                action = input("Enter an action 0-6, Enter 'k' to kill the program.\n")
                if action == 'k':
                    break
                else:
                    action = int(action)

                if action not in action_choices:
                    print("Received action: ", action)
                    raise EnvironmentError("Action choices must be 0-6 or 'k'.")
                self.step(action)                
                self.dh.update_plot(self.agent.agent_pos_curr, self.courier_goal)
                # time.sleep(2)

        # This infinite loop is in place to allow keyboard inputs until the program is manually terminated
        while mode == "human":
            # print("POS", self.agent.agent_pos_curr, "PREV", self.agent.agent_pos_prev, "CUR_ANGLE",
            #       self.agent.curr_angle)
            # Wait for input
            key = input("Enter operation (Left, Right, Forward, Reverse, Quit): ").lower()

            # Map keyboard inputs to your actions
            if key == app_config.InputActions.LEFT:
                print("Left")
                self.agent.go_left()
            elif key == app_config.InputActions.RIGHT:
                print("Right")
                self.agent.go_right()
            elif key == app_config.InputActions.FORWARD:
                print("Forward")
                self.agent.go_straight()
            elif key == app_config.InputActions.REVERSE:
                print("Reverse")
                self.agent.go_back()
            elif key == app_config.InputActions.KILL_PROGRAM:
                print("Killing Program")
                break

            # If input key is an action related to moving in a certain direction, update the plot based on the action
            # already taken and update the window.
            if key in app_config.DIRECTION_ACTIONS:
                self.dh.update_plot(self.agent.agent_pos_curr, self.courier_goal)

    # Copied this from CarlaEnv
    
    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    # Implementation will be similar to this file: https://github.com/deepmind/streetlearn/blob/master/streetlearn/python/environment/courier_game.py
    
    def courier_reward_fn(self, distance = None):

        reward = 0
        found_goal = False

        # Does not give reward if the agent visited old locations:
        if self.agent.agent_pos_curr in self.dh.visited_locations:

            return reward, found_goal

        # If distance is not None then we are in testing mode:
        if distance is None:

            distance_to_goal = self.dh.distance_to_goal(self.agent.agent_pos_curr, self.courier_goal)

            # Add current location to visited locations list:
            self.dh.visited_locations.add(self.agent.agent_pos_curr)

        else:
            
            distance_to_goal = distance

        # The goal is reached:
        if distance_to_goal < self.min_radius_meters:
            reward = self.goal_reward
            found_goal = True
        else:
            if distance_to_goal < self.max_radius_meters:
                print("max_radius_meters: ", self.max_radius_meters)
                print("min_distance_reached: ", self.min_distance_reached)
                # Only rewards the agent if the agent has decreased the closest distance so far to the goal:
                if distance_to_goal < self.min_distance_reached:
                    
                    # Reward is linear function to the distance to goal:
                    reward = (self.goal_reward *
                        (self.max_radius_meters - distance_to_goal) /
                        (self.max_radius_meters - self.min_radius_meters))

                    self.min_distance_reached = distance_to_goal
        
        return reward, found_goal

    def coin_reward_fn(self):
        
        pass

    def instruction_reward_fn(self):
        
        pass


def get_gps_data():
    """
    Read GPS data from a CSV file
    """

    csv_file = f"{app_config.GPS_DATA_PATH}"
    image_data = dict()
    data_image = dict()
    with open(csv_file, newline='') as csvfile:
        gps_reader = csv.reader(csvfile)
        for row in gps_reader:
            # print(row)
            image_name = row[2]
            if image_data.get(image_name):
                raise ValueError("Duplicate images")

            coord_tuple = (row[0], row[1],)
            coord_str = f"{row[0]},{row[1]}"
            image_data[image_name] = coord_tuple
            data_image[coord_str] = image_name
    # print(image_data.keys())
    return image_data, data_image


if __name__ == "__main__":
    # Load the hdf5 files into a global variable
    global coord_to_sect

    pano_mode = app_config.PANO_IMAGE_MODE
    headless_mode = app_config.HEADLESS_MODE
    #mode = app_config.INTERACTION_MODE
    #mode = app_config.ConfigModes.HUMAN
    mode = app_config.ConfigModes.SPPLANNER
    hov_val = app_config.PANO_HOV

    if app_config.SAVE_IMAGE_PATH:
        if not os.path.isabs(app_config.IMAGE_PATH_DIR):
            # Remove directory
            if os.path.exists(app_config.IMAGE_PATH_DIR) and os.path.isdir(app_config.IMAGE_PATH_DIR):
                shutil.rmtree(app_config.IMAGE_PATH_DIR)

            # Create directory
            Path(app_config.IMAGE_PATH_DIR).mkdir(parents=True, exist_ok=True)

    if not pano_mode:
        f = h5py.File("../hd5_files/coordinate_file_map.hdf5")
        for key in f.keys():
            group = f[key]
            values = group.values()
            for dataset in values:
                file_path = dataset[()].decode()
                coord_to_sect[key] = file_path
        
        gps_map, coord_to_filename_map = get_gps_data()
        # Deep copy global dictionary
        #for key in coord_to_filename_map:
        #    coord_to_filename[key] = coord_to_filename_map[key]

        #gps_map, coord_to_filename_map = get_gps_data()

    csv_file = "data/test.csv"
    # dh = dataHelper(csv_file,app_config.PANO_HOV)
    # turning_range = 30 if pano_mode else 60
    # view_res = app_config.PANO_IMAGE_RESOLUTION
    # agent = Agent(dh, turning_range, view_res, pano_mode)
    # seed = 1

    # goal = None # Sample from dataset.

    goal = (3245.662888495351, 26233.404940228444)
    goal = (6.787983881738484, -39.73080249032006)
    # env_config={'agent':agent, 'data_helper': dh, 'goal':goal, 'game':'courier', 'max_steps':1000,'seed':seed}
    #env = BeoGym(agent, dh, goal, seed = seed)
    # def create_env():
    #     # env = BeoGym(env_config)
    #     return BeoGym()
    # # env_creator = lambda config: create_env()
    # env_creator = lambda config: create_env()
    # register_env('Bgm', lambda config: BeoGym())
    # bgm = gym.make("BeoGym", env=env_creator)
    # env = gym.make("BeoGym", apply_api_compatibility=True)
    #check_env(env) # Checking to see if the custom env is compatible with stable-baselines3

    # Training:
    #n_envs = 4 # Number of environments
    #train_env = make_vec_env(env, n_envs= n_envs)

    # Environment for evaluation:
    #eval_eps = 100
    #eval_env = BeoGym(agent, dh, goal, seed = seed)

    #n_expriments = 3 # Number of experiments.
    #train_steps = 10000

    #model = A2C('MlpPolicy', train_env, verbose = 1)

    #rewards = []

    #for experiment in range(n_expriments):
    #    train_env.reset()
    #    model.learn(total_timesteps= train_steps)
    #    mean_reward, _ = evaluate_policy(model,eval_env, n_eval_episodes=eval_eps)
    #    rewards.append(mean_reward)

    #action = env.action_space.sample()
    #action = 2
    #obs, reward, done, info = env.step(action)
    #print("\n")
    #print("Reward: ", reward)
    #print("Done: ", done)
    #env.render()

    r = 128
    steps = 100


    torch, nn = try_import_torch()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
    )
    parser.add_argument(
        "--framework",
        choices=["torch"],
        default="torch",
    )
    parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test: --stop-reward must "
             "be achieved within --stop-timesteps AND --stop-iters.",
    )
    parser.add_argument(
        "--stop-iters", type=int, default=50, help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
    )
    parser.add_argument(
        "--stop-reward", type=float, default=0.1, help="Reward at which we stop training."
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Run without Tune using a manual train loop instead. In this case,"
             "use PPO without grid search and no TensorBoard.",
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )



    class TorchCustomModel(TorchModelV2, nn.Module):
        """Example of a PyTorch custom model that just delegates to a fc-net."""

        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            TorchModelV2.__init__(
                self, obs_space, action_space, num_outputs, model_config, name
            )
            nn.Module.__init__(self)

            self.torch_sub_model = TorchZero(
                obs_space, action_space, num_outputs, model_config, name
            )

        def forward(self, input_dict, state, seq_lens):
            # input_dict["obs"]["obs"] = input_dict["obs"]["obs"].float()
            fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
            return fc_out, []

        def value_function(self):
            return torch.reshape(self.torch_sub_model.value_function(), [-1])

    if headless_mode:
        env.render_headless(mode=mode, steps=steps)
    else:
        # env.render(mode="random", steps=100)
        print('we are here')
        # env.render(mode=mode, steps=steps)
        args = parser.parse_args()
        ray.init(local_mode=args.local_mode)
        ModelCatalog.register_custom_model(
            "my_model", TorchCustomModel
        )

        config = (
            get_trainable_cls(args.run)
                .get_default_config()
                .environment(BeoGym)
                .framework(args.framework)
                .rollouts(num_rollout_workers=1)
                .training(
                model={
                    "custom_model": "my_model",
                    "vf_share_layers": True,
                    "conv_filters": [[16, [7, 13], 6], [32, [5, 13], 4], [256, [5, 14], 5]],
                    "post_fcnet_hiddens":[64,32],
                }
            )
                # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
                .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        )

        stop = {
            "training_iteration": args.stop_iters,
            "timesteps_total": args.stop_timesteps,
            "episode_reward_mean": args.stop_reward,
        }

        if args.no_tune:
            # manual training with train loop using PPO and fixed learning rate
            if args.run != "PPO":
                raise ValueError("Only support --run PPO with --no-tune.")
            print("Running manual train loop without Ray Tune.")
            # use fixed learning rate instead of grid search (needs tune)
            config.lr = 1e-3
            algo = config.build()
            # run manual training loop and print results after each iteration
            for _ in range(args.stop_iters):
                result = algo.train()
                print(pretty_print(result))
                # stop training of the target train steps or reward are reached
                if (
                        result["timesteps_total"] >= args.stop_timesteps
                        or result["episode_reward_mean"] >= args.stop_reward
                ):
                    break
            algo.stop()
        else:
            # automated run with Tune and grid search and TensorBoard
            print("Training automatically with Ray Tune")
            tuner = tune.Tuner(
                args.run,
                param_space=config.to_dict(),
                run_config=air.RunConfig(stop=stop),
            )
            results = tuner.fit()

            if args.as_test:
                print("Checking if learning goals were achieved")
                check_learning_achieved(results, args.stop_reward)

    env.close()
    
 
