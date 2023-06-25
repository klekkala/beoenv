#import graph_tool.all as gt
#import gymnasium as gym
import gym
import time
from gym import spaces
from gym.utils import seeding
from beogym.data_helper import dataHelper
from beogym.agent import Agent
#from graph_tool.all import *
#import graph_tool.all as gt
import numpy as np
import beogym.config as app_config
import math, cv2, argparse, csv, copy, time, os, shutil
from pathlib import Path
import graph_tool.all as gt

class BeoGym(gym.Env):

    def __init__(self, config):
        config = config or {}

        self.no_image = config.get("no_image", False)
        self.separate = config.get("separate", '')
        self.data_path= config.get("data_path", '')
        self.city = config.get("city", None)
        turning_range = 30 if app_config.PANO_IMAGE_MODE else 60

        super(BeoGym, self).__init__()
        
        self.dh = dataHelper("beogym/data/pano_gps.csv", app_config.PANO_HOV, city=self.city, data_path=self.data_path)
        self.agent = Agent(self.dh, turning_range, app_config.PANO_IMAGE_RESOLUTION, app_config.PANO_IMAGE_MODE)
        self.action_space = spaces.Discrete(5)
        if self.no_image:
            print("No image mode")
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Dict(
                {"obs": spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
                 "aux": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)})

        self.seed(1)

        #if self.separate == ''

        # Settings:
        self.game = 'courier'
        self.max_steps = 1000
        self.curr_step = 0
        self.min_radius_meters = 100  # The radius around the goal that can be considered the goal itself.
        self.max_radius_meters = 2000  # The outer radius around the goal where the rewards kicks in.
        self.min_distance_reached = 15  # The closest distance the agent has been so far to the goal.
        self.goal_reward = 500
        #self.courier_goal = (66.20711657663878, -17.83818898981032)
        #self.courier_goal = (9.02626694481603, -38.50893524264249)
        #self.courier_goal = (33.75675480746176, 47.60359414010759)
        #self.courier_goal = (35.49040814970007, 23.44676699952089)
        #self.courier_goal = (-20.314332722434187, -32.017853841967224)
        self.last_action = -1
        self.this_action = -1

        # while True:
        #     self.courier_goal = self.dh.sample_location()
        #     dis=gt.shortest_distance(self.dh.G, source=self.dh.G.vertex(self.dh.Gdict[self.agent.agent_pos_curr]), target=self.dh.G.vertex(self.dh.Gdict[self.courier_goal]), weights=self.dh.G.ep['weight'])
        #     # if dis>=100 and dis<=600:
        #     if dis > 1500 and dis <= 2000:
        #         self.long=dis/10
        #         print('goal')
        #         print(self.courier_goal)
        #         print(dis)
        #         break
        #self.courier_goal = (-86.7451736372966, -54.15525697354867)

        # random goal
        # while True:
        #     self.courier_goal = self.dh.sample_location()
        #     self.initial_distance_to_goal = self.dh.distance_to_goal(self.agent.agent_pos_curr, self.courier_goal)
        #     # Make sure the goal is not within the max_radius_meters to the agent's current position. Also, the sampled goal
        #     # should not be the same as the agent's current position:
        #     if (self.initial_distance_to_goal > self.max_radius_meters and self.courier_goal != self.agent.agent_pos_curr):
        #         break
        #     print("Goal is None. Sampled a location as goal: ", self.courier_goal)
        #print('Goal is' + str(self.courier_goal))

        # Logging: to be implemented.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        info = {}
        self.curr_step = 0
        self.turn_count=0
        self.out_count=0
        self.right=0
        self.agent.reset()
        
        self.minD=300
        self.maxD=500
        #short300-500
        while True:
            self.courier_goal = self.dh.sample_location()
            dis=gt.shortest_distance(self.dh.G, source=self.dh.G.vertex(self.dh.Gdict[self.agent.agent_pos_curr]), target=self.dh.G.vertex(self.dh.Gdict[self.courier_goal]), weights=self.dh.G.ep['weight'])
            if dis>=self.minD and dis<=self.maxD:
                self.long=dis
                break

        self.shortest=99999
        

        #self.dh.path=self.dh.getShortestPathNodes(self.agent.agent_pos_curr,self.courier_goal)
        adj=self.dh.find_adjacent(self.agent.agent_pos_curr)
        self.dh.path_loc=1
        #self.next_goal=self.dh.path[1]
        #print(self.dh.path[1])
        #print(adj)
        

        #self.courier_goal=self.dh.route[self.dh.route_loc]
        
        #temp=[-1 for i in range(4)]
        #if len(adj)>4:
        #    adj=adj[:4]
        
        #angle=self.dh.fix_angle(self.dh.get_angle(self.agent.agent_pos_curr, self.next_goal))
        
        #for pos in range(len(adj)):
        #    angle = self.dh.fix_angle(self.dh.get_angle(self.agent.agent_pos_curr, adj[pos]))
        #    temp[pos]=angle/360
        #aux = [(self.agent.agent_pos_curr[0] + 100) / 200, (self.agent.agent_pos_curr[1] + 100) / 200,self.agent.curr_angle / 360, (self.next_goal[0] + 100) / 200,(self.next_goal[1] + 100) / 200,angle/360]
        #return self.agent.curr_view, reward, done, info
        # temp = [self.dh.fix_angle(self.dh.get_angle(self.agent.agent_pos_curr, self.courier_goal))/360]
        # aux = [(self.agent.agent_pos_curr[1] + 100) / 200, (self.agent.agent_pos_curr[0] + 100) / 200,self.agent.curr_angle / 360, (self.courier_goal[1] + 100) / 200,(self.courier_goal[0] + 100) / 200]
        # aux+=temp

        aux = [(self.courier_goal[1] + 100) / 200,(self.courier_goal[0] + 100) / 200]
        #self.agent.dis_to_goal = gt.shortest_distance(self.dh.G, source=self.dh.G.vertex(self.dh.Gdict[self.agent.agent_pos_curr]), target=self.dh.G.vertex(self.dh.Gdict[self.courier_goal]), weights=self.dh.G.ep['weight'])
        
        
        if self.no_image:
            #self.agent.reset()
            return np.array(aux)
        else:
            # gray_image = cv2.cvtColor(self.agent.curr_view, cv2.COLOR_RGB2GRAY)
            # self.agent.past_view = np.zeros((gray_image.shape[0], gray_image.shape[1], 4), dtype=np.float32)
            # self.agent.past_view = np.concatenate((self.agent.past_view, np.expand_dims(gray_image, axis=-1)), axis=-1)
            # return {'obs': self.agent.past_view, 'aux': np.array(aux)}
            p_view = self.agent.curr_view[0:208,104:312]
            # p_view = cv2.resize(self.agent.curr_view, (84, 84))
            p_view = cv2.resize(p_view, (84, 84))
            return {'obs': p_view, 'aux': np.array(aux)}

    def step(self, action):
        truncated = False
        info = {}
        # print("Step with action: ", action)
        self.agent.take_action(action)
        if action>=1:
            self.turn_count+=1
        else:
            self.turn_count=0
        self.last_action = self.this_action
        self.this_action = action
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

        # print("comparison: ", self.game == 'courier')

        # Three different type of games: https://arxiv.org/pdf/1903.01292.pdf
        if self.game == 'courier':
            reward, done = self.courier_reward_fn()
            #reward, terminated = self.reward_test()
        elif self.game == 'coin':
            reward, done = self.coin_reward_fn()
        elif self.game == 'instruction':
            reward, done = self.instruction_reward_fn()
        

        #adj=self.dh.find_adjacent(self.agent.agent_pos_curr)
        #temp=[-1 for i in range(4)]
        #if len(adj)>4:
            #adj=adj[:4]

        #for pos in range(len(adj)):
        #    angle = self.dh.fix_angle(self.dh.get_angle(self.agent.agent_pos_curr, adj[pos]))
        #    temp[pos]=angle/360
        #angle=self.dh.fix_angle(self.dh.get_angle(self.agent.agent_pos_curr, self.next_goal))

        #aux = [(self.agent.agent_pos_curr[0] + 100) / 200, (self.agent.agent_pos_curr[1] + 100) / 200,self.agent.curr_angle / 360, (self.next_goal[0] + 100) / 200,(self.next_goal[1] + 100) / 200, angle/360]
        #return self.agent.curr_view, reward, done, info
        # temp = [self.dh.fix_angle(self.dh.get_angle(self.agent.agent_pos_curr, self.courier_goal))/360]
        # aux = [(self.agent.agent_pos_curr[1] + 100) / 200, (self.agent.agent_pos_curr[0] + 100) / 200,self.agent.curr_angle / 360, (self.courier_goal[1] + 100) / 200,(self.courier_goal[0] + 100) / 200]
        # aux+=temp
        aux = [(self.courier_goal[1] + 100) / 200,(self.courier_goal[0] + 100) / 200]
        if (self.curr_step >= self.max_steps):
            truncated = True
            done = True
            info['time_limit_reached'] = True
        if self.no_image:
            return np.array(aux), reward, done, info
        else:
            # gray_image = cv2.cvtColor(self.agent.curr_view, cv2.COLOR_RGB2GRAY)
            # self.agent.past_view[:-1] = self.agent.past_view[1:]
            # self.agent.past_view[..., -1] = gray_image
            # #return {'obs': self.agent.curr_view, 'aux': np.array(aux)}, reward, terminated,truncated, info
            p_view = self.agent.curr_view[0:208,104:312]
            p_view = cv2.resize(p_view, (84, 84))
            # p_view = cv2.resize(self.agent.curr_view, (84, 84))
            return {'obs': p_view, 'aux': np.array(aux)}, reward, done, info

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
                if abs(dis) <= 15:
                    last_angle = angle
                else:
                    if abs(self.dh.get_distance(last_angle + 15, angle)) < abs(
                            self.dh.get_distance(last_angle - 15, angle)):
                        last_angle += 15
                    else:
                        last_angle -= 15
                self.agent.update_agent(self.agent.agent_pos_curr, self.agent.agent_pos_curr, last_angle)
                if window_name:
                    cv2.imshow(window_name, self.agent.curr_view)
                    key = cv2.waitKey(1000)
                last_angle = angle

    def random_mode(self, steps, window_name=None):
        self.reset()
        start = time.time()
        for i in range(steps):
            # 7 since there are 6 actions that can be taken. (0-7 since 0 is inclusive and 7 is exclusive )
            self.step(i%5)
            #self.step(np.random.randint(low=0, high=5))
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

    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

        # Implementation will be similar to this file: https://github.com/deepmind/streetlearn/blob/master/streetlearn/python/environment/courier_game.py
    def courier_reward_fn(self, distance=None):

        reward = 0
        found_goal = False

        if self.this_action >=1:
            return reward, found_goal

        #distance_to_goal = 1
        distance_to_goal = gt.shortest_distance(self.dh.G, source=self.dh.G.vertex(self.dh.Gdict[self.agent.agent_pos_curr]), target=self.dh.G.vertex(self.dh.Gdict[self.courier_goal]), weights=self.dh.G.ep['weight'])
        # Does not give reward if the agent visited old locations:
        
        if self.agent.agent_pos_curr in self.dh.visited_locations:
            # self.agent.dis_to_goal = distance_to_goal
            return reward, found_goal
        #if self.agent.agent_pos_curr == self.courier_goal:
        if distance_to_goal < self.min_radius_meters:
            return (1000-self.curr_step)/5, True
            self.shortest=999999
            reward = self.long
            self.minD+=100
            self.maxD+=100
            while True:
                self.courier_goal = self.dh.sample_location()
                dis=gt.shortest_distance(self.dh.G, source=self.dh.G.vertex(self.dh.Gdict[self.agent.agent_pos_curr]), target=self.dh.G.vertex(self.dh.Gdict[self.courier_goal]), weights=self.dh.G.ep['weight'])
                if dis>=self.minD and dis<=self.maxD:
                    self.long=dis
                    break

            self.dh.visited_locations=set()

        
     
        else:

            if distance_to_goal<self.shortest:
                # reward=max(0,min(1,(200-distance_to_goal)/100))*self.long
                reward=1
                self.shortest=distance_to_goal


        self.dh.visited_locations.add(self.agent.agent_pos_curr)
        return reward, found_goal



    def coin_reward_fn(self):

        pass

    def instruction_reward_fn(self):

        pass
