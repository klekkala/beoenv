#import graph_tool.all as gt
#import gymnasium as gym
import gym
import random
import time,json
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
import matplotlib.pyplot as plt

class BeoGym(gym.Env):

    def __init__(self, config):
        config = config or {}

        self.no_image = config.get("no_image", False)
        self.separate = config.get("separate", '')
        self.data_path= config.get("data_path", '')
        self.city = config.get("city", None)
        self.read_info = config.get("read_info", True)
        turning_range = 30 if app_config.PANO_IMAGE_MODE else 60

        super(BeoGym, self).__init__()
        
        self.dh = dataHelper("beogym/data/pano_gps.csv", app_config.PANO_HOV, city=self.city, data_path=self.data_path)
        self.agent = Agent(self.dh, turning_range, app_config.PANO_IMAGE_RESOLUTION, app_config.PANO_IMAGE_MODE)
        self.action_space = spaces.Discrete(5)
        if self.no_image:
            print("No image mode")
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        else:
            # self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 9), dtype=np.uint8)
            # self.observation_space = gym.spaces.Dict(
            #     {"obs": spaces.Box(low=0, high=255, shape=(4, 84, 84, 3), dtype=np.uint8),
            #      "aux": spaces.Box(low=-1.0, high=1.0, shape=(4, 1), dtype=np.float32)})
            self.observation_space = gym.spaces.Dict(
                {"obs": spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
                 "aux": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)})
        self.seed(1)

        #if self.separate == ''

        # Settings:
        self.game = 'courier'
        self.max_steps = 700
        self.curr_step = 0
        self.min_radius_meters = 36 # The radius around the goal that can be considered the goal itself.
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
        self.frames=[]
        self.aux=[]
        self.random_source = []
        # if self.city=='Wall_Street':    
        #     self.goal_img = self.dh.panorama_split(0,(-91.26371576373391, -92.93867163751861),0,True)
        # elif self.city=='Union_Square':
        #     self.goal_img = self.dh.panorama_split(0,(-21.75919854376224, -22.524020509970285),0,True)

        # elif self.city=='Hudson_River':
        #     self.goal_img = self.dh.panorama_split(0,(82.75054306685817, -4.9921430896916235),0,True)
        # elif self.city == 'CMU':
        #     self.goal_img = self.dh.panorama_split(0,(-26.494148909291482, 98.54096082676443),0,True)
        # elif self.city == 'Allegheny':
        #     self.goal_img = self.dh.panorama_split(0,(98.11349718215925, -65.91599501277392),0,True)
        # elif self.city == 'South_Shore':
        #     self.goal_img = self.dh.panorama_split(0,(-40.86770901528804, -57.97858763520175),0,True)
        # elif 'navigation' in self.city:
        #     self.goal_img = self.dh.panorama_split(self.dh.pre_deg, self.dh.navi_routes[0],0,True)
        #     self.end_img = self.dh.panorama_split(self.dh.post_deg, self.dh.navi_routes[-1],0,True)

        # height, width, _ = self.goal_img.shape
        # size = min(height, width)
        # x = (width - size) // 2
        # y = (height - size) // 2
        # self.goal_img = self.goal_img[y:y+size,x:x+size]
        # self.goal_img= cv2.resize(self.goal_img, (84, 84))
        # if 'navigation' in self.city:
        #     self.end_img = self.end_img[y:y+size,x:x+size]
        #     self.end_img= cv2.resize(self.end_img, (84, 84))


        # cv2.imwrite('pre.jpg', self.goal_img)
        # cv2.imwrite('post.jpg', self.end_img)
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

        if self.read_info:
            with open('task.json', 'r') as json_file:
                for line in json_file:
                    temp = json.loads(line)
                    if self.city==temp['city']:
                        self.info = temp
                        self.info['routes'] = [tuple(i) for i in self.info['routes']]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        info = {}
        self.curr_step = 0
        self.turn_count=0
        self.out_count=0
        self.right=0
        if self.read_info:
            # print(type(self.info['source']))
            pos_p=[tuple(self.info['source'])]
            for i in range(5):
                new_ps=[]
                for q in pos_p:
                    new_ps+=self.dh.find_adjacent(q)
                pos_p+=new_ps
                pos_p = list(set(pos_p))
            # _,self.long = self.agent.reset(tuple(self.info['source']))
            _,self.long = self.agent.reset(random.choice(pos_p))
        
        self.minD=300
        self.minD=900
        self.maxD=1200
        # self.maxD=99999
        #short300-500
        # while True:
        #     self.courier_goal = self.dh.sample_location()
        #     dis=gt.shortest_distance(self.dh.G, source=self.dh.G.vertex(self.dh.Gdict[self.agent.agent_pos_curr]), target=self.dh.G.vertex(self.dh.Gdict[self.courier_goal]), weights=self.dh.G.ep['weight'])
        #     if dis>=self.minD and dis<=self.maxD:
        #         print(self.courier_goal)
        #         self.long=dis
        #         break
        
        if self.city=='Wall_Street':    
            self.courier_goal= (-91.26371576373391, -92.93867163751861)
        elif self.city=='Union_Square':
            self.courier_goal = (-21.75919854376224, -22.524020509970285)
        elif self.city=='Hudson_River':
            self.courier_goal = (82.75054306685817, -4.9921430896916235)
        elif self.city == 'CMU':
            self.courier_goal = (-26.494148909291482, 98.54096082676443)
        elif self.city == 'Allegheny':
            self.courier_goal = (98.11349718215925, -65.91599501277392)
        elif self.city == 'South_Shore':
            self.courier_goal = (-40.86770901528804, -57.97858763520175)
        elif 'navigation' in self.city:
            self.courier_goal = self.dh.navi_routes[-1]
            self.right_angle = False
            self.idx_route = 0

        if self.read_info:
            self.courier_goal = tuple(self.info['goal'])

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
        # self.agent.dis_to_goal = gt.shortest_distance(self.dh.G, source=self.dh.G.vertex(self.dh.Gdict[self.agent.agent_pos_curr]), target=self.dh.G.vertex(self.dh.Gdict[self.courier_goal]), weights=self.dh.G.ep['weight'])
        self.agent.dis_to_goal  =  math.sqrt((self.agent.agent_pos_curr[0] - self.courier_goal[0])**2 + (self.agent.agent_pos_curr[1] - self.courier_goal[1])**2)

        threshold = 15
        self.marks = []
        self.marks_idx = 0
        for i in range(threshold):
            # self.marks.append(8*((self.agent.dis_to_goal/8)**(1/threshold))**i)
            self.marks.append(self.agent.dis_to_goal/threshold*(i+1))
        self.marks=self.marks[:-1]
        self.marks.reverse()
        self.marks.append(0.0)

        # self.dis_to_source = gt.shortest_distance(self.dh.G, source=self.dh.G.vertex(self.dh.Gdict[self.agent.agent_pos_curr]), target=self.dh.G.vertex(self.dh.Gdict[self.agent.source_pos]), weights=self.dh.G.ep['weight'])
        # aux = [(self.courier_goal[1] + 100) / 200,(self.courier_goal[0] + 100) / 200, self.agent.dis_to_goal/5000]
        # self.aux = [[self.agent.dis_to_goal/5000] for i in range(4)]
        aux = [(self.agent.agent_pos_curr[1] - self.agent.source_pos[1] + 200) / 400,(self.agent.agent_pos_curr[0] - self.agent.source_pos[0] + 200) / 400]

        if self.no_image:
            #self.agent.reset()
            return np.array(aux)
        else:
            # gray_image = cv2.cvtColor(self.agent.curr_view, cv2.COLOR_RGB2GRAY)
            # self.agent.past_view = np.zeros((gray_image.shape[0], gray_image.shape[1], 4), dtype=np.float32)
            # self.agent.past_view = np.concatenate((self.agent.past_view, np.expand_dims(gray_image, axis=-1)), axis=-1)
            # return {'obs': self.agent.past_view, 'aux': np.array(aux)}
            height, width, _ = self.agent.curr_view.shape
            size = min(height, width)
            x = (width - size) // 2
            y = (height - size) // 2
            p_view = self.agent.curr_view[y:y+size,x:x+size]
            p_view = cv2.resize(p_view, (84, 84))
            # return np.concatenate((p_view,self.goal_img,self.end_img),axis=2)
            # gray_image = cv2.cvtColor(p_view, cv2.COLOR_RGB2GRAY)
            # self.frames = np.repeat(gray_image[:, :, np.newaxis], 4, axis=2)
            return {'obs': p_view, 'aux': np.array(aux)}
            self.frames = np.repeat(p_view[np.newaxis, :, :, :], 4, axis=0)
            # return {'obs': self.frames, 'aux': np.array(self.aux)}

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
            if 'navigation' in self.city:
                reward, done = self.courier_reward_fn_navi()
            else:
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
        # self.agent.dis_to_goal = gt.shortest_distance(self.dh.G, source=self.dh.G.vertex(self.dh.Gdict[self.agent.agent_pos_curr]), target=self.dh.G.vertex(self.dh.Gdict[self.courier_goal]), weights=self.dh.G.ep['weight'])
        # aux = [(self.courier_goal[1] + 100) / 200,(self.courier_goal[0] + 100) / 200, self.agent.dis_to_goal/5000]
        # aux = [(self.courier_goal[1] + 100) / 200,(self.courier_goal[0] + 100) / 200]
        aux = [(self.agent.agent_pos_curr[1] - self.agent.source_pos[1] + 200) / 400,(self.agent.agent_pos_curr[0] - self.agent.source_pos[0] + 200) / 400]
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
            height, width, _ = self.agent.curr_view.shape
            size = min(height, width)
            x = (width - size) // 2
            y = (height - size) // 2
            p_view = self.agent.curr_view[y:y+size,x:x+size]
            p_view = cv2.resize(p_view, (84, 84))
            # self.dis_to_source = gt.shortest_distance(self.dh.G, source=self.dh.G.vertex(self.dh.Gdict[self.agent.agent_pos_curr]), target=self.dh.G.vertex(self.dh.Gdict[self.agent.source_pos]), weights=self.dh.G.ep['weight'])
            return {'obs': p_view, 'aux': np.array(aux)}, reward, done, info
            # return np.concatenate((p_view,self.goal_img,self.end_img),axis=2),reward, done, info
            # gray_image = cv2.cvtColor(p_view, cv2.COLOR_RGB2GRAY)
            # self.frames[:-1] = self.frames[1:]
            # self.frames[..., -1] = p_view
            self.frames[:-1] = self.frames[1:]
            self.frames[-1] = p_view
            self.aux[:-1] = self.aux[1:]
            self.aux[-1] = np.array([self.agent.dis_to_goal/5000])
            return {'obs': self.frames, 'aux': np.array(self.aux)}, reward, done, info
            # return {'obs': p_view, 'aux': np.array(aux)}, reward, done, info

    def render(self, mode='human', steps=100):

        # Renders initial plot
        window_name = 'window'
        # print("POS", self.agent.agent_pos_curr, "PREV", self.agent.agent_pos_prev, "CUR_ANGLE", self.agent.curr_angle)
        if mode != 'random':
            # self.dh.update_plot(self.agent.agent_pos_curr, self.courier_goal)

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
                # self.dh.update_plot(self.agent.agent_pos_curr, self.courier_goal)
                # img = cv2.imshow(self.agent.curr_view)
                cv2.imshow(window_name, self.agent.curr_view)

            # Update Bird's-Eye-View graph
            # graph = self.dh.bird_eye_view(self.agent.agent_pos_curr, r)
            # if graph is None:
            #     raise EnvironmentError("Graph is None")
            # self.dh.draw_bird_eye_view(self.agent.agent_pos_curr, r, graph, self.agent.curr_angle)

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
        # distance_to_goal = gt.shortest_distance(self.dh.G, source=self.dh.G.vertex(self.dh.Gdict[self.agent.agent_pos_curr]), target=self.dh.G.vertex(self.dh.Gdict[self.courier_goal]), weights=self.dh.G.ep['weight'])
        distance_to_goal = math.sqrt((self.agent.agent_pos_curr[0] - self.courier_goal[0])**2 + (self.agent.agent_pos_curr[1] - self.courier_goal[1])**2)
        # Does not give reward if the agent visited old locations:
        self.agent.dis_to_goal = distance_to_goal
        if self.agent.agent_pos_curr in self.dh.visited_locations:
            # self.agent.dis_to_goal = distance_to_goal
            return reward, found_goal
        # if self.agent.agent_pos_curr == self.courier_goal:
        # if distance_to_goal < self.min_radius_meters:
        if distance_to_goal < 1:
            # reward = self.long
            # _, self.long = self.agent.reset()
            return 1,True
            
        #     return (1000-self.curr_step)/5, True
        #     self.shortest=999999
        #     reward = self.long
        #     self.minD+=100
        #     self.maxD+=100
        #     while True:
        #         self.courier_goal = self.dh.sample_location()
        #         dis=gt.shortest_distance(self.dh.G, source=self.dh.G.vertex(self.dh.Gdict[self.agent.agent_pos_curr]), target=self.dh.G.vertex(self.dh.Gdict[self.courier_goal]), weights=self.dh.G.ep['weight'])
        #         if dis>=self.minD and dis<=self.maxD:
        #             self.long=dis
        #             break

        #   self.dh.visited_locations=set()

        
     
        else:

            if distance_to_goal<self.shortest:
                # reward=max(0,min(2,(400-distance_to_goal)/100))
                # if self.agent.agent_pos_curr in self.info['routes'] and len(self.dh.find_adjacent(self.agent.agent_pos_curr)) >= 3:
                if distance_to_goal<=self.marks[self.marks_idx]:
                    reward=1
                    self.marks_idx+=1
                else:
                    reward=0
                self.shortest=distance_to_goal


        self.dh.visited_locations.add(self.agent.agent_pos_curr)
        return reward, found_goal


    def courier_reward_fn_navi(self, distance=None):


        if not self.right_angle:
            if self.agent.agent_pos_curr == self.dh.navi_routes[0]:
                agl = self.dh.fix_angle(self.dh.get_distance(self.agent.curr_angle, self.dh.pre_deg))
                if agl<=10 or agl>=350:
                    self.right_angle = True
                    return 1,False
                else:
                    return 0,False
            else:
                return 0,False
        else:
            if self.agent.agent_pos_curr == self.dh.navi_routes[self.idx_route+1]:
                self.idx_route+=1

                if self.idx_route == len(self.dh.navi_routes)-1:
                    # _, self.long = self.agent.reset()
                    # self.idx_route = 0
                    # self.right_angle == False
                    return 1, True
                else:
                    return 1,False
            else:
                return 0, False
            



    def coin_reward_fn(self):

        pass

    def instruction_reward_fn(self):

        pass



    def shortest_rec(self):
        allObs=[]
        allAux=[]
        allAct=[]
        allRew=[]
        allTar=[]
        allSrt=[]
        allTer=[]
        allp=[]
        total_steps=0
        all_nodes=[]
        for i in self.dh.Gdict.keys():
            all_nodes.append(i)
        all_x,all_y =  zip(*all_nodes)
        while total_steps<1000000:
            self.reset()
            pos_p=[tuple(self.info['source'])]
            for i in range(5):
                new_ps=[]
                for q in pos_p:
                    new_ps+=self.dh.find_adjacent(q)
                pos_p+=new_ps
                pos_p = list(set(pos_p))
            # _,self.long = self.agent.reset(tuple(self.info['source']))
            srt_loc = random.choice(pos_p)
            _,self.long = self.agent.reset(srt_loc)
            # self.agent.reset(tuple(self.info['source']))
            self.courier_goal = tuple(self.info['goal'])

            obsRec=[]
            auxRec=[]
            actRec=[]
            rewRec=[]
            tarRec=[]
            srtRec=[]
            terRec=[]
            now_shortest = gt.shortest_distance(self.dh.G, source=self.dh.G.vertex(self.dh.Gdict[self.agent.agent_pos_curr]), target=self.dh.G.vertex(self.dh.Gdict[self.courier_goal]), weights=self.dh.G.ep['weight'])
            source_goal = now_shortest
            while True:
                now_shortest = source_goal
                node4=[]
                for rdms in range(2):
                    fgg=0
                    while fgg==0: 
                        pos = self.dh.sample_location()
                        tmp_dis_goal = gt.shortest_distance(self.dh.G, source=self.dh.G.vertex(self.dh.Gdict[pos]), target=self.dh.G.vertex(self.dh.Gdict[self.courier_goal]), weights=self.dh.G.ep['weight'])
                        tmp_dis_source = gt.shortest_distance(self.dh.G, source=self.dh.G.vertex(self.dh.Gdict[pos]), target=self.dh.G.vertex(self.dh.Gdict[self.agent.agent_pos_curr]), weights=self.dh.G.ep['weight'])
                        if tmp_dis_goal< now_shortest and tmp_dis_source<source_goal:
                            node4.append(pos)
                            now_shortest = tmp_dis_goal
                            fgg=1
                roads4=[]
                tmp_rd = self.dh.getShortestPathNodes(self.agent.agent_pos_curr, node4[0])
                # roads4+=self.dh.getShortestPathNodes(self.agent.agent_pos_curr, node4[0])
                roads4+=tmp_rd
                # roads4=roads4[:-1]
                tmp_rd = self.dh.getShortestPathNodes(node4[0], node4[1])
                idx = 0
                while tmp_rd[idx]==roads4[-1]:
                    if idx==len(tmp_rd)-1:
                        break
                    idx+=1
                roads4+=tmp_rd[idx:]
                tmp_rd = self.dh.getShortestPathNodes(node4[1], self.courier_goal)
                if len(tmp_rd) != 0:
                    idx = 0
                    while tmp_rd[idx]==roads4[-1]:
                        if idx==len(tmp_rd)-1:
                            break
                        idx+=1
                    roads4+=tmp_rd[idx:]
                # tmp_rd = self.dh.getShortestPathNodes(node4[2], node4[3])
                # idx = 0
                # while tmp_rd[idx]==roads4[-1]:
                #     idx+=1
                # roads4+=tmp_rd[idx:]
                # tmp_rd = self.dh.getShortestPathNodes(node4[3], self.courier_goal)
                # idx = 0
                # while tmp_rd[idx]==roads4[-1]:
                #     idx+=1
                # roads4+=tmp_rd[idx:]
                if len(roads4) <= 500:
                # if len(roads4) == len(set(roads4)):
                    shortest_path = roads4
                    break

            flag=False
            for i in range(len(shortest_path)):
                if i != len(shortest_path) - 1:
                    new_pos, curr_pos, new_angle = self.dh.find_nearest(self.agent.agent_pos_curr, self.agent.agent_pos_prev, self.agent.curr_angle, "forward")
                    while new_pos!=shortest_path[i + 1]:
                        agl = self.dh.fix_angle(self.dh.get_angle(shortest_path[i], shortest_path[i + 1]))
                        minAngle=500
                        cact=-1
                        acts=[10, -10, 20, -20]
                        for act in range(len(acts)):
                            tAngle=abs(self.dh.get_distance(agl, self.agent.curr_angle+acts[act]))
                            if tAngle < minAngle:
                                cact=act
                                minAngle = tAngle
                        obs, reward, done, info = self.step(cact+1)
                        obsRec.append(obs['obs'])
                        auxRec.append(obs['aux'])
                        tarRec.append((self.courier_goal[0], self.courier_goal[1]))
                        srtRec.append((srt_loc[0], srt_loc[1]))
                        actRec.append(cact+1)
                        rewRec.append(reward)
                        if not done:
                            terRec.append(0)
                        else:
                            flag=True
                            break
                        new_pos, curr_pos, new_angle = self.dh.find_nearest(self.agent.agent_pos_curr, self.agent.agent_pos_prev, self.agent.curr_angle, "forward")

                    # if not done:
                    if not flag:

                        obs, reward, done, info = self.step(0)
                        if reward>0:
                            allp.append(self.agent.agent_pos_curr)
                        obsRec.append(obs['obs'])
                        auxRec.append(obs['aux'])
                        tarRec.append((self.courier_goal[0], self.courier_goal[1]))
                        srtRec.append((srt_loc[0], srt_loc[1]))
                        actRec.append(0)
                        rewRec.append(reward)
                    else:
                        done=True
                    # print(curr_steps)
                if done:
                    terRec.append(1)
                    allObs+=obsRec
                    allAux+=auxRec
                    allAct+=actRec
                    allRew+=rewRec
                    allTar+=tarRec
                    allSrt+=srtRec
                    allTer+=terRec
                    total_steps+=len(terRec)

                    x_r,y_r = zip(*shortest_path)
                    plt.scatter(all_x, all_y, color='blue', s=1)
                    plt.scatter(x_r, y_r, color='red', s=1)
                    plt.legend()
                    plt.savefig(f'/home6/tmp/kiran/expert_3chan_beogym/skill2/expert_3chan_hudsonriver/5/50/plots_aux/{total_steps}.png')
                    plt.clf()
                    print(total_steps)
                    break
                else:
                    terRec.append(0)
        sliceL = total_steps - 1000000
        allObs = allObs[:-sliceL]
        allAux = allAux[:-sliceL]
        allAct = allAct[:-sliceL]
        allRew = allRew[:-sliceL]
        allTar = allTar[:-sliceL]
        allSrt = allSrt[:-sliceL]
        allTer = allTer[:-sliceL]
        allTer[-1] = 1

        allObs = np.array(allObs)
        allAux = np.array(allAux)
        allAct = np.array(allAct)
        allRew = np.array(allRew)
        allTar = np.array(allTar)
        allSrt = np.array(allSrt)
        allTer = np.array(allTer)
        allp = np.array(allp)
        fileName = '/home6/tmp/kiran/expert_3chan_beogym/skill2/expert_3chan_hudsonriver/5/50/'
        np.save(fileName+'observation.npy',allObs)
        np.save(fileName+'aux.npy',allAux)
        np.save(fileName+'action.npy',allAct)
        np.save(fileName+'reward.npy',allRew)
        np.save(fileName+'goal.npy',allTar)
        np.save(fileName+'start.npy',allSrt)
        np.save(fileName+'terminal.npy',allTer)
        np.save(fileName+'allp.npy',allp)

    def output_video(self):
        obsRec=[]
        auxRec=[]
        actRec=[]
        rewRec=[]
        tarRec=[]
        terRec=[]
        posRec=[]
        all_nodes=[]
        for i in self.dh.Gdict.keys():
            all_nodes.append(i)
        all_x,all_y =  zip(*all_nodes)
        self.reset()
        self.agent.reset(tuple(self.info['source']))
        self.courier_goal = tuple(self.info['goal'])
        shortest_path = self.dh.getShortestPathNodes(tuple(self.info['source']), tuple(self.info['goal']))
        flag=False
        for i in range(len(shortest_path)):
            if i != len(shortest_path) - 1:
                new_pos, curr_pos, new_angle = self.dh.find_nearest(self.agent.agent_pos_curr, self.agent.agent_pos_prev, self.agent.curr_angle, "forward")
                while new_pos!=shortest_path[i + 1]:
                    agl = self.dh.fix_angle(self.dh.get_angle(shortest_path[i], shortest_path[i + 1]))
                    minAngle=500
                    cact=-1
                    acts=[10, -10, 20, -20]
                    for act in range(len(acts)):
                        tAngle=abs(self.dh.get_distance(agl, self.agent.curr_angle+acts[act]))
                        if tAngle < minAngle:
                            cact=act
                            minAngle = tAngle
                    obs, reward, done, info = self.step(cact+1)
                    obsRec.append(self.agent.curr_view)
                    x_r,y_r = self.agent.agent_pos_curr
                    plt.scatter(all_x, all_y, color='blue', s=1)
                    plt.scatter(x_r, y_r, color='red', s=40)
                    # plt.legend()
                    figure = plt.gcf()
                    figure.canvas.draw()
                    img = np.array(figure.canvas.buffer_rgba())
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    plt.clf()
                    posRec.append(img)
                    auxRec.append(obs['aux'])
                    tarRec.append((self.courier_goal[0], self.courier_goal[1]))
                    actRec.append(cact+1)
                    rewRec.append(reward)
                    if not done:
                        terRec.append(0)
                    else:
                        flag=True
                        break
                    new_pos, curr_pos, new_angle = self.dh.find_nearest(self.agent.agent_pos_curr, self.agent.agent_pos_prev, self.agent.curr_angle, "forward")

                # if not done:
                if not flag:

                    obs, reward, done, info = self.step(0)
                    obsRec.append(self.agent.curr_view)
                    x_r,y_r = self.agent.agent_pos_curr
                    plt.scatter(all_x, all_y, color='blue', s=1)
                    plt.scatter(x_r, y_r, color='red', s=40)
                    # plt.legend()
                    figure = plt.gcf()
                    figure.canvas.draw()
                    img = np.array(figure.canvas.buffer_rgba())
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    plt.clf()
                    posRec.append(img)
                    auxRec.append(obs['aux'])
                    tarRec.append((self.courier_goal[0], self.courier_goal[1]))
                    actRec.append(0)
                    rewRec.append(reward)
                else:
                    done=True
                # print(curr_steps)
            if done:
                terRec.append(1)
                # cv2.imwrite('se1e.png', img)
                break
            else:
                terRec.append(0)

        height, width, _ = obsRec[0].shape
        for i in range(len(posRec)):
            posRec[i] = cv2.resize(posRec[i], (height, height))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID', 'MP4V', 'MJPG', etc.
        output_video = cv2.VideoWriter('Wall.mp4', fourcc, 2, (208, 624))
        nn=0
        for left_img, right_img in zip(obsRec, posRec):
            # print(left_img.shape)
            # print(right_img.shape)
            concatenated_image = np.hstack((left_img, right_img))
            cv2.imwrite(f'./all_road/Wall/{nn}.jpg', concatenated_image)
            nn+=1
            # output_video.write(concatenated_image)
        output_video.release()

