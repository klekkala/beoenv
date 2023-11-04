import graph_tool.all as gt
import pickle
import time

import shutil
import json
#import Equirec2Perspec as E2P
import beogym.Equirec2Perspec_cpu as E2P
import pandas as pd
#import graph_tool.all as gt
#from graph_tool.all import *
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
#import cupy as cp
import math
import os,sys
#import h5py
from math import radians, cos, sin, asin, acos, sqrt, pi

import beogym.config as app_config

#import graph_tool.all as gt

new_min = -100
new_max = 100
lat_min = 40.707091
lat_max = 40.764838
long_min = -74.005022
long_max = -73.976614

coord_to_sect = {}
scaled_coord_to_coord = {}
coord_to_filename = {}


# Helper function used to calculate the distance of an edge in Graph

def calcDist(start, end):
    phi1 = radians(start[0])
    lambda1 = radians(start[1])
    phi2 = radians(end[0])
    lambda2 = radians(end[1])
    r = 6371  # Radius of the earth in kilometers

    temp = (np.square(sin((phi2 - phi1) / 2)) + cos(phi1) * cos(phi2) *
            np.square(sin((lambda2 - lambda1) / 2)))
    sphereDist = 2 * r * asin(np.sqrt(temp))

    return sphereDist


class dataHelper():

    def __init__(self, csv_file, hov_val, city, data_path, isFile=False):

        self.G = gt.Graph()
        self.Gdict={}
        self.Greversed={}
        self.end_points = []
        self.route=[(1.3984153906729517, -24.730699803708518), (12.281358723435147, -14.786001935545627), (25.395387695338428, 1.13332033476172), (35.49040814970007, 23.44676699952089)]
        self.route_loc=0
        self.path_loc=0
        self.city = city
        self.equ = E2P.Equirectangular(data_path, self.city)
        # Create canvas for plot rendering:
        self.xdata = []
        self.ydata = []
        self.tm = [0]

        # Set of visited locations.
        self.visited_locations = set()

        self.read_routes(csv_file)

        self.camera_angle_map = {
            1: 0,
            2: 30,
            3: 60,
            4: 90,
            5: 120,
            6: 150,
            7: 180,
            8: 210,
            9: 240,
            10: 270,
            11: 300,
            12: 330
        }

        self.hov = hov_val
        self.isFile=isFile


        # Dictionary used to get the shortest path using latlong positions
        #self.allNodes = dict(self.G.nodes(data="image_name"))


    def getShortestPathNodes(self, start, end):
        vlist, elist = gt.shortest_path(self.G, self.G.vertex(self.Gdict[start]), self.G.vertex(self.Gdict[end]),weights=self.G.ep['weight'])
        posPath = [self.Greversed[i] for i in vlist]
         
        return posPath

    def new_lat_scale(self, x):
        normalized_new_val = ((x - lat_min) / (lat_max - lat_min) * (new_max - new_min)) + new_min
        return normalized_new_val

    def new_long_scale(self, x):
        normalized_new_val = ((x - long_min) / (long_max - long_min) * (new_max - new_min)) + new_min
        return normalized_new_val

    def image_name(self, pos):
        return self.image_names[pos]

    def get_image_orientation(self, image_coord, camera):
        if camera > 5 or camera < 1:
            raise ValueError("Camera should be an integer between 1 and 5 inclusive.")

        # Switched to using hd5 files
        """
        image_file = "%s.tiff" % str(image)
        camera_dir_name = f"Cam{camera}"
        img_path = os.path.join(app_config.CAMERA_IMAGES_PATH, camera_dir_name, image_file)
        """
        global coord_to_sect
        global scaled_coord_to_coord

        coord = scaled_coord_to_coord[(image_coord[0], image_coord[1])]
        coord_str = ','.join([str(value) for value in coord])

        path = coord_to_sect[coord_str]
        f = h5py.File(f"../{path}", "r")
        curr_image = f[coord_str]["rgb_pano"][camera - 1]

        """
        print("\n")
        print("Showing image: ", img_path)
        print("Current camera: ", camera)
        print("\n")
        """

        print("\n")
        # print("Showing image: ", curr_image)
        print("Current camera: ", camera)
        print("\n")

        # return img_path
        return curr_image

    def panorama_split(self, theta, curr_pos, resolution, move=False):
        if move:
            self.equ.get_image(curr_pos)
        #
        # FOV unit is degree
        # theta is z-axis angle(right direction is positive, left direction is negative)
        # phi is y-axis angle(up direction positive, down direction negative)
        # height and width is output image dimension
        #
        img = self.equ.GetPerspective(90, theta)

        return img

    def build_graph(self, data, csv_file):
        links = True
        csv_file = csv_file.split('/')[-1]
        localMap = 'beogym/'+csv_file.split('.')[0] + '.gpickle'
        print("\n")
        print("Building graph. \n")

        if os.path.isfile(localMap):
            self.G = gt.load_graph("beogym/nyc.gt")
            
            self.end_points = pickle.load(open(localMap.split('.')[0] + '.pickle', 'rb'))
            self.Gdict = pickle.load(open('beogym/nycgraph.pkl', 'rb'))
            self.Greversed={value: key for key, value in self.Gdict.items()}
            #self.image_names = nx.get_node_attributes(self.G, 'image_name')
            # print(self.G)
        if links == True and not os.path.isfile(localMap):
            links = pd.read_csv('beogym/data/manhattan_metadata_links.tsv', delimiter='\t')
            nodes = pd.read_csv('beogym/data/manhattan_metadata_nodes.tsv', delimiter='\t')
            nodes = nodes[['pano_id', 'coords.lat', 'coords.lng']]

            img2loc = {}
            for i in nodes.itertuples():
                img2loc[i[1]] = (i[2], i[3])

            for i in links.itertuples():
                point1 = (self.new_lat_scale(img2loc[i[1]][0]), self.new_long_scale(img2loc[i[1]][1]))
                point2 = (self.new_lat_scale(img2loc[i[2]][0]), self.new_long_scale(img2loc[i[2]][1]))
                self.G.add_edge(point1, point2, weight=calcDist(point1, point2))
                self.end_points.append(point2)

            nx.write_gpickle(self.G, localMap)
            pickle.dump(self.end_points, open(localMap.split('.')[0] + '.pickle', 'wb'))
        if not os.path.isfile(localMap) and not links:
            for index, row in data.iterrows():
                lat = float(row['coords.lat'])
                lon = float(row['coords.lng'])
                scaled_lat = self.new_lat_scale(lat)
                scaled_long = self.new_long_scale(lon)
                self.G.add_node((scaled_lat, scaled_long))

            allDis = {}
            isOpen = [1 for i in range(len(list(self.G.nodes)))]
            openNodes = list(self.G.nodes)
            for i in range(len(isOpen)):
                temp = {}
                for q in range(len(isOpen)):
                    if i < q:
                        temp[openNodes[q]] = calcDist(openNodes[i], openNodes[q])
                    elif i > q:
                        temp[openNodes[q]] = allDis[openNodes[q]][openNodes[i]]
                allDis[openNodes[i]] = temp
            while sum(isOpen) != 0:
                temp = isOpen.index(1)
                cloestNode = -1
                shortestDistance = float('inf')
                for i in range(len(isOpen)):
                    if i != temp:
                        tempDis = allDis[openNodes[i]][openNodes[temp]]
                        if tempDis < shortestDistance:
                            cloestNode = i
                            shortestDistance = tempDis
                isOpen[cloestNode] = 0
                isOpen[temp] = 0
                self.G.add_edge(openNodes[cloestNode], openNodes[temp], weight=shortestDistance)

            self.image_names = nx.get_node_attributes(self.G, 'image_name')
            subGraph = nx.connected_components(self.G)
            subGraph = list(subGraph)
            subGraphDis = {}
            for i in range(len(subGraph)):
                temp = {}
                for w in range(i + 1, len(subGraph)):
                    srtest = float('inf')
                    point1 = -1
                    point2 = -1
                    for q in range(len(list(subGraph[i]))):
                        for p in range(len(list(subGraph[w]))):
                            if allDis[list(subGraph[i])[q]][list(subGraph[w])[p]] < srtest:
                                srtest = allDis[list(subGraph[i])[q]][list(subGraph[w])[p]]
                                point1 = list(subGraph[i])[q]
                                point2 = list(subGraph[w])[p]
                    temp[w] = [srtest, point1, point2]
                    self.end_points.append(point2)
                subGraphDis[i] = temp

            temp = nx.Graph()
            for i in range(len(subGraph)):
                temp.add_node(i)

            while len(list(nx.connected_components(self.G))) > 1:
                print(self.G)
                srtest = float('inf')
                point1 = -1
                point2 = -1
                pos1 = -1
                pos2 = -2
                for i, j in subGraphDis.items():
                    for q, w in j.items():
                        if w[0] < srtest and not nx.has_path(temp, i, q):
                            srtest = w[0]
                            point1 = w[1]
                            point2 = w[2]
                            pos1 = i
                            pos2 = q
                self.G.add_edge(point1, point2, weight=srtest)
                temp.add_edge(pos1, pos2)

            nx.write_gpickle(self.G, localMap)
            pickle.dump(self.end_points, open(localMap.split('.')[0] + '.pickle', 'wb'))
        
        depth=0
        vertices=set()
        vfilt = self.G.new_vertex_property('bool');
        new_dict={}


        
        Wall_Street = [(-89.8135819320945, -86.99411624217562),7224]
        Union_Square = [(-21.67198697935339, -22.411925691329387),15525]
        Hudson_River = [(85.92341169389903, -12.685304196935363),18085]


        if self.city == 'Wall_Street':    
            chosen_city = Wall_Street
        elif self.city == 'Union_Square':
            chosen_city = Union_Square
        elif self.city == 'Hudson_River':
            chosen_city = Hudson_River
        else:
            chosen_city = None
        if chosen_city!=None:
            for e in gt.bfs_iterator(self.G, self.G.vertex(self.Gdict[chosen_city[0]])):
                vertices.add(e.source())
                vertices.add(e.target())
                vfilt[e.source()] = True
                vfilt[e.target()] = True
                new_dict[self.Greversed[e.source()]]=e.source()
                new_dict[self.Greversed[e.target()]]=e.target()
                if len(vertices)==chosen_city[1]:
                    break
            Gdict=new_dict
            Greversed={value: key for key, value in Gdict.items()}
            sub = gt.GraphView(self.G, vfilt)
            self.G = sub
            self.Gdict = Gdict
            self.Greversed=Greversed
        print(self.G)
        plt.savefig("filename5.png")

    def build_6city(self):
        NYC = ['Wall_Street','Union_Square', 'Hudson_River']
        Pits= ['CMU', 'Allegheny', 'South_Shore']
        if self.city in NYC:
            self.G = gt.load_graph("beogym/nyc.gt")
            self.Gdict = pickle.load(open('beogym/nycgraph.pkl', 'rb'))
            self.Greversed={value: key for key, value in self.Gdict.items()}
        elif self.city in Pits:
            self.G = gt.load_graph("beogym/pits.gt")
            self.Gdict = pickle.load(open('beogym/pitsgraph.pkl', 'rb'))
            self.Greversed={value: key for key, value in self.Gdict.items()}
        elif 'navigation' in self.city:
            self.G = gt.load_graph("beogym/tcd.gt")
            self.Gdict = pickle.load(open('beogym/tcdgraph.pkl', 'rb'))
            self.Greversed={value: key for key, value in self.Gdict.items()}
            task = self.city.replace('navigation','')
            with open ('skillset1.json', 'r') as f:
                navi_data = f.read().strip().split('\n')
                self.task = json.loads(navi_data[int(task)])
            img2pos = pd.read_csv('./manhattan_touchdown_metadata_nodes.tsv', delimiter='\t')
            self.navi_routes=[]
            for point in self.task['route_panoids']:
                info = img2pos[img2pos['pano_id'] == point]
                self.navi_routes.append((self.new_lat_scale(info['coords.lat'].values[0]), self.new_long_scale(info['coords.lng'].values[0])))
            trun=-1
            for idx,pos in enumerate(self.navi_routes):
                if idx==0:
                    continue
                if len(self.find_adjacent(pos)) >= 3:
                    trun=idx+1
                    break
            if trun==-1:
                raise Exception('this task has no intersectin')
            self.navi_routes = self.navi_routes[:trun]
            if self.task['start_heading']!='':
                self.pre_deg = float(self.task['start_heading'])
            else:
                self.pre_deg = random.randint(0,359)
            if self.task['end_heading']!='':
                self.post_deg = float(self.task['end_heading'])
            else:
                self.post_deg = random.randint(0,359)
            if len(self.navi_routes)<=1:
                raise Exception('No task')
                    

        depth=0
        vertices=set()
        vfilt = self.G.new_vertex_property('bool');
        new_dict={}


        
        Wall_Street = [(-89.8135819320945, -86.99411624217562),7224]
        Union_Square = [(-21.67198697935339, -22.411925691329387),15525]
        Hudson_River = [(85.92341169389903, -12.685304196935363),18085]
        CMU = [(-26.740129338424083, 93.62126227277517), 15947]
        Allegheny = [(95.86598041695194, -64.54825006689578), 14073]
        South_Shore = [(-45.15996521572513, -55.452800619759564), 14967]


        if self.city == 'Wall_Street':    
            chosen_city = Wall_Street
        elif self.city == 'Union_Square':
            chosen_city = Union_Square
        elif self.city == 'Hudson_River':
            chosen_city = Hudson_River
        elif self.city == 'CMU':
            chosen_city = CMU
        elif self.city == 'Allegheny':
            chosen_city = Allegheny
        elif self.city == 'South_Shore':
            chosen_city = South_Shore
        else:
            chosen_city = None
        if chosen_city!=None:
            for e in gt.bfs_iterator(self.G, self.G.vertex(self.Gdict[chosen_city[0]])):
                vertices.add(e.source())
                vertices.add(e.target())
                vfilt[e.source()] = True
                vfilt[e.target()] = True
                new_dict[self.Greversed[e.source()]]=e.source()
                new_dict[self.Greversed[e.target()]]=e.target()
                if len(vertices)==chosen_city[1]:
                    break
            Gdict=new_dict
            Greversed={value: key for key, value in Gdict.items()}
            sub = gt.GraphView(self.G, vfilt)
            self.G = sub
            self.Gdict = Gdict
            self.Greversed=Greversed
        print(self.G)
        # self.random_source=[]
        # for i in range(4):
        #     self.random_source.append(self.sample_source((-21.75919854376224, -22.524020509970285))[0])
        plt.savefig("filename5.png")


    def read_routes(self, csv_file="beogym/data/pano_gps.csv"):
        data = pd.read_csv(csv_file, keep_default_na=False, dtype='str')
        if app_config.PANO_IMAGE_LIMIT:
            data = data[:app_config.PANO_IMAGE_LIMIT]
        # self.build_graph(data,csv_file)
        self.build_6city()

    def find_adjacent(self, pos, action="next"):
        # print("Finding next position based on action/direction and position \n")
        if action == "next":
            return [self.Greversed[i] for i in self.G.vertex(self.Gdict[pos]).all_neighbors()]  # Return list of keys of nodes adjacent to node with key pos.

    def reset(self,pos=None):
        self.visited_locations=set()
        self.path=[]
        self.route_loc=0
        self.path_loc=1
        if self.route==[]:
            self.route=[random.choice(list(self.Gdict.keys()))]
            while len(self.route)<30:
                temp=random.choice(list(self.Gdict.keys()))
                dis=gt.shortest_distance(self.G, source=self.G.vertex(self.Gdict[self.route[-1]]), target=self.G.vertex(self.Gdict[temp]), weights=self.G.ep['weight'])
                if dis>=200 and dis<=900:
                    self.route.append(temp)
                    print(temp)
            print(self.route)
        
        if pos!=None:
            return pos,0

        if self.city=='Wall_Street':    
            return self.sample_source((-91.26371576373391, -92.93867163751861))
        elif self.city=='Union_Square':
            # return random.choice(self.random_source),9999
            # return (-21.75919854376224, -22.524020509970285)
            return self.sample_source((-21.75919854376224, -22.524020509970285))
        elif self.city=='Hudson_River':
            return self.sample_source((82.75054306685817, -4.9921430896916235))
        elif self.city == 'CMU':
            return self.sample_source((-26.494148909291482, 98.54096082676443))
        elif self.city == 'Allegheny':
            return self.sample_source((98.11349718215925, -65.91599501277392))
        elif self.city == 'South_Shore':
            return self.sample_source((-40.86770901528804, -57.97858763520175))
        elif 'navigation' in self.city:
            return self.sample_source(self.navi_routes[0])
            

        # return (-61.63857417592947, -64.93267436677775)
        # random_loc = random.choice(list(self.Gdict.keys()))
        # print('random')
        # print(random_loc)
        # return random_loc
        #return (25.435566549339654, 44.19074421319752)
        #return self.route[0]
        #return (1.3984153906729517, -24.730699803708518)
        #return (-7.173385138661828, -18.22405017381506)
        # random start
        # return random.choice(self.end_points)

    def sample_source(self,goal):
        while True:
            temp = self.sample_location()
            dis=gt.shortest_distance(self.G, source=self.G.vertex(self.Gdict[temp]), target=self.G.vertex(self.Gdict[goal]), weights=self.G.ep['weight'])
            if dis>=1200 and dis<=2000:
                long=dis
                break
        return temp,long

    def sample_location(self):

        return random.choice(list(self.Gdict.keys()))
        #location_list = [keys for keys, values in self.G.nodes.items()]
        #location = random.choice(location_list)

        #return location

    # Function to find the distances to adjacent nodes.
    # This is used to check to see if the node found is actually the nearest node.

    def find_distances(self, pos, adj_nodes_list):

        distance_list = []

        for node in adj_nodes_list:
            #distance_list.append(np.linalg.norm(np.array(pos) - np.array(node)))
            distance_list.append(self.G.ep['weight'][self.G.edge(self.G.vertex(self.Gdict[pos]),self.G.vertex(self.Gdict[node]))])

        return distance_list

        for node in adj_nodes_list:
            #distance_list.append(np.linalg.norm(np.array(pos) - np.array(node)))
            distance_list.append(self.G.ep['weight'][self.G.edge(self.G.vertex(self.Gdict[pos]),self.G.vertex(self.Gdict[node]))])

        return distance_list

    def fix_angle(self, angle):

        if angle < 0:
            angle = 360 + angle
        elif angle >= 360:
            angle = angle - 360

        return angle


    def get_distance(self,unit1, unit2):
        phi = abs(unit2 - unit1) % 360
        sign = 1
        # used to calculate sign
        if not ((unit1 - unit2 >= 0 and unit1 - unit2 <= 180) or (
                unit1 - unit2 <= -180 and unit1 - unit2 >= -360)):
            sign = -1
        if phi > 180:
            result = 360 - phi
        else:
            result = phi
        return result * sign

    def get_angle(self, p1, p2):
        res = math.degrees(math.atan2(p2[0] - p1[0], p2[1] - p1[1]))
        # return res
        if res < 90 and res >= 0:
            return 90 - res
        elif res >= 90 and res <= 180:
            return 360 - (res - 90)
        else:
            return 90 - res


    def get_all_paths(self, source, goal, max_l):
        res=[]
        for path in gt.all_paths(self.G, self.G.vertex(self.Gdict[source]), self.G.vertex(self.Gdict[goal]), max_l):
            res.append(path)
        return res
    # Convention we are using: in the angle_range, the first value always represent the right boundary of the cone.
    # While the second value represent the left boundary of the cone.
    # This function return 1 if angle is in range, 0 if not.

    def angle_in_range(self, angle, angle_range):
        # This is the case where the fix_angle adjusted the angle to be only from 0 to 360.
        if angle_range[0] > angle_range[1]:
            if angle < angle_range[1] or angle > angle_range[0]:
                return 1
            else:
                return 0
        # This is the regular case:
        else:
            if angle > angle_range[0] and angle < angle_range[1]:
                return 1
            else:
                return 0


    def find_nearest(self, curr_pos, prev_pos, curr_angle, direction):

        center_angle = self.fix_angle(curr_angle)


        # Check the current view angle against the search angle range:
        if direction == "forward":

            left_bound = self.fix_angle(center_angle + 30)
            right_bound = self.fix_angle(center_angle - 30)

            if self.angle_in_range(curr_angle, (right_bound, left_bound)):
                search_angle_range = (right_bound, left_bound)
            else:
                search_angle_range = (left_bound, right_bound)

        elif direction == "backward":

            left_bound = self.fix_angle(self.fix_angle(center_angle + 180) + 30)
            right_bound = self.fix_angle(self.fix_angle(center_angle + 180) - 30)

            if self.angle_in_range(curr_angle, (right_bound, left_bound)):
                search_angle_range = (left_bound, right_bound)
            else:
                search_angle_range = (right_bound, left_bound)

        next_pos_list = self.find_adjacent(curr_pos)  # This is a list of adjacent nodes to node agents_pos_1
        decision = curr_pos

        filtered_pos_list = []
        # Filtering the adjacent nodes by angle cone.
        for pos in next_pos_list:
            # Getting the angle between the current nodes and all adjacent nodes.
            angle = self.fix_angle(self.get_angle(curr_pos, pos))

            if self.angle_in_range(angle, search_angle_range):
                filtered_pos_list.append(pos)

        if (len(filtered_pos_list) == 0):
            mark=1
        else:
            mina=361
            decision=0
            for i in filtered_pos_list:
                angle = self.fix_angle(self.get_angle(curr_pos, i))
                diff=self.fix_angle(abs(curr_angle-angle))
                if diff<=mina:
                    decision=i
                    mina=diff
            #filtered_distances_list = self.find_distances(curr_pos, filtered_pos_list)
            #if len(filtered_distances_list) > 1:
            #    second_smallest_min_value = sorted(filtered_distances_list)[0]
            #else:
            #    second_smallest_min_value = filtered_distances_list[0]
            #decision = filtered_pos_list[filtered_distances_list.index(second_smallest_min_value)]

        return decision, decision, center_angle

    # The next two functions help in the render method.

    def draw_angle_cone(self, curr_pos, angle, color='m'):

        x = curr_pos[0]
        y = curr_pos[1]

        angle_range = [self.fix_angle(angle - 45), self.fix_angle(angle + 45)]
        line_length = 50

        for angle in angle_range:
            end_y = y + line_length * math.sin(math.radians(angle))
            end_x = x + line_length * math.cos(math.radians(angle))

            self.canvas.axes.plot([x, end_x], [y, end_y], ':' + color)

        self.canvas.draw()

    def update_plot(self, curr_pos, goal):
        if app_config.HEADLESS_MODE:
            return

        y = curr_pos[1]
        x = curr_pos[0]

        self.ydata = self.ydata + [y]
        self.xdata = self.xdata + [x]

        x_nodes = [x[0] for x in self.allNodes.keys()]
        y_nodes = [y[1] for y in self.allNodes.keys()]
        self.canvas.axes.cla()  # Clear the canvas.

        self.canvas.axes.scatter(x_nodes, y_nodes, c='blue', s=10)
        self.canvas.axes.scatter([x],[y], c='red', s=10)

        if len(self.ydata) > 0 or len(self.xdata) > 0:
            self.canvas.axes.scatter([self.xdata[0]], [self.ydata[0]], c='green', s=10)

        if goal != None:
            self.canvas.axes.scatter([goal[0]], [goal[1]], c='orange', s=10)

        self.canvas.axes.set_xlim([x-5, x+5])
        self.canvas.axes.set_ylim([y-5, y+5])

        self.canvas.draw()
        self.canvas.show()

    def bird_eye_view(self, curr_pos, radius):
        in_range_nodes_list = []
        temp = [key for key, value in self.G[curr_pos].items()]
        next = []
        for i in range(10):
            for q in temp:
                next += [key for key, value in self.G[q].items()]
            in_range_nodes_list += next
            temp = next
            next = []
        in_range_nodes_list = list(set(in_range_nodes_list))

        if len(in_range_nodes_list) == 0:
            print("No nodes found in range for bird eye's view.")
            return None

        bird_eye_graph = self.G.subgraph(in_range_nodes_list)

        return bird_eye_graph

    def draw_bird_eye_view(self, curr_pos, radius, graph, curr_angle):
        p_1 = []
        p_2 = []
        for i in graph.edges():
            p_1.append([i[0][0] - curr_pos[0], i[0][1] - curr_pos[1]])
            p_2.append([i[1][0] - curr_pos[0], i[1][1] - curr_pos[1]])

        p_1 = np.array(p_1)
        p_2 = np.array(p_2)

        new_pos, _, new_angle = self.find_nearest(curr_pos, curr_pos, curr_angle, "forward")
        agl = self.get_angle(new_pos, curr_pos)
        theta = np.deg2rad(agl)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        p_1 = np.dot(rotation_matrix, p_1.T).T
        p_2 = np.dot(rotation_matrix, p_2.T).T
        p_1 = -p_1
        p_2 = -p_2

        self.bev_graph.axes.clear()
        self.bev_graph.axes.plot([p_1[:, 0], p_2[:, 0]], [p_1[:, 1], p_2[:, 1]], '-or', linewidth=15)
        self.bev_graph.axes.plot(0, 0, color='green', marker='o')
        self.bev_graph.axes.text(0, 0, '({}, {})'.format(0, 0))
        self.bev_graph.axes.set_xlim(-2, +2)
        self.bev_graph.axes.set_ylim(0, 4)
        self.bev_graph.draw()
        self.bev_graph.show()
