import pickle
import time

import plyvel,shutil

from MplCanvas import MplCanvas
import Equirec2Perspec as E2P
import Equirec2Perspec_cpu as E2PC

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import cv2
import numpy as np
import cupy as cp
import math
import os
import h5py
from math import radians, cos, sin, asin, acos, sqrt, pi

import config as app_config

new_min = -100
new_max = 100
lat_min = 40.701149
lat_max = 40.787206
long_min = -74.018179
long_max = -73.947363

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

    def __init__(self, csv_file, hov_val,isFile=False):

        self.G = nx.Graph()
        self.end_points = []
        # try:
        #     self.db = plyvel.DB('data/')
        # except IOError:

        self.db = ''
        count = 0
        while self.db == '':
            try:
                if os.path.exists('data' + str(count) + '/'):
                    self.db = plyvel.DB('data' + str(count) + '/')
                else:
                    shutil.copytree('data/', 'data' + str(count) + '/')
                    self.db = plyvel.DB('data' + str(count) + '/')
            except IOError:
                count += 1

        # Create canvas for plot rendering:
        self.canvas = MplCanvas(self, width=8, height=7, dpi=100)
        self.bev_graph = MplCanvas(self, width=5, height=4, dpi=100)
        self.xdata = []
        self.ydata = []
        self.tm = [0]
        f_read = open('headoff.pkl', 'rb')
        self.headOff = pickle.load(f_read)
        f_read.close()
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

        #self.menu = h5py.File(f"hd5_files/coordinate_file_map.hdf5", 'r')
        #self.detail={}
        #for i in range(1,7):
            #self.detail[bytes("hd5_files/sector_"+str(i)+"_rgb.hdf5",'utf-8')] = h5py.File('hd5_files/sector_'+str(i)+'_rgb.hdf5', 'r')

        # Dictionary used to get the shortest path using latlong positions
        self.allNodes = dict(self.G.nodes(data="image_name"))

    def getShortestPath(self, start, end):
        posPath = nx.shortest_path(self.G, source=start, target=end,
                                   weight="weight")
        nodePath = []
        for x in posPath:
            nodePath.append(self.allNodes[x])
        return nodePath

    def getShortestPathNodes(self, start, end):
        posPath = nx.shortest_path(self.G, source=start, target=end,
                                   weight="weight")
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
        # print(curr_image[0][0][0])

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

    def panorama_split(self, theta, curr_pos, resolution):
        # global coord_to_sect
        # global scaled_coord_to_coord
        # global coord_to_filename

        # coord = scaled_coord_to_coord[(image_coord[0], image_coord[1])]
        # coord_str = ','.join([str(value) for value in coord])

        # filename = coord_to_filename[coord_str]


        # file_path = f"{app_config.CAMERA_IMAGES_PATH}/{image_name}.jpg"

        # path = coord_to_sect[coord_str]
        # f = h5py.File(f"../{path}", "r")
        # curr_image = f[coord_str]["rgb_pano"]

        # image_file = "%s.png" % str(image)
        print("\n")
        # print("Showing image: ", file_path)
        print("Current viewing angle: ", theta)
        print("\n")
        stream_done: bool = cp.cuda.get_current_stream().done
        if stream_done:
            equ = E2P.Equirectangular(curr_pos,self.db, self.tm)  # Load equirectangular image
        else:
            equ = E2PC.Equirectangular(curr_pos,self.db)
        print(self.tm)
        #
        # FOV unit is degree
        # theta is z-axis angle(right direction is positive, left direction is negative)
        # phi is y-axis angle(up direction positive, down direction negative)
        # height and width is output image dimension
        #
        img = equ.GetPerspective(450, theta - self.headOff[curr_pos], 0, *resolution)  # Specify parameters(FOV, theta, phi, height, width)
        midpoint = (int) (img.shape[1] / 2)
        img = img[:, (midpoint - self.hov):(midpoint + self.hov)]
        return img

    def build_graph(self, data, csv_file, isRoute=False):
        nyc = True
        i = 0
        prev_route = -1
        prev_pos = (-1, -1)
        prev_name = "abc"
        x = []
        y = []
        csv_file=csv_file.split('/')[-1]
        localMap=csv_file.split('.')[0]+'.gpickle'
        print("\n")
        print("Building graph. \n")
        # for index, row in data.iterrows():
        #     route_no = row['route_no']
        #     lat = float(row['latitude'])
        #     long = float(row['longitude'])
        #     scaled_lat = self.new_lat_scale(lat)
        #     scaled_long = self.new_long_scale(long)
        #     image_name = row['image_name']
        #     current_node = (scaled_lat, scaled_long)
        #     scaled_coord_to_coord[current_node] = (lat, long,)
        #     if (image_name not in self.G):
        #         self.G.add_node((scaled_lat, scaled_long), image_name=image_name, latitude=lat, longitude=long,
        #                         yaw=row['yaw'], )  # saves a node as image name
        #         # print((scaled_lat,scaled_long), image_name)
        #         # Why is prev_pos only compares to one integer while it is a tuple?
        #         if isRoute==True:
        #             if route_no == prev_route and prev_pos != (-1, -1):
        #                 # So the edges only connect nodes of the same route?
        #                 # print("adding edge")
        #                 # What are these x, y lists for? Look at the elif below.
        #                 x.append(scaled_lat)
        #                 y.append(scaled_long)
        #
        #                 # Calculate the distance between the two nodes
        #                 edgeDist = calcDist(prev_pos, current_node)
        #                 self.G.add_edge(prev_pos, current_node,
        #                                 weight=edgeDist)  # need to add something like a direction on this edge like right left straight...
        #
        #             elif route_no != prev_route:  ## going to a new route
        #                 plt.plot(x, y, '-o', linewidth=1, markersize=2)  # x and y used to plot the previous route.
        #                 if (prev_pos != (-1, -1)):  # end_points mean the end point of each route.
        #                     self.end_points.append(prev_pos)
        #                 x = []
        #                 y = []
        #     prev_pos = current_node
        #     prev_route = route_no
        # save the graph as a json?

        if os.path.isfile(localMap):
            self.G = nx.read_gpickle(localMap)
            self.end_points=pickle.load(open(localMap.split('.')[0]+'.pickle','rb'))
            self.image_names = nx.get_node_attributes(self.G, 'image_name')
            print(self.G)
        if nyc == True and not os.path.isfile(localMap):
            links = pd.read_csv('data/manhattan_metadata_links.tsv', delimiter='\t')
            nodes = pd.read_csv('data/manhattan_metadata_nodes.tsv', delimiter='\t')
            nodes = nodes[['pano_id', 'coords.lat', 'coords.lng']]

            img2loc = {}
            for i in nodes.itertuples():
                img2loc[i[1]] = (i[2], i[3])


            for i in links.itertuples():
                point1 = (self.new_lat_scale(img2loc[i[1]][0]),self.new_long_scale(img2loc[i[1]][1]))
                point2 = (self.new_lat_scale(img2loc[i[2]][0]),self.new_long_scale(img2loc[i[2]][1]))
                self.G.add_edge(point1, point2, weight=calcDist(point1, point2))
                self.end_points.append(point2)

            nx.write_gpickle(self.G, localMap)
            pickle.dump(self.end_points, open(localMap.split('.')[0] + '.pickle', 'wb'))
        if isRoute==False and not os.path.isfile(localMap) and not nyc:
            allDis={}
            isOpen=[1 for i in range(len(list(self.G.nodes)))]
            openNodes=list(self.G.nodes)
            for i in range(len(isOpen)):
                temp={}
                for q in range(len(isOpen)):
                    if i < q:
                        temp[openNodes[q]]=calcDist(openNodes[i],openNodes[q])
                    elif i > q:
                        temp[openNodes[q]] = allDis[openNodes[q]][openNodes[i]]
                allDis[openNodes[i]]=temp
            while sum(isOpen)!=0:
                temp = isOpen.index(1)
                cloestNode=-1
                shortestDistance=float('inf')
                for i in range(len(isOpen)):
                    if i != temp:
                        tempDis=allDis[openNodes[i]][openNodes[temp]]
                        if tempDis < shortestDistance:
                            cloestNode = i
                            shortestDistance = tempDis
                isOpen[cloestNode]=0
                isOpen[temp]=0
                self.G.add_edge(openNodes[cloestNode],openNodes[temp],weight=shortestDistance)


            self.image_names = nx.get_node_attributes(self.G, 'image_name')
            subGraph = nx.connected_components(self.G)
            subGraph = list(subGraph)
            subGraphDis={}
            for i in range(len(subGraph)):
                temp={}
                for w in range(i + 1, len(subGraph)):
                    srtest = float('inf')
                    point1 = -1
                    point2 = -1
                    for q in range(len(list(subGraph[i]))):
                        for p in range(len(list(subGraph[w]))):
                            if allDis[list(subGraph[i])[q]][list(subGraph[w])[p]]<srtest:
                                srtest=allDis[list(subGraph[i])[q]][list(subGraph[w])[p]]
                                point1=list(subGraph[i])[q]
                                point2=list(subGraph[w])[p]
                    temp[w] = [srtest,point1,point2]
                    self.end_points.append(point2)
                subGraphDis[i]=temp

            temp=nx.Graph()
            for i in range(len(subGraph)):
                temp.add_node(i)

            while len(list(nx.connected_components(self.G)))>1:
                print(self.G)
                srtest = float('inf')
                point1 = -1
                point2 = -1
                pos1=-1
                pos2=-2
                for i,j in subGraphDis.items():
                    for q,w in j.items():
                        if w[0] < srtest and not nx.has_path(temp,i,q):
                            srtest = w[0]
                            point1 = w[1]
                            point2 = w[2]
                            pos1 = i
                            pos2 = q
                self.G.add_edge(point1, point2, weight=srtest)
                temp.add_edge(pos1,pos2)

            nx.write_gpickle(self.G, localMap)
            pickle.dump(self.end_points,open(localMap.split('.')[0]+'.pickle','wb'))
        print(self.G)
        plt.savefig("filename5.png")

    def read_routes(self, csv_file="data/test.csv"):
        data = pd.read_csv(csv_file, keep_default_na=False, dtype='str')
        if app_config.PANO_IMAGE_LIMIT:
            data = data[:app_config.PANO_IMAGE_LIMIT]
        self.build_graph(data,csv_file)

    def find_adjacent(self, pos, action="next"):
        # print("Finding next position based on action/direction and position \n")
        if action == "next":
            # print(self.end_points)
            # print("Current node: \n", pos)
            # print("Adjacent nodes and edges: \n", (self.G.adj[pos])) # Finding adjacent nodes and edges to pos node.
            # adj_nodes_list = [keys for keys,value in self.G.adj[pos].items()]
            # print("Coordinate of the adjacent nodes: \n", adj_nodes_list)
            # return list([keys for keys, value in self.G.nodes.items()])  # HERE
            return [keys for keys, value in
                         self.G[pos].items()]  # Return list of keys of nodes adjacent to node with key pos.

    def reset(self):
        # reset the position of the agent
        print("Resets the position to a start \n")
        # i = random.choice(range(len(self.end_points)))
        i = 0  # HERE
        return random.choice(self.end_points)

    def sample_location(self):

        location_list = [keys for keys, values in self.G.nodes.items()]
        location = random.choice(location_list)

        return location

    # Function to find the distances to adjacent nodes.
    # This is used to check to see if the node found is actually the nearest node.

    def find_distances(self, pos, adj_nodes_list):

        distance_list = []

        for node in adj_nodes_list:
            #distance_list.append(np.linalg.norm(np.array(pos) - np.array(node)))
            distance_list.append(self.G[pos][node]['weight'])

        return distance_list

    def fix_angle(self, angle):

        if angle < 0:
            angle = 360 + angle
        elif angle >= 360:
            angle = angle - 360

        return angle

    # This function should also convert from triangular to abosulute angle?

    # def get_angle(self, curr, prev):
    #     if (curr[0] - prev[0]) != 0:
    #         slope = (curr[1] - prev[1]) / (curr[0] - prev[0])
    #     else:
    #         return 0
    #     # print(slope)
    #     angle = math.degrees(math.atan(slope))
    #     # The direction is from the second to the fourth quadrant.
    #     # So angle is negative.
    #     if (curr[0] > prev[0] and curr[1] < prev[1]):
    #         angle = 360 + angle
    #     # Direction: from fourth to second quadrant.
    #     # Angle is negative.
    #     elif (curr[0] < prev[0] and curr[1] > prev[1]):
    #         angle = 180 + angle
    #     # Direction: from first to third.
    #     # Angle is positive.
    #     elif (curr[0] < prev[0] and curr[1] < prev[1]):
    #         angle = 180 + angle
    #
    #     # angle = fix_angle(angle)
    #     return angle

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
        res = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        return res
        # if res < 90 and res >= 0:
        #     return 90 - res
        # elif res >= 90 and res <= 180:
        #     return 360 - (res - 90)
        # else:
        #     return 90 - res

    def get_angle_plot(self, p1, p2):
        res = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        if res < 90 and res >= 0:
            return 90 - res
        elif res >= 90 and res <= 180:
            return 360 - (res - 90)
        else:
            return 90 - res

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

    # Note on the process of finding the nearest node:
    # My speculation:
    # 1. Find the current angle cone of the agent, which is where the agent is looking in absolute angles.
    # 2. Then get the adjacent nodes' absolute angles. Note: adjacent is defined as being connected by an edge.
    # 3. Filter the adjacent nodes by fov cone using the abosolute angles.
    # 4. Move to the nearest node within the cone.
    # Note: Process of graph creation: Dynamic_plot.py called build_graph. Build_graph go through every line
    # of the csv file then add all the nodes. What about edges?

    def find_nearest(self, curr_pos, prev_pos, curr_angle, direction):
        print("\n")

        # This is the view angle.
        center_angle = self.fix_angle(curr_angle)

        # The search angle is based on positions. Independent of viewing angle.
        search_angle = self.get_angle(curr_pos, prev_pos)

        # left_bound = self.fix_angle(search_angle+90)
        # right_bound = self.fix_angle(search_angle-90)

        # Check the current view angle against the search angle range:
        if direction == "forward":

            left_bound = self.fix_angle(center_angle + 45)
            right_bound = self.fix_angle(center_angle - 45)

            if self.angle_in_range(curr_angle, (right_bound, left_bound)):
                search_angle_range = (right_bound, left_bound)
            else:
                search_angle_range = (left_bound, right_bound)

        elif direction == "backward":

            left_bound = self.fix_angle(self.fix_angle(center_angle + 180) + 45)
            right_bound = self.fix_angle(self.fix_angle(center_angle + 180) - 45)

            if self.angle_in_range(curr_angle, (right_bound, left_bound)):
                search_angle_range = (left_bound, right_bound)
            else:
                search_angle_range = (right_bound, left_bound)

        print("Current center angle: ", center_angle)
        next_pos_list = self.find_adjacent(curr_pos)  # This is a list of adjacent nodes to node agents_pos_1
        decision = curr_pos

        # image_name = self.image_name(curr_pos)

        # print("Possible next nodes: ", len(next_pos_list))
        # print("List of adjacent nodes: ", next_pos_list)
        # print("Distances from current node to the adjacent nodes: ", self.find_distances(curr_pos, next_pos_list))
        print("Search angle range: ", search_angle_range)
        filtered_pos_list = []
        # Filtering the adjacent nodes by angle cone.
        for pos in next_pos_list:
            # Getting the angle between the current nodes and all adjacent nodes.
            angle = self.fix_angle(self.get_angle(curr_pos, pos))
            print('potential  ' + str(angle) + '  ' + str(pos) + '  ' + str(curr_angle))
            # print("Angle from ", curr_pos,"to ", pos, "is ", angle)

            if self.angle_in_range(angle, search_angle_range):
                filtered_pos_list.append(pos)

        # print("Filtered adjacent nodes list: ", filtered_pos_list)
        if (len(filtered_pos_list) == 0):
            print("\n")
            print("No nodes found. Agent standing still.")
        else:
            filtered_distances_list = self.find_distances(curr_pos, filtered_pos_list)
            # print("#"*100)
            # print(filtered_distances_list)
            # print("#"*100)
            # print("Distances from current node to the filtered adjacent nodes: ", filtered_distances_list)
            if len(filtered_distances_list) > 1:
                second_smallest_min_value = sorted(filtered_distances_list)[1]
            else:
                second_smallest_min_value = filtered_distances_list[0]
            # print("Index of min value: ", (min(filtered_distances_list)))
            # print("Index of min value: ", second_smallest_min_value)
            decision = filtered_pos_list[filtered_distances_list.index(second_smallest_min_value)]
            print(decision)
            print("The nearest node within the angle cone is: ", decision)
            print("Found a node within the angle cone. New node position: ", decision)
            # image_name = self.image_name(decision)
            # print("Showing new node's image: ", image_name)

        # self.panorama_split(center_angle, image_name)
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

        # adjacent_pos_list = self.find_adjacent(curr_pos)
        # adjacent_pos_list.append(curr_pos)
        # distances_list = self.find_distances(curr_pos, adjacent_pos_list)
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
        # for distance, pos in zip(distances_list, adjacent_pos_list):

        # if distance <= radius:
        # in_range_nodes_list.append(pos)

        if len(in_range_nodes_list) == 0:
            print("No nodes found in range for bird eye's view.")
            return None

        bird_eye_graph = self.G.subgraph(in_range_nodes_list)

        return bird_eye_graph

    def draw_bird_eye_view(self, curr_pos, radius, graph, curr_angle):
        print('im pringtin')
        # self.bev_graph.axes.cla()

        # nodes_list = [keys for keys, values in graph.nodes().items()]

        p_1 = []
        p_2 = []
        for i in graph.edges():
            p_1.append([i[0][0] - curr_pos[0], i[0][1] - curr_pos[1]])
            p_2.append([i[1][0] - curr_pos[0], i[1][1] - curr_pos[1]])
            # x_1.append(i[0][0])
            # x_2.append(i[1][0])
            # y_1.append(i[0][1])
            # y_2.append(i[1][1])

        p_1 = np.array(p_1)
        p_2 = np.array(p_2)


        # nodes_list = np.array([[x_coor, y_coor] for x_coor, y_coor in nodes_list])

        new_pos, _, new_angle = self.find_nearest(curr_pos, curr_pos, curr_angle, "forward")
        agl = self.get_angle_plot(new_pos, curr_pos)
        theta = np.deg2rad(agl)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        p_1 = np.dot(rotation_matrix, p_1.T).T
        p_2 = np.dot(rotation_matrix, p_2.T).T
        # x = curr_pos[0]
        # y = curr_pos[1]
        p_1 = -p_1
        p_2 = -p_2

        self.bev_graph.axes.clear()
        # self.bev_graph.axes.plot([x_pos_list, nodes_list[:, 0]], [y_pos_list, nodes_list[:, 1]], '--or')
        # self.bev_graph.axes.plot([x_1, x_2], [y_1, y_2], '--or')
        self.bev_graph.axes.plot([p_1[:, 0], p_2[:, 0]], [p_1[:, 1], p_2[:, 1]], '-or', linewidth=15)
        self.bev_graph.axes.plot(0, 0, color='green', marker='o')
        self.bev_graph.axes.text(0, 0, '({}, {})'.format(0, 0))
        # self.bev_graph.axes.set_xlim([np.min(nodes_list[:, 0]), np.max(nodes_list[:, 0])])
        # self.bev_graph.axes.set_ylim([np.min(nodes_list[:, 1]), np.max(nodes_list[:, 1])])
        self.bev_graph.axes.set_xlim(-2, +2)
        self.bev_graph.axes.set_ylim(0, 4)

        # Draw a circle to see if the BEV is done correctly.
        # draw_circle = plt.Circle(curr_pos, radius=radius, fill=False)
        # self.bev_graph.axes.add_artist(draw_circle)

        self.bev_graph.draw()

        self.bev_graph.show()

    def distance_to_goal(self, curr_pos, goal):

        return np.linalg.norm(np.array(curr_pos) - np.array(goal))


