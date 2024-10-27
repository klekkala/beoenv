# from beogym import BeoGym
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
# env = BeoGym({})
new_min = -100
new_max = 100
lat_min = 40.707091
lat_max = 40.764838
long_min = -74.005022
long_max = -73.976614

def new_lat_scale(x):
    normalized_new_val = ((x - lat_min) / (lat_max - lat_min) * (new_max - new_min)) + new_min
    return normalized_new_val

def new_long_scale(x):
    normalized_new_val = ((x - long_min) / (long_max - long_min) * (new_max - new_min)) + new_min
    return normalized_new_val


file_path = '/lab/kiran/test.json'

nodes = pd.read_csv('manhattan_touchdown_metadata_nodes.tsv', delimiter='\t')
print(nodes.columns)

all_routes = []
all_text = []
all_angle = []
print('taraazz')
with open(file_path, 'r') as json_file:
    for line in json_file:
        data = json.loads(line)
        raw_routes = data['route_panoids']
        routes=[]
        for point in raw_routes:
            info = nodes[nodes['pano_id'] == point]
            pos = (new_lat_scale(info['coords.lat'].values[0]), new_long_scale(info['coords.lng'].values[0]))
            routes.append(pos)
        all_routes.append(routes)
        all_text.append(data['navigation_text'])
        all_angle.append(data['end_heading'])


map_x = nodes['coords.lng'].apply(new_long_scale)
map_y = nodes['coords.lat'].apply(new_lat_scale)

# all_routes = all_routes[13]



# xdata = [i[1] for i in all_routes]
# ydata = [i[0] for i in all_routes]

starts = [i[0] for i in all_routes]
ends = [i[-1] for i in all_routes]
start_x = [i[1] for i in starts]
start_y = [i[0] for i in starts]
end_x = [i[1] for i in ends]
end_y = [i[0] for i in ends]
plt.figure(figsize=(18, 16))
# plt.scatter(map_x, map_y, c='blue')

dx = [a - b for a, b in zip(end_x, start_x)]
dy = [a - b for a, b in zip(end_y, start_y)]
plt.scatter(dx, dy, c='red')
# plt.scatter(start_x, start_y, c='red')
# plt.scatter(end_x, end_y, c='green')


# Create a scatter plot
# plt.plot(xdata, ydata, label="Points", color="blue", marker="o", linestyle="-")

# Add labels and a title
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.title("Scatter Plot of Points")

# Add a legend
plt.legend()

# Show the plot
plt.savefig("dis_touch.png")

        # current_pos = (x,y)

        # adj_nodes_list = [keys for keys, values in self.o.G.adj[current_pos].items()]
        # num_adj_nodes = len(adj_nodes_list)
        # adj_nodes_list = np.array( [[x_coor, y_coor] for x_coor, y_coor in adj_nodes_list])


        # x_pos_list = np.array([x] * num_adj_nodes)
        # y_pos_list = np.array([y] * num_adj_nodes)

        # self.canvas.axes.plot([x_pos_list,adj_nodes_list[:,0]], [y_pos_list, adj_nodes_list[:,1]], '--or')
        # self.canvas.axes.plot(x, y, color = 'green', marker = 'o')
        # self.canvas.axes.text(x, y, '({}, {})'.format(x, y))
        # self.canvas.axes.plot(self.agents_pos_prev[0], self.agents_pos_prev[1], color = 'purple', marker = 'o')

        # self.draw_angle_cone(self.agents_pos_curr, self.curr_angle, color = 'g')
        # self.canvas.axes.set_xlim([-100, 100])
        # self.canvas.axes.set_ylim([-100, 100])

        # self.canvas.draw()



# env.shortest_rec_routes(pth+'train/',all_routes,all_text,all_angle)


#python main.py --train --eval --env_name beogym -set=Wall_Street --setting=singlegame --machine iGpu7

