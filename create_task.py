from beogym.beogym import BeoGym
import json
import math
import matplotlib.pyplot as plt

NYC = ['Wall_Street','Union_Square', 'Hudson_River']
Pits= ['CMU', 'Allegheny', 'South_Shore']
cities = NYC+Pits

print(cities)
def euclidean_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

dX,dY=(40.384,47.536)
# dX,dY=(15,85)
maxdX=[]
maxdY=[]
with open('task.json', 'w') as f:
    for idx,i in enumerate(cities):
        env = BeoGym({'city':i,'data_path':'/home6/tmp/kiran/'})
        temp = {'no.':1, 'task_id':idx, 'city':i}
        if i=='Wall_Street':  
            courier_goal= (-96.8055251689076, -94.02399239301302)
            #(-91.26371576373391, -92.93867163751861)
        elif i=='Union_Square':
            # courier_goal = (-36.14906196121752, -57.522601891088236) #center
            courier_goal = (-56.353110976479435, -52.67864397355789)
            # courier_goal = (-21.75919854376224, -22.524020509970285)
        elif i=='Hudson_River':
            courier_goal = (32.34645696528139, 13.539046591584182) #center
            # courier_goal = (17.013602534793975, -71.66840979149978)
            #(82.75054306685817, -4.9921430896916235)
        elif i == 'CMU':
            courier_goal = (-4.421576280722888, 38.25098452572334) #center
            # courier_goal = (-34.20243940774522, 13.512665953129144)
            #(-26.494148909291482, 98.54096082676443)
        elif i == 'Allegheny':
            courier_goal = (32.03421933355679, -81.37135677147891) #center
            # courier_goal = (-27.688091834544252, -96.02609714096437)
            #(98.11349718215925, -65.91599501277392)
        elif i == 'South_Shore':
            # courier_goal = (-99.62801690964798, -93.04901822643482)
            courier_goal = (-49.372566159171285, -72.84128179076082) #center
            #(-40.86770901528804, -57.97858763520175)
        closest_distance = float('inf')
        closest_pos = None
        for pos in env.dh.Gdict.keys():
            dis = euclidean_distance(pos, (courier_goal[0]+dX, courier_goal[1]+dY))
            # dis = euclidean_distance(pos, (-100, -100))
            if dis < closest_distance:
                closest_pos = pos
                closest_distance = dis
        temp['goal'] = courier_goal
        temp['source'] = closest_pos
        maxdX.append(closest_pos[0]-courier_goal[0])
        maxdY.append(closest_pos[1]-courier_goal[1])
        print(closest_pos)
        print(courier_goal)
        temp['routes'] = env.dh.getShortestPathNodes(closest_pos, courier_goal)
        temp['reward'] = [1 if len(env.dh.find_adjacent(i))>=3 else 0 for i in temp['routes']]
        print(len(temp['routes']))
        print(len(temp['reward']))
        # getShortestPathNodes


        all_node = []
        for q in env.dh.Gdict.keys():
            if q not in temp['routes']:
                all_node.append(q)
        x,y = zip(*all_node)
        x_r,y_r = zip(*temp['routes'])
        plt.scatter(x, y, color='blue', s=1)
        plt.scatter(x_r, y_r, color='red', s=1)
        plt.legend()
        plt.savefig('./maps/'+i+".png")
        plt.clf()

        del env
        json.dump(temp, f)
        f.write('\n')

print(maxdX)
print(maxdY)


# dX,dY=(28.606923655052135, -38.298098878997024)
# maxdX=[]
# maxdY=[]
# with open('task.json', 'w') as f:
#     for idx,i in enumerate(cities):
#         env = BeoGym({'city':i,'data_path':'/home6/tmp/kiran/'})
#         temp = {'no.':1, 'task_id':idx, 'city':i}
#         if i=='Wall_Street':  
#             courier_goal= (-80.39042704760945, -31.047069791662082)
#             #(-91.26371576373391, -92.93867163751861)
#         elif i=='Union_Square':
#             courier_goal = (-41.60343624636463, 28.359422740943074)
#             # courier_goal = (-21.75919854376224, -22.524020509970285)
#         elif i=='Hudson_River':
#             courier_goal = (30.497565333121173, 51.72475046700376)
#             #(82.75054306685817, -4.9921430896916235)
#         elif i == 'CMU':
#             courier_goal = (-96.93842345120045, 76.17579409219312)
#             #(-26.494148909291482, 98.54096082676443)
#         elif i == 'Allegheny':
#             courier_goal = (-10.776231027346142, -33.24376751033488)
#             #(98.11349718215925, -65.91599501277392)
#         elif i == 'South_Shore':
#             courier_goal = (-82.49770148844517, 11.12157812439672)
#             #(-40.86770901528804, -57.97858763520175)
#         closest_distance = float('inf')
#         closest_pos = None
#         for pos in env.dh.Gdict.keys():
#             dis = euclidean_distance(pos, (courier_goal[0]+dX, courier_goal[1]+dY))
#             # dis = euclidean_distance(pos, (-100, -100))
#             if dis < closest_distance:
#                 closest_pos = pos
#                 closest_distance = dis
#         temp['goal'] = courier_goal
#         temp['source'] = closest_pos
#         maxdX.append(closest_pos[0]-courier_goal[0])
#         maxdY.append(closest_pos[1]-courier_goal[1])
#         print(closest_pos)
#         print(courier_goal)
#         temp['routes'] = env.dh.getShortestPathNodes(closest_pos, courier_goal)
#         temp['reward'] = [1 if len(env.dh.find_adjacent(i))>=3 else 0 for i in temp['routes']]
#         print(len(temp['routes']))
#         print(len(temp['reward']))
#         # getShortestPathNodes
#         del env
#         json.dump(temp, f)
#         f.write('\n')

# print(maxdX)
# print(maxdY)