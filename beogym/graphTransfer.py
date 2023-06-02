import graph_tool.all as gt
import networkx as nx
import pickle
import time


g=gt.Graph(directed=False)

oG = pickle.load(open('pano_gps.gpickle', 'rb'))

idx=0
verdict={}
weight = g.new_edge_property("double")
for i in oG.edges(data=True):
    if i[0] not in verdict:
        v1=g.add_vertex()
        verdict[i[0]]=int(v1)
    else:
        v1=g.vertex(verdict[i[0]])
    if i[1] not in verdict:
        v2=g.add_vertex()
        verdict[i[1]]=int(v2)
    else:
        v2=g.vertex(verdict[i[1]])
    e = g.add_edge(v1, v2)
    weight[e] = i[2]['weight']

dist=gt.shortest_distance(g, source=g.vertex(0),target=g.vertex(500),weights=weight)
g.ep['weight'] = weight

g.save("nyc.gt")

with open("nycgraph.pkl", "wb") as f:
    pickle.dump(verdict, f)

print(dist)
