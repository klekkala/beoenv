import graph_tool.all as gt
import pickle
# create an empty directed graph

G = gt.load_graph("nyc.gt")

Gdict = pickle.load(open('nycgraph.pkl', 'rb'))
Greversed={value: key for key, value in Gdict.items()}


new = gt.Graph(directed=False)

wall_street = (-89.8135819320945, -86.99411624217562)



bfs_iterator, _ = gt.topology.bfs_iterator(G, G.vertex(Gdict[wall_street]),max_depth=200)

print(len(bfs_iterator))

