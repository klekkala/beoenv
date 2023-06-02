import graph_tool.all as gt
import pickle



G = gt.load_graph("nyc.gt")          
Gdict = pickle.load(open('nycgraph.pkl', 'rb'))
Greversed={value: key for key, value in Gdict.items()}

depth=0
vertices=set()

vfilt = G.new_vertex_property('bool');
new_dict={}

for e in gt.bfs_iterator(G, G.vertex(Gdict[(-89.8135819320945, -86.99411624217562)])):
    vertices.add(e.source())
    vertices.add(e.target())
    vfilt[e.source()] = True
    vfilt[e.target()] = True
    new_dict[Greversed[e.source()]]=e.source()
    new_dict[Greversed[e.target()]]=e.target()
    if len(vertices)==7224:
        break
print(len(vertices))
Gdict=new_dict
Greversed={value: key for key, value in Gdict.items()}
sub = gt.GraphView(G, vfilt)
print(sub)
print(Gdict)
# all_vertices = G.vertices()
# for v in all_vertices:
#     if v not in vertices:
#         G.remove_vertex(v)
# print(len(set(vertices)))