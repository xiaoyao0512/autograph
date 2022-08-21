import programl as pg
import glob
import networkx as nx
from dgl.data.utils import save_graphs
import dgl

run_dir = "training_data-small"
graphs = []
for file in glob.glob(run_dir+"/*.c"):
    print(file)
    file_name = file.split('/')[1]
    if (file_name == "header.c"):
        print("lol")
        continue
    g = pg.from_clang([file])
    #print(type(g).__name__)
    nxg = pg.to_networkx(g)
    #print("nodes[0] = ", nxg.nodes[0])
    #print("nodes[1] = ", nxg.nodes[1])    
    #print("edges[1, 6, 0] = ", nxg.edges[1,6,0])
    g = dgl.from_networkx(nxg)
    graphs.append(g)
    #nx.write_gexf(nxg, "training_edge_list/"+file_name+".gexf")
    #break

save_graphs("graphs.bin", graphs)
