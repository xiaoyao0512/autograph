import networkx as nx
import numpy as np
from collections import Counter
import scipy.stats as stats
#import torch

def node_dimension(G, weight=True):
    #G = G.to_networkx(edge_attrs=['weight'])
    G = nx.Graph(G).to_undirected()
    node_dimension = {}
    for node in G.nodes():
        #cutoff = 0
        #if (node == 0):
        #    cutoff = 3
        #elif (node > 0 and node < 3):
        #    cutoff = 2
        #elif (node > 2 and node < 7):
        #    cutoff = 2
        #else:
        #    cutoff = 3
        #print("node = ", node)
        grow = []
        r_g = []
        num_g = []
        num_nodes = 0
        if weight == None:
            spl = nx.single_source_shortest_path_length(G,node)
        else:
            spl = nx.single_source_dijkstra_path_length(G,node)
            #print("spl = ", spl)
        for s in spl.values():
            if s>0:
                grow.append(s)
        grow.sort()
        num = Counter(grow)
        for i,j in num.items():
            num_nodes += j
            if i>0:
                #if np.log(num_nodes) < 0.95*np.log(G.number_of_nodes()):
                r_g.append(i)
                num_g.append(num_nodes)
#                 # delete
#                 if np.log(num_nodes) > 0.9*np.log(G.number_of_nodes()):
#                     break
        #print("r_g = ", r_g)
        #print("num_g = ", num_g)
        x = np.log(r_g)
        y = np.log(num_g)
        #if len(r_g) < 3:
        #    print("local",node)
        #slope = 0
        if len(r_g) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            node_dimension[node] = [slope]
        else:
            node_dimension[node] = [0]
    return list(node_dimension.values())#torch.FloatTensor(list(node_dimension.values()))
    

