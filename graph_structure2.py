import pickle
import glob, csv
import re
from utility import find_loops
import pickle, json
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import large_clique_size
import mfa
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci


f = open('runtimes.pickle', 'rb')
runtimes = pickle.load(f)    
f.close()

#print("emb keys = ", emb.keys())

vf_if = {}
times = {}
files_VF_IF = runtimes.keys()
vf_list = []
if_list = []
for file_VF_IF in files_VF_IF:
    tmp = file_VF_IF.split('.')
    fn = tmp[0]
    tmp = tmp[1].split('-')
    VF = int(tmp[0])
    IF = int(tmp[1])
    vf_list.append(VF)
    if_list.append(IF)
    rt_mean = np.mean(runtimes[file_VF_IF])
    #print("filename = ", fn)
    #print("VF = ", VF)
    #print("IF = ", IF)
    #print("mean = ", rt_mean)
    if fn not in vf_if.keys():
        vf_if[fn] = (rt_mean, VF, IF)
    else:
        rt_mean_pre = vf_if[fn][0]
        if rt_mean < rt_mean_pre:
            vf_if[fn] = (rt_mean, VF, IF)
    if fn not in times.keys():
        times[fn] = {}
    if VF not in times[fn].keys():
        times[fn][VF] = {}
    if IF not in times[fn][VF].keys():
        times[fn][VF][IF] = rt_mean
    else:
        rt_mean_pre = times[fn][VF][IF]
        if (rt_mean < rt_mean_pre):
            times[fn][VF][IF] = rt_mean


run_dir = "training_data_default"
save_dir = "training_data_vec"
graphs = []
cnt = 0

for file in glob.glob("json-small/*.json"):
    graphs.append([])
    print(file)
    file_name = file.split('/')[1]

    g = None
    with open(file) as fh:
        g = nx.readwrite.json_graph.node_link_graph(json.load(fh))
        g = nx.Graph(g)
    
    # features
    #print("starting ...")
    graphs[cnt].append(file_name)

    # find the generalized dimension and spectrum width
    nodes = len(g)
    Q = [q for q in range(-10,11,1)]
    ntau = mfa.NFD(g, Q)
    al_list, fal_list = mfa.nspectrum(ntau, Q)
    q_list, dim_list = mfa.ndimension(ntau, Q, nodes) 
    width = max(al_list) - min(al_list)
    #print("size = ", len(dim_list))
    graphs[cnt].extend(dim_list)
    graphs[cnt].append(width)

    # find the curvature
    orc = OllivierRicci(g, alpha=0.5)
    orc.compute_ricci_curvature()
    #print(orc.G)
    cur_list = []
    for u, v, _ in g.edges(data=True):
    	cur_list.append(orc.G[u][v]["ricciCurvature"])
    print("ricci done")
    graphs[cnt].append(np.mean(cur_list))
    cur_list = []
    frc = FormanRicci(g)
    frc.compute_ricci_curvature()
    for u, v, _ in g.edges(data=True):
    	cur_list.append(frc.G[u][v]["formanCurvature"])
    graphs[cnt].append(np.mean(cur_list))

    graphs[cnt].append(vf_if[file_name.split('.')[0]][1])
    graphs[cnt].append(vf_if[file_name.split('.')[0]][2])
    cnt = cnt + 1
    


# save pca to csv file   
multifractal_str = ['dim_q=' + str(i) for i in range(-10,11,1) if i != 0]

header = ['filename']
header.extend(multifractal_str)
header.append("spectrumWidth")
header.append("ricciCurvature")
header.append("formanCurvature")
header.append("VF")
header.append("IF")
with open('graph_structure2.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)
    # write multiple rows
    writer.writerows(graphs)       

