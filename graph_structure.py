import pickle
import glob, csv
import programl as pg
import re
from utility import find_loops
import pickle, json
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import large_clique_size


def insert_pragma(save_dir, run_dir, fname, VF1, IF1, idx):
    file = fname.split('.')[0]
    new_file = save_dir + '/' + file + ".c"
    fw = open(new_file, 'w')
    num = 0
    with open(run_dir+'/'+fname) as fr:
        for line in fr:
            #print(line.rstrip())
            if re.match(r'^\s*for\s*\(',line) or re.match(r'^\s*while\s*\(',line):
                num = num + 1
                if (num == idx):
                    fw.write("#pragma clang loop vectorize_width("+str(VF1)+") interleave_count("+str(IF1)+")\n") 
                else:
                    fw.write("#pragma clang loop vectorize_width("+str(2)+") interleave_count("+str(2)+")\n")                 
            fw.write(line)
    fw.close()   

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
    
    # features
    #print("starting ...")
    graphs[cnt].append(file_name)
    graphs[cnt].append(nx.number_attracting_components(g))
    graphs[cnt].append(nx.number_strongly_connected_components(g))
    graphs[cnt].append(nx.number_weakly_connected_components(g))
    g = nx.Graph(g)
    #print("#nodes")
    graphs[cnt].append(g.number_of_nodes())
    graphs[cnt].append(g.number_of_edges())
    graphs[cnt].append(nx.node_connectivity(g))
    graphs[cnt].append(large_clique_size(g))
    graphs[cnt].append(nx.average_clustering(g))
    #print("diameter")
    graphs[cnt].append(nx.diameter(g))
    graphs[cnt].append(nx.estrada_index(g))
    graphs[cnt].append(nx.degree_pearson_correlation_coefficient(g))
    graphs[cnt].append(nx.transitivity(g))
    graphs[cnt].append(nx.number_connected_components(g))
    #print("local_efficiency")
    graphs[cnt].append(nx.local_efficiency(g))
    graphs[cnt].append(nx.global_efficiency(g))
    graphs[cnt].append(nx.number_of_isolates(g))
    #graphs[cnt].append(nx.shortest_path_length(g))
    #graphs[cnt].append(nx.average_shortest_path_length(g))
    #print("sigma")
    #graphs[cnt].append(nx.sigma(g))
    #print("omega")
    #graphs[cnt].append(nx.omega(g))
    #print("s_metric")
    #graphs[cnt].append(nx.s_metric(g))
    graphs[cnt].append(nx.wiener_index(g))
    graphs[cnt].append(nx.density(g))
    graphs[cnt].append(vf_if[file_name.split('.')[0]][1])
    graphs[cnt].append(vf_if[file_name.split('.')[0]][2])
    cnt = cnt + 1
    


# save pca to csv file   
header = ['filename', '#attracting_comp', '#SCC', '#WCC', '#nodes', '#edges', 'node_connectivity', 'large_clique_size', 'average_clustering', 'diameter', 'estrada_index', 'assortativity', 'transitivity', '#CC',  'local_efficiency', 'global_efficiency', '#isolates', 'wiener_index', 'density', 'VF', 'IF'] 
with open('graph_structure.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)
    # write multiple rows
    writer.writerows(graphs)       

