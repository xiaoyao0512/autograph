
import json, glob, dgl, re, os
import programl as pg
from utility import find_loops
import pickle
import numpy as np


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



run_dir = "training_data_default"
save_dir = "training_data_vec"
vf_list = [1,2,4,8,16,32,64] # TODO: change this to match your hardware
if_list = [1,2,4,8,16] # TODO: change this to match your hardware



graphs = {}
max_nodes = -1
filenames = []
nodes = {}
for file in glob.glob(run_dir+"/*.c"):
    print(file)
    file_name = file.split('/')[1]
    if (file_name == "header.c"):
        continue
    num = find_loops(run_dir, file_name)
    print("num = ", num)
    # prepare graphs
    graphs[file_name] = {}
    vf_len = len(vf_list)
    if_len = len(if_list)
    for idx in range(num):
        graphs[file_name][idx] = [[None for _ in range(if_len)] for _ in range(vf_len)]
        for vf_idx1 in range(vf_len):
            for if_idx1 in range(if_len):
                VF1 = vf_list[vf_idx1]
                IF1 = if_list[if_idx1]
                insert_pragma(save_dir, run_dir, file_name, VF1, IF1, idx+1)
                full_path = save_dir + '/' + file_name
                #print("full_path = ", full_path)
                g = pg.from_clang([full_path, "-O3"], timeout=300000)
                g = pg.to_networkx(g)
                #print(g.nodes(data=True))
                curr_nodes = g.number_of_nodes()
                #print(g.number_of_nodes())
                if (curr_nodes > max_nodes):
                    max_nodes = curr_nodes
                fn = file_name + '-' + str(idx) + '-' + str(vf_idx1) + '-' + str(if_idx1)
                print("fn = ", fn)
                nodes[fn] = curr_nodes
                #if (curr_nodes > 10000):
                #    filenames.append(file_name)
                #    break
                #graphs[file_name][idx][vf_idx1][if_idx1] = g
                #print("g nodes = ", graphs[file_name][idx][vf_idx1][if_idx1].number_of_nodes())

print("max nodes = ", max_nodes)
with open('nodes.pickle', 'wb') as handle:
    pickle.dump(nodes, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''
print("filenames = ", list(set(filenames)))
filenames = list(set(filenames))
for file in filenames:
    full_path = run_dir + '/' + file
    print("delete ", full_path)
    os.system("rm " + full_path)

with open('graphs.pickle', 'wb') as handle:
    pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''
'''
import json, pickle, torch, dgl
from utility import load_observations
import programl as pg
import networkx as nx

BATCH_SIZE = 512
EMBED = 8


#Observations dictionary
# prepare graphs
graphs = {}
with open('/home/yao/Simulator/neuro-vectorizer/graphs.pickle', 'rb') as handle:
    graphs = pickle.load(handle)



obs_encodings = load_observations("training_data", "embeddings_"+str(EMBED)+".json", 7, 5)

current_fn = 's8n_64_mul_0.c'




g = graphs[current_fn][0][0][0]
G2 = pg.from_clang(["training_data/"+current_fn, "-O3"])

nxg2 = pg.to_networkx(G2)
print("g nodes = ", g.number_of_nodes())
print("nxg2 nodes = ", nxg2.number_of_nodes())
print(nx.is_isomorphic(g, nxg2))
G = dgl.from_networkx(g)
feat = torch.FloatTensor(obs_encodings[current_fn][0][0][0])
model = torch.load('gnn_model_'+str(EMBED)+'.pt')
model.eval()
print("#nodes = ", G.number_of_nodes())
print("filename = ", current_fn)
print("feat shape = ", feat.shape)
pred = model.inference(G, feat, BATCH_SIZE)
obs = pred.tolist()     
'''
