import glob, os, subprocess
from utility import insert_pragma
import json, copy
import programl as pg
import networkx as nx
import csv
import numpy as np
from torch import nn
import torch
import csv
from sklearn.decomposition import PCA

run_dir = "training_data_default"
save_dir = "training_data_vec"
adjlist_dir = "training_adjlist"
gexf_dir = "training_gexf"
repeat = 10
exec_time = {}
# feature mode is either 0 or 1.
# 0: Find graphs, execution times
# 1: find features, require exec_time.json (generated when set to 0) is present 
# 1 is helpful in evaluating different embeddings in GNN.
structure_mode = 0
semantics_mode = 1
json_file = os.path.isfile("exec_time.json")


def read_feature_from_files(embed_path):
    assert os.path.isfile(embed_path)
    feat = {}
    cnt = 0
    feat_matrix = []
    with open(embed_path) as file_in:
        for line in file_in:
            #print(line)
            cnt += 1
            if cnt == 1:
                continue 
            line = line.rstrip()
            embed = line.split(' ')
            #print("embed = ", embed)
            node = embed.pop(0)
            feat[node] = list(map(float, embed))
            
    #print("feat = ", feat)
    for k in sorted(feat.keys()):
        #print("k = ", k, " d = ", feat[k])
        feat_matrix.append(feat[k])
    return feat_matrix


def deepwalk(exec_time):
    assert exec_time != {}
    extended_exec = copy.deepcopy(exec_time)
    for filename in exec_time.keys():
        full_path = adjlist_dir+'/'+filename+'.adjlist'
        embed_path = adjlist_dir+'/'+filename+'.embeddings'
        os.system("deepwalk --input "+full_path+" --output "+embed_path+" --representation-size 8")
        feat = read_feature_from_files(embed_path)
        #print("feat = ", feat)
        #print("before, ", extended_exec[filename])
        extended_exec[filename].append(feat)
        #print("after, ", extended_exec[filename])
    return extended_exec

def feature_extract(mode, exec_time):
    if (mode == "deepwalk"):
        return deepwalk(exec_time)


if (structure_mode and json_file):
    assert semantics_mode == 0
    print("structure_mode = 1")
    with open('exec_time.json') as f:
        exec_time = json.load(f)
    exec_time = feature_extract("deepwalk", exec_time)
    with open('exec_time.json', 'w') as f:
        json.dump(exec_time, f)
    # skip the rest of the code
    print("Quit and skip the rest of the code")
    quit()

def dict_from_csv(filename):
    bag = {}
    with open(filename, 'r') as file:
        csvreader = csv.reader(file, delimiter="\t")
        header = next(csvreader)
        #print("header = ", header)
        for elem in header:
            bag[elem] = []
        for row in csvreader:
            for i in range(len(row)):
                #print("i = ", i)
                #print("row = ", row)
                val = row[i]
                key = header[i]
                bag[key].append(val)
    return bag

def z(exec_times):
    exec_mean = np.mean(exec_times)
    exec_std = np.std(exec_times)
    exec_normalized = []
    for exec_time in exec_times:
        exec_normalized.append((exec_time - exec_mean) / exec_std)
    return exec_normalized

def normalize(matrix):
    matrix_tp = np.array(matrix).T
    rows = matrix_tp.shape[0]
    cols = matrix_tp.shape[1]
    rt_matrix = []
    for row in matrix_tp:
        row_norm = z(row)
        rt_matrix.append(list(row_norm))
    return rt_matrix

def vocal(nodes):
    #print(nodes)
    feat_matrix = []
    bag_of_words = dict_from_csv("vocab/programl.csv")
    dim = 6 #int(pow(2230, 0.25))
    emb = nn.Embedding(2230, dim)
    for (node_id, node_dict) in nodes:
        #print(node_id)
        #print(node_dict)
        # check if a keyword is in the vocabulary
        val = 0
        if (node_dict['text'] in bag_of_words['text']):
            text_idx = bag_of_words['text'].index(node_dict['text'])
            val = text_idx + 1
            '''            
            if (text_idx != 0):
                val = float(bag_of_words['cumulative_node_frequency'][text_idx]) - float(bag_of_words['cumulative_node_frequency'][text_idx-1])
            else:
                val = float(bag_of_words['cumulative_node_frequency'][text_idx])
            '''
        #print(node_dict)
        #print("dict type = ", node_dict['type'])
        #idx_emb = [val]
        #type_emb = [node_dict['type']]
        idx_emb = emb(torch.IntTensor([val])).flatten().tolist()
        type_emb = emb(torch.IntTensor([node_dict['type']])).flatten().tolist()
        
        feat = []
        feat.extend(idx_emb)
        feat.extend(type_emb)
        #print(feat)
        #quit()
        feat_matrix.append(feat)
    #print("feat_matrix = ", feat_matrix)
    return feat_matrix#normalize(feat_matrix)


def pca_analysis(feature, n_comp):
    #feat = torch.FloatTensor(feature)
    pca = PCA(n_components=n_comp, random_state=2022)
    pca.fit(feature)
    #return pca.explained_variance_ratio_ * 100
    pca_8 = pca.transform(feature)
    return pca_8.tolist()
    #print("pca ratio = ", pca.explained_variance_ratio_ * 100)
    

def semantic_extract(exec_time):
    extended_exec = copy.deepcopy(exec_time)
    #pca = []
    for file in glob.glob(run_dir+"/*.c"):
        filename = file.split('/')[1]
        if (filename == "header.c"):
            continue     
        g = pg.from_clang([file, "-O3"])
        nxg = pg.to_networkx(g)      
        feat = vocal(nxg.nodes(data=True))
        
        feat = pca_analysis(feat, 8)
        #pca.append(pca_r)
        #print("feat = ", feat)
        #quit()
        extended_exec[filename].append(feat) 
    # save pca to csv file   
    #header = ['pca_'+str(i+1) for i in range(12)] 
    #with open('pca_ratio.csv', 'w', encoding='UTF8', newline='') as f:
        #writer = csv.writer(f)
        # write the header
        #writer.writerow(header)
        # write multiple rows
        #writer.writerows(pca)       
    return extended_exec

if (semantics_mode):
    assert structure_mode == 0
    print("semantics_mode = 1")
    with open('exec_time.json') as f:
        exec_time = json.load(f)
    for filename in exec_time.keys():
        exec_time[filename].pop()
    exec_time = semantic_extract(exec_time)
    #quit()
    print("saving exec_time")
    with open('exec_time.json', 'w') as f:
        json.dump(exec_time, f)
    # skip the rest of the code
    print("Quit and skip the rest of the code")
    quit()


for file in glob.glob(run_dir+"/*.c"):
    print(file)
    file_name = file.split('/')[1]
    if (file_name == "header.c"):
        continue
    # prepare graphs
    g = pg.from_clang([file, "-O3"])
    nxg = pg.to_networkx(g)
    #adjlist = adjlist_dir+'/'+file_name+'.adjlist'
    #nx.write_adjlist(nxg, adjlist)
    #nx.write_gexf(nxg, gexf_dir+'/'+file_name+'.gexf')
    g_dict = nx.to_dict_of_dicts(nxg)
    #g = dgl.from_networkx(nxg)
    # prepare timing
    
    insert_pragma(save_dir, run_dir, file_name, 1, 1, 1)
    os.system("clang -O3 " + save_dir+'/'+file_name + ' ' + run_dir+'/header.c')

    time_slots = []
    for i in range(repeat):
        output = subprocess.check_output("./a.out", shell=True)
        time = float(output.split()[3])
        time_slots.append(time)

    average = sum(time_slots) / len(time_slots)
    print("average = ", average)
    exec_time[file_name] = [time_slots, average, g_dict]
    # potential to convert back to graph
    # from_dict_of_dicts; G = nx.Graph(d)
    # https://networkx.org/documentation/stable/reference/convert.html
    
with open('exec_time.json', 'w') as f:
    json.dump(exec_time, f) 
#with open('1.json') as f:
#    d = json.load(f)
