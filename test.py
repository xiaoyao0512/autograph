'''
from utility import get_encodings_from_local

obs = get_encodings_from_local("training_data-small")

print("type0 = ", type(obs))
for i in obs:
    print(i)
    print("type1 = ", type(i))
    print("------------")
    print("dict = ", obs[i])
    print("type2 = ", type(obs[i]))
    for j in obs[i]:
        print("j = ", j)  
        print("type3 = ", type(j)) 
        print("obs[i][j] = ", obs[i][j])
        print("len = ", len(obs[i][j]))  
        print("type4 = ", type(obs[i][j]))  
    break


from utility import insert_pragma
import csv, os, glob
import subprocess
import numpy as np


run_dir = "training_data-small"
save_dir = "training_data_vec"
file_names = ["s1_64_add_0.c", "s1_64_sub_0.c", "s1_64_mul_0.c"]
write_names = ["add_time.csv", "sub_time.csv", "mul_time.csv"]

repeat = 5
MAX_RANGE = 32

def measure_exec(file_name, write_name):
    exec_time = []
    for VF in range(1, MAX_RANGE+1):
        exec_time.append([])
        for IF in range(1, MAX_RANGE+1):
            
            insert_pragma(save_dir, run_dir, file_name, VF, IF, 1)
            os.system("clang -O3 " + save_dir+'/'+file_name + ' ' + run_dir+'/header.c')

            time_slots = []
            for i in range(repeat):
                output = subprocess.check_output("./a.out", shell=True)
                time = float(output.split()[3])
                time_slots.append(time)

            average = sum(time_slots) / len(time_slots)
            #print("average = ", average)
            exec_time[VF-1].append(average)
    return np.array(exec_time)

#    with open(write_name,"w") as my_csv:
#        csvWriter = csv.writer(my_csv)
#        csvWriter.writerows(exec_time)

#for i in range(len(file_names)):
#    filename = file_names[i]
#    writename = write_names[i]
#    measure_exec(filename, writename)
exec_time = np.zeros((MAX_RANGE, MAX_RANGE))
for f in glob.glob(run_dir+"/*.c"):
#for f in file_names:
    print("f = ", f)
    filename = f.split('/')[1]
    #print("filename = ", filename)
    exec_time += measure_exec(filename, "")
print("final exec = ", exec_time)
with open("exec_time.csv","w") as my_csv:
    csvWriter = csv.writer(my_csv)
    csvWriter.writerows(exec_time)


import programl as pg
import glob 
import networkx as nx
import json
import pickle
import numpy as np

f = open('runtimes.pickle', 'rb')
runtimes = pickle.load(f)    
f.close()

vf_if = {}
times = {}
files_VF_IF = runtimes.keys()

baseline = {}
for file_VF_IF in files_VF_IF:
    tmp = file_VF_IF.split('.')
    fn = tmp[0]
    tmp = tmp[1].split('-')
    VF = int(tmp[0])
    IF = int(tmp[1])
    fn_c = fn + '.c'
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
# Construct a program graph from C++:

for f in glob.glob("json-small/*.json"):
    fn = f.split('/')[-1].split('.')[0]
    fn_c = fn + '.c'
    with open(f) as fh:
        g = nx.readwrite.json_graph.node_link_graph(json.load(fh))
    VF = vf_if[fn][1]
    IF = vf_if[fn][2]
    nx.write_gexf(g, "training_gexf/"+fn+"-"+str(VF)+"-"+str(IF)+".gexf")

'''


import json, pickle
import numpy as np
from sklearn import preprocessing

input_size = 64
pool = 'max'

#with open('embeddings_gcn2.json') as f:
#with open('embeddings_'+str(input_size)+'_'+pool+'_pooling.json') as f:
with open('embeddings2_graphsage_gcn128.json') as f:
#with open('embeddings2_graphsage_ggnn32.json') as f:
#with open('embeddings2_gcn_nn_fulltext128.json') as f:
    emb = json.load(f)



f = open('runtimes.pickle', 'rb')
runtimes = pickle.load(f)    
f.close()

f = open('runtimes_none_pragma.pickle', 'rb')
base_runtimes = pickle.load(f)    
f.close()


VF_list = [1, 2, 4, 8, 16]
IF_list = [1, 2, 4, 8, 16]

vf_if = {}
times = {}
files_VF_IF = runtimes.keys()
vf_list = []
if_list = []
baseline = {}
num_loops = {
    "s1": 1, "s2": 1, "s6": 1, "s6n": 1,
    "s7": 1, "s7n": 1, "s8": 1, "s8n": 1,
    "s9": 2, "s9n": 2, "s10": 1, "s11": 1,
    "s12": 1, "s12n": 1, "s12nn": 1, "s13": 1, 
    "s15": 2
}
loop_count = {}
for file_VF_IF in files_VF_IF:
    tmp = file_VF_IF.split('.')
    fn = tmp[0]
    fn_tmp = fn.split('_')
    fn_c = fn + '.c'
    if fn_c not in loop_count.keys():
        loop_count[fn_c] = [0, 0]
        for i in range(num_loops[fn_tmp[0]]):
            loop_count[fn_c][i] = int(fn_tmp[i+1])

    tmp = tmp[1].split('-')
    VF = int(tmp[0])
    IF = int(tmp[1])
    label = VF_list.index(VF) * 5 + IF_list.index(IF)
    rt_mean = np.median(runtimes[file_VF_IF])
    base_mean = np.median(base_runtimes[fn])
    #print("filename = ", fn)
    #print("VF = ", VF)
    #print("IF = ", IF)
    #print("mean = ", rt_mean)
    if fn not in vf_if.keys():
        vf_if[fn] = (rt_mean, label)
    else:
        rt_mean_pre = vf_if[fn][0]
        if rt_mean < rt_mean_pre:
            vf_if[fn] = (rt_mean, label)    
    if fn_c not in times.keys():
        times[fn_c] = {}
    if VF not in times[fn_c].keys():
        times[fn_c][VF] = {}
    if IF not in times[fn_c][VF].keys():
        times[fn_c][VF][IF] = rt_mean
    else:
        rt_mean_pre = times[fn][VF][IF]
        if (rt_mean < rt_mean_pre):
            times[fn_c][VF][IF] = rt_mean

    if fn_c not in baseline.keys():
        baseline[fn_c] = base_mean
    else:
        base_mean_pre = baseline[fn_c]
        if base_mean < base_mean_pre:
            baseline[fn_c] = base_mean

features = {}
files = []
feat = []
labels = []
idx_f = {}
count = 0
loop2d = []
for f in loop_count.keys():
    idx_f[f] = count
    loop2d.append(loop_count[f])
    count += 1

min_max_scaler = preprocessing.MinMaxScaler()
loop2d = min_max_scaler.fit_transform(loop2d)



for f in emb.keys():
    files.append(f)
    temp = emb[f][0]
    
    #temp.extend(loop2d[idx_f[f]])
    #print(loop2d[idx_f[f]])
    feat.append([temp])
    print(len(temp))
    labels.append(vf_if[f.split('.')[0]][1])

features['feat'] = feat
features['labels'] = labels
features['files'] = files
#with open('features_gcn_degree.json', 'w') as f:
with open('features2_graphsage_gcn128.json', 'w') as f:
#with open('features2_graphsage_ggnn32.json', 'w') as f:
#with open('features2_gcn_nn_fulltext128.json', 'w') as f:
    json.dump(features, f) 
