
import json, pickle
import numpy as np
from sklearn import preprocessing
import os

input_size = 64
pool = 'max'

filename = 'lore_embeddings2_graphsage_gcn128.json'
filename_n = 'lore_features2_graphsage_gcn128.json'
#with open('embeddings_gcn2.json') as f:
#with open('embeddings_'+str(input_size)+'_'+pool+'_pooling.json') as f:

#print(0)
with open(filename) as f:
#with open('embeddings2_graphsage_ggnn32.json') as f:
#with open('embeddings2_gcn_nn_fulltext128.json') as f:
    emb = json.load(f)

#print(1)

f = open('lore_runtimes2.pickle', 'rb')
runtimes = pickle.load(f)    
f.close()

#print(2)

f = open('lore_runtimes_none_pragma2.pickle', 'rb')
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

for file_VF_IF in files_VF_IF:
    #print(file_VF_IF)
    tmp = file_VF_IF.rpartition('.')
    fn = tmp[0]
    fn_c = fn
    tmp = tmp[2].split('-')
    #print(fn, tmp)
    #exit()
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

features = {}
files = []
feat = []
labels = []

cnt = 0

for f in vf_if.keys():
    if f not in emb:
        #print(f)
        os.system("rm -rf lore-src/"+f)
        continue
    print(f)
    files.append(f)
    temp = emb[f]
    print(len(temp))
    assert len(temp[0]) == 256, "ERROR"
    #temp[0].extend([0, 0])
    feat.append(temp)
    #print(len(temp))
    #print(f)
    labels.append(vf_if[f][1])
    cnt += 1

print("num of files = ", cnt)

features['feat'] = feat
features['labels'] = labels
features['files'] = files
#with open('features_gcn_degree.json', 'w') as f:
with open(filename_n, 'w') as f:
#with open('features2_graphsage_ggnn32.json', 'w') as f:
#with open('features2_gcn_nn_fulltext128.json', 'w') as f:
    json.dump(features, f) 
