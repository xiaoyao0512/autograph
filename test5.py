import pickle
import numpy as np
import glob
import os
import json

def merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def f1(data_temp):
    data = {}
    for f in data_temp:
        #print(f)
        #f_n = f.replace(".", "_").replace("-", "_")
        #print(f_n)
        for vf_if in data_temp[f]:
            f_new = f + "." + str(vf_if[0]) + "-" + str(vf_if[1])
            data[f_new] = data_temp[f][vf_if]
    return data

def f2(data_temp):
    data = {}
    for f in data_temp:
        #f_n = f.replace(".", "_").replace("-", "_")
        data[f] = data_temp[f]
    return data

filename1 = "runtimes_brute_spec.pickle"
filename2 = "runtimes_lore_brute_reliable_training.pickle"
filename3 = "runtimes_lore_orig_reliable_training.pickle"
filename4 = "runtimes_orig_spec.pickle"
with open(filename1, 'rb') as f:
    data1 = pickle.load(f)
with open(filename2, 'rb') as f:
    data2 = pickle.load(f)
with open(filename3, 'rb') as f:
    data3 = pickle.load(f)
with open(filename4, 'rb') as f:
    data4 = pickle.load(f)


data_temp1 = merge(data1, data2)
data_temp1 = f1(data_temp1)

with open('lore_runtimes.pickle', 'wb') as f:
    pickle.dump(data_temp1, f)


data_temp2 = merge(data3, data4)
data_temp2 = f2(data_temp2)

with open('lore_runtimes_none_pragma.pickle', 'wb') as f:
    pickle.dump(data_temp2, f)

f_new = "spec2006_v1_1/400_perlbench/numeric_c_S_mulexp10_line803"
f_sub = "spec2006_v1_1/400_perlbench/numeric_c_S"
vf_if = {}
for key in list(data_temp1.keys()):
    tmp = key.split('.')
    fn = tmp[0]
    vf_if[fn] = 1

for key in list(vf_if.keys()):
    if key.find(f_sub) != -1:
        print(key)

cnt = 0
cnt2 = 0
f_sub = 'ALPBench_v1_0/MPGdec/getpic_c_Saturate_line870'
f_sub1 = 'ALPBench_v1.0'

#with open("loregcc_embeddings2_128.json") as f:
#    emb = json.load(f)


#for filename in glob.glob("json_lore/**/*.json", recursive=True):
for f in list(data_temp2.keys()):
    #f = filename.split('/', 2)[-1][:-5]
    #print(f)
    if f.find(f_sub1) != -1:
        print(f)

    #print(af)
    #if f in vf_if.keys():
    #    print(1111)
    #if f == f_new:
    #    print(2222)
    #print(f_sub)


#for f in glob.glob("json_lore/**/*.json", recursive = True):
#    print(f)

#print(data1['spec2006_v1.1/482.sphinx3/new_fe_sp.c_fe_create_2d_line511'][(1, 1)])
#print(data4) key: filename; value: [XXXXXX]
