import pickle
import numpy as np
import glob

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


data_temp = merge(data1, data2)
data_temp = f1(data_temp)

with open('lore_runtimes2.pickle', 'wb') as f:
    pickle.dump(data_temp, f)


data_temp = merge(data3, data4)
data_temp = f2(data_temp)

with open('lore_runtimes_none_pragma2.pickle', 'wb') as f:
    pickle.dump(data_temp, f)


#for f in glob.glob("json_lore/**/*.json", recursive = True):
#    print(f)

#print(data1['spec2006_v1.1/482.sphinx3/new_fe_sp.c_fe_create_2d_line511'][(1, 1)])
#print(data4) key: filename; value: [XXXXXX]
