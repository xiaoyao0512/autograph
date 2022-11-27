import json, pickle
import numpy as np

benchmark = 'npb'

nv = open('lore_features_testing_neurovec_'+benchmark+'.json')
data_nv = json.load(nv)
nv.close()

ag = open('lore_features_testing_'+benchmark+'.json')
data_ag = json.load(ag)
ag.close()


f = open('lore_runtimes.pickle', 'rb')
runtimes = pickle.load(f)
f.close()
f = open('lore_runtimes_none_pragma.pickle', 'rb')
base_runtimes = pickle.load(f)
f.close()
vf_if = {}
times = {}
files_VF_IF = runtimes.keys()
vf_list = [1, 2, 4, 8, 16]
if_list = [1, 2, 4, 8, 16]
baseline = {}
for file_VF_IF in files_VF_IF:
    tmp = file_VF_IF.rpartition('.')
    fn = tmp[0]
    tmp = tmp[2].split('-')
    VF = int(tmp[0])
    IF = int(tmp[1])
    label = 5 * vf_list.index(VF) + if_list.index(IF)
    fn_c = fn
    rt_mean = np.median(runtimes[file_VF_IF])
    if fn_c not in vf_if.keys():
        vf_if[fn_c] = (rt_mean, label)
    else:
        rt_mean_pre = vf_if[fn_c][0]
        if rt_mean < rt_mean_pre:
            vf_if[fn_c] = (rt_mean, label)


def Intersection(lst1, lst2):
    return set(lst1).intersection(lst2)

nv_labels = data_nv["labels"]
ag_labels = data_ag["labels"]

final = {}
final["labels"] = []
final["files"] = []
final["feat"] = []
for f in Intersection(data_nv["files"], data_ag["files"]):
    fidx_ag = data_ag["files"].index(f)
    fidx_nv = data_nv["files"].index(f)
    #print(len(data_nv["feat"][fidx_ag][0]))
    final["files"].append(f)
    final["labels"].append(vf_if[f][1])
    temp = data_ag["feat"][fidx_ag][0]
    temp.extend(data_nv["feat"][fidx_nv][0])
    print(len(temp))
    final["feat"].append([temp])

with open('lore_features_testing_fuse_'+benchmark+'.json', 'w') as json_file:
  json.dump(final, json_file)
    


