import pickle, json
import numpy as np

f = open('lore_runtimes.pickle', 'rb')
runtimes = pickle.load(f)
f.close()
f = open('lore_runtimes_none_pragma.pickle', 'rb')
base_runtimes = pickle.load(f)
f.close()

vf_if = {}
times = {}
files_VF_IF = runtimes.keys()
VFs = [1, 2, 4, 8, 16]
IFs = [1, 2, 4, 8, 16]
baseline = {}
for file_VF_IF in files_VF_IF:
    tmp = file_VF_IF.rpartition('.')
    fn = tmp[0]
    tmp = tmp[2].split('-')
    VF_idx = VFs.index(int(tmp[0]))
    IF_idx = IFs.index(int(tmp[1]))
    fn_c = fn
    label = 5 * VF_idx + IF_idx
    rt_mean = np.mean(runtimes[file_VF_IF])
    base_mean = np.mean(base_runtimes[fn])
            
    if fn_c not in vf_if.keys():
        vf_if[fn_c] = (rt_mean, label)
    else:
        rt_mean_pre = vf_if[fn_c][0]
        if rt_mean < rt_mean_pre:
            vf_if[fn_c] = (rt_mean, label)

f = open('lore_embeddings2_graphsage_gcn128.json', 'rb')
exeT = json.load(f)
f.close()

label_dist = {}
cnt = 0
for f in vf_if:
    cnt += 1
    if vf_if[f][1] not in label_dist:
        label_dist[vf_if[f][1]] = 1
    else:
        label_dist[vf_if[f][1]] += 1

print("label distribution = ", label_dist.values())
print("cnt = ", cnt)
