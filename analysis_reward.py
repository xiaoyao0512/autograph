import json, pickle
import matplotlib.pyplot as plt
from scipy.stats import gmean
import numpy as np

with open('lore_full_prediction.json') as f:
    ag_pred = json.load(f)

with open('lore_full_neurovec.json') as f:
    nv_pred = json.load(f)

with open('lore_gcn_pred.json') as f:
    gcn_pred = json.load(f)

with open('lore_code2vec_pred.json') as f:
    c2v_pred = json.load(f)


f = open('lore_runtimes2.pickle', 'rb')
runtimes = pickle.load(f)
f.close()
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
    tmp = tmp[2].split('-')
    VF = int(tmp[0])
    IF = int(tmp[1])
    label = VF_list.index(VF) * 5 + IF_list.index(IF)
    fn_c = fn
    rt_mean = np.mean(runtimes[file_VF_IF])
    base_mean = np.mean(base_runtimes[fn])
    if fn not in vf_if.keys():
        vf_if[fn] = (rt_mean, label)
    else:
        rt_mean_pre = vf_if[fn][0]
        if rt_mean < rt_mean_pre:
            vf_if[fn] = (rt_mean, label)
    if fn_c not in times.keys():
        times[fn_c] = {}
    if label not in times[fn_c].keys():
        times[fn_c][label] = rt_mean
    else:
        rt_mean_pre = times[fn][label]
        if (rt_mean < rt_mean_pre):
            times[fn_c][label] = rt_mean

    if fn_c not in baseline.keys():
        baseline[fn_c] = base_mean
    else:
        base_mean_pre = baseline[fn_c]
        if base_mean < base_mean_pre:
            baseline[fn_c] = base_mean


files = list(ag_pred.keys())
highest_rwd1 = []
highest_rwd2 = []
ag_rwd = []
nv_rwd = []
gcn_rwd = []
c2v_rwd = []
benchmark = "poly"
bf_rwd = []
for i in range(len(files)):
    f = files[i]
    bench = f.split('/')[0]
    if benchmark not in bench:
        continue
    print(f)
    bf_rwd.append(baseline[f]/times[f][vf_if[f][1]])
    ag_rwd.append(baseline[f]/times[f][ag_pred[f]])
    if f in nv_pred:
        nv_rwd.append(baseline[f]/times[f][nv_pred[f]])
    else:
        nv_rwd.append(-1)
    if f in gcn_pred:
        gcn_rwd.append(baseline[f]/times[f][gcn_pred[f]])
    else:
        gcn_rwd.append(-1)

    if f in c2v_pred:
        c2v_rwd.append(baseline[f]/times[f][c2v_pred[f]])
    else:
        c2v_rwd.append(-1)

acc_ag = 0
acc_nv = 0
acc_gcn = 0
acc_c2v = 0
acc_max1 = 0
acc_max2 = 0
cnt = 0
for i in range(len(files)):
    fn = files[i]
    bench = fn.split('/')[0]
    if benchmark not in bench:
        cnt += 1
        continue
    rwd_l1 = [ag_rwd[i-cnt], nv_rwd[i-cnt], gcn_rwd[i-cnt], c2v_rwd[i-cnt]]
    rwd_l2 = [ag_rwd[i-cnt], nv_rwd[i-cnt]]
    pred_l = [ag_pred[fn], nv_pred[fn], gcn_pred[fn], c2v_pred[fn]]
    max_rwd1 = max(rwd_l1)
    max_rwd1_idx = rwd_l1.index(max_rwd1)
    max_rwd2 = max(rwd_l2)
    max_rwd2_idx = rwd_l2.index(max_rwd2)
    
    highest_rwd1.append(max_rwd1)
    highest_rwd2.append(max_rwd2)

    gt = vf_if[fn][1]
    if (gt == ag_pred[fn]):
        acc_ag += 1
    if (gt == nv_pred[fn]):
        acc_nv += 1
    if (gt == gcn_pred[fn]):
        acc_gcn += 1
    if (gt == c2v_pred[fn]):
        acc_c2v += 1
    if (gt == pred_l[max_rwd1_idx]):
        acc_max1 += 1
    if (gt == pred_l[max_rwd2_idx]):
        acc_max2 += 1

acc_ag /= len(c2v_rwd)
acc_nv /= len(c2v_rwd)
acc_gcn /= len(c2v_rwd)
acc_c2v /= len(c2v_rwd)
acc_max1 /= len(c2v_rwd)
acc_max2 /= len(c2v_rwd)


print("autograph geometric mean = ", gmean(ag_rwd))
print("neurovec geometric mean = ", gmean(nv_rwd))
print("gcn geometric mean = ", gmean(gcn_rwd))
print("code2vec geometric mean = ", gmean(c2v_rwd))
print("max1 geometric mean = ", gmean(highest_rwd1))
print("max2 geometric mean = ", gmean(highest_rwd2))
print("bruteforce geometric mean = ", gmean(bf_rwd))
print("autograph accuracy = ", acc_ag)
print("neurovec accuracy = ", acc_nv)
print("gcn accuracy = ", acc_gcn)
print("code2vec accuracy = ", acc_c2v)
print("max1 accuracy = ", acc_max1)
print("max2 accuracy = ", acc_max2)

