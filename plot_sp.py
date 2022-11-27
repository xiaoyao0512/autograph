import json
import pickle

import numpy as np
import matplotlib.pyplot as plt

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
    base_mean = np.median(base_runtimes[fn])
    if fn_c not in vf_if.keys():
        vf_if[fn_c] = (rt_mean, label)
    else:
        rt_mean_pre = vf_if[fn_c][0]
        if rt_mean < rt_mean_pre:
            vf_if[fn_c] = (rt_mean, label)
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




benchmark = 'spec'

def read_json(fn, vf_if, times, baseline):
    fp = open(fn)
    data = json.load(fp)
    fp.close()

    acc_cls = {}
    num_cls = {}
    for i in range(25):
        acc_cls[i] = 0
        num_cls[i] = 0;

    sp = []
    gt = []
    for f in data:
        true = vf_if[f][1]
        pred = data[f]
        num_cls[true] += 1
        sp.append(baseline[f]/times[f][pred])
        gt.append(baseline[f]/times[f][true])
        if (true == pred):
            acc_cls[pred] += 1

    for i in range(25):
        if num_cls[i] == 0:
            acc_cls[i] = 0
        else:
            acc_cls[i] = (acc_cls[i] / num_cls[i]) * 100
    acc = []
    for i in sorted(acc_cls.keys()):
        acc.append(acc_cls[i])
    return acc, sp, gt


acc1, sp1, gt = read_json('lore_testing1_autograph_800k_'+benchmark+'.json', vf_if, times, baseline)
acc2, sp2, _ = read_json('lore_testing2_autograph_800k_'+benchmark+'.json', vf_if, times, baseline)
acc3, sp3, _ = read_json('lore_testing3_autograph_800k_'+benchmark+'.json', vf_if, times, baseline)
acc4, sp4, _ = read_json('lore_testing4_autograph_800k_'+benchmark+'.json', vf_if, times, baseline)
acc5, sp5, _ = read_json('lore_testing5_autograph_800k_'+benchmark+'.json', vf_if, times, baseline)
acc6, sp6, _ = read_json('lore_testing6_autograph_800k_'+benchmark+'.json', vf_if, times, baseline)

N = 25
ind = np.arange(N)
width = 0.1

plt.rcParams["figure.figsize"] = [16, 8]
plt.rcParams.update({'font.size': 20})
bar1 = plt.bar(ind, acc1, width)
bar2 = plt.bar(ind+width, acc2, width)
bar3 = plt.bar(ind+2*width, acc3, width)
bar4 = plt.bar(ind+3*width, acc4, width)
bar5 = plt.bar(ind+4*width, acc5, width)
bar6 = plt.bar(ind+5*width, acc6, width)

plt.xlabel("Class label")
plt.ylabel("Accuracy (%)")

plt.xticks(ind+2*width, list(range(25)))
plt.legend((bar1, bar2, bar3, bar4, bar5, bar6), ('b=2', 'b=3', 'b=4', 'b=5', 'b=6', 'b=7'))
plt.savefig(benchmark+'_acc_per_cls_b.png')

plt.figure()
plt.rcParams["figure.figsize"] = [16, 8]
plt.rcParams.update({'font.size': 20})
x = list(range(len(sp1)))
plt.plot(x, gt, label="true labels")
plt.plot(x, sp1, label="b=2")
plt.plot(x, sp2, label="b=3")
plt.plot(x, sp3, label="b=4")
plt.plot(x, sp4, label="b=5")
plt.plot(x, sp5, label="b=6")
plt.plot(x, sp6, label="b=7")
plt.xlabel("Kernel index")
plt.ylabel("Speedup")
plt.legend()

plt.savefig(benchmark+'_sp_b.png')

cnt = 0
for i in range(len(gt)): 
    if (gt[i] < sp1[i]):
        cnt = cnt + 1
print(cnt)
