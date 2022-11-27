#!/usr/bin/env python
# coding: utf-8

# In[8]:


## predictions on LORE icx testing, confusion matrix

import json
import pickle
from statistics import mean
from sklearn.metrics import confusion_matrix
import numpy as np
import random
import itertools
import json
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gmean
from mlxtend.plotting import plot_confusion_matrix
import time


def two2one(VF, IF):
    VF_list = [1,2,4,8,16] # TODO: change this to match your hardware
    IF_list = [1,2,4,8,16] # TODO: change this to match your hardware
    vf_idx = VF_list.index(VF)
    if_idx = IF_list.index(IF)
    return 5 * vf_idx + if_idx
    

def one2two(label):
    VF_list = [1,2,4,8,16] # TODO: change this to match your hardware
    IF_list = [1,2,4,8,16] # TODO: change this to match your hardware    
    y1 = VF_list[int(int(label) / 5)]
    y2 = IF_list[int(int(label) % 5)]
    return (y1, y2)    
        
file_runtimes = "lore_runtimes.pickle"
file2 = open(file_runtimes, 'rb')
runtimes = pickle.load(file2)



file_runtimes_none = "lore_runtimes_none_pragma.pickle"
file3 = open(file_runtimes_none, 'rb')
base_runtimes = pickle.load(file3)



'''
bestbf_hints = {}
bestbf_speedups = {}
runtimes_base = {}
for kernel in runtimes.keys():
    runtimes_base[kernel] = mean(runtimes_base[kernel])
    kernel_runtimes = runtimes[kernel]
    for k, v in kernel_runtimes.items():
        kernel_runtimes[k] = np.mean(v)

    bestbf_hints[kernel] = min(kernel_runtimes,key=kernel_runtimes.get)
    bestbf_speedups[kernel] = runtimes_base[kernel]/min(kernel_runtimes.values())
'''
  
  
vf_if = {}
times = {}
files_VF_IF = runtimes.keys()
vf_list = []
if_list = []
baseline = {}

VF_list = [1,2,4,8,16] # TODO: change this to match your hardware
IF_list = [1,2,4,8,16] # TODO: change this to match your hardware
for file_VF_IF in files_VF_IF:
    tmp = file_VF_IF.rpartition('.')
    fn = tmp[0]
    tmp = tmp[2].split('-')
    VF = int(tmp[0])
    IF = int(tmp[1])
    fn_c = fn
    rt_mean = np.mean(runtimes[file_VF_IF])
    base_mean = np.mean(base_runtimes[fn])
    #print("filename = ", fn)
    #print("VF = ", VF)
    #print("IF = ", IF)
    #print("mean = ", rt_mean)
    if fn_c not in vf_if.keys():
        vf_if[fn_c] = (rt_mean, VF, IF)
    else:
        rt_mean_pre = vf_if[fn_c][0]
        if rt_mean < rt_mean_pre:
            vf_if[fn_c] = (rt_mean, VF, IF)    
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


prediction_list=[]
label_list = []
accurate = [0 for i in range(25)]
prediction_count = [0 for i in range(25)]
speedups = []
bf_speedups = []

'''
file = "LORE_ICX/lore_testing_subset_spec2006_icx7_omp.json"
f = open(file)
Spec = json.load(f)
print(len(Spec))
'''

benchmark = 'spec'
method = 'autograph'
fh = ''
if method == 'neurovec':
    fh = 'lore_features_testing_'+method+'_'+benchmark+'.json'
elif method == 'autograph':
    fh = 'lore_features_testing''.json'
with open(fh) as f:
    features = json.load(f)

#with open('lore_features2_graphsage_gcn128_testing.json') as f:
#    features = json.load(f)
Spec = features["files"]

filename = 'lore_testing_'+method+'_800k_'+benchmark+'.json'

# lore_neurovec_predictions_800k_Aug25.json
#print(filename)
f1 = open(filename)
predictions = json.load(f1)
f1.close()
count = [0 for i in range(25)]
#print(count)
for k, v in predictions.items():
    if k in Spec:
        y1, y2 = one2two(v)
        rt_mean, VF, IF = vf_if[k]
        # pruning
        if (baseline[k] / rt_mean <= 1.02):
            continue
        speedups.append(baseline[k]/times[k][y1][y2])
        bf_speedups.append(baseline[k] / rt_mean)
        # convert to 0-24
        best_label = two2one(VF, IF)
        prediction_list.append(v)
        label_list.append(best_label)
        prediction_count[v] += 1 
        count[best_label] += 1
        if best_label == v:
            accurate[v] += 1
print(accurate)   
print(label_list)
#hints_list=['no pragma',1, 2, 4, 8, 16, 32, 64]
hints_list = ['(' + str(i) + ', ' + str(j) + ')' for i in range(5) for j in range(5)]
# print(label_list)
# print(prediction_list)
print(len(speedups))
print("The geometric mean of the brute-force speedup is:")
print(gmean(bf_speedups))
print("The geometric mean of the predicted speedup is:")
print(gmean(speedups))
print("The prediction accuracy is:")
print(sum(accurate)/len(label_list))
# print("The count per class are:")
# print(count)
cmatrix = confusion_matrix(label_list,prediction_list)
#df=pd.DataFrame(cmatrix) 
#df.index = hints_list
# df.to_excel("ConfusionMatrix_loregraphsage.xls",header = hints_list,index = True)
# print(cmatrix)

plt.figure()
fig, ax = plot_confusion_matrix(conf_mat=cmatrix, figsize=(6, 6), cmap=plt.cm.Greens)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
# plt.title('Confusion Matrix', fontsize=18)
plt.savefig('cmatrix.png')

acc = [accurate[v]/count[v] for v in range(25)]

# print(truecount)
# print(prediction_list)
# print(label_list)
x = range(25)

plt.figure()
plt.bar(x, acc)
plt.xlabel('The classes')
# plt.ylabel('Number of Samples per Class')
plt.ylabel('prediction accuracy per class')
plt.ylim(0, 0.8)
plt.xticks(x,hints_list,rotation=0)

plt.savefig("acc.png")


