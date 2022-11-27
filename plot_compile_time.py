import pickle
import numpy as np
import json
from numpy import inf
import matplotlib.pyplot as plt


with open('runtimes_full.pickle', 'rb') as f:
    execT = pickle.load(f)

with open('binaries_compile_times.pickle', 'rb') as f:
    compileT = pickle.load(f)

with open('lore_full_autograph.json', 'rb') as f:
    autograph_pred = json.load(f)

autograph_predictionT = {} #2.13*e9/3928
autograph_compileT = {}
brute_force_compileT = {}

pred_time = 2.13 * (10**9)/ 3928
cnt = 0
for f in compileT:
    sum_time = 0
    if f not in execT:
        continue
    cnt += 1
    for label in range(25):
        label_execT = np.mean(execT[f][str(label)])
        label_compileT = np.mean(compileT[f][str(label)])
        sum_time += label_execT + label_compileT
    brute_force_compileT[f] = sum_time
    autograph_predictionT[f] = pred_time
    autograph_compileT[f] = pred_time + np.mean(compileT[f][str(autograph_pred[f])])

print(cnt)
print(pred_time)

brute_force_sort = dict(sorted(brute_force_compileT.items(), key=lambda item: item[1]))
#print(autograph_rwd_sort.values())


sorted_files = list(brute_force_sort.keys())
sorted_time_bruteforce = list(brute_force_sort.values())
sorted_predtime = []
sorted_comptime = []
for f in sorted_files:
    sorted_comptime.append(autograph_compileT[f])
    sorted_predtime.append(autograph_predictionT[f])

#print("autograph geometric mean = ", gmean(sorted_rwd_autograph))
#print("neurovec geometric mean = ", gmean(sorted_rwd_neurovec))
#print("autograph mean = ", np.mean(sorted_rwd_autograph))
#print("neurovec mean = ", np.mean(sorted_rwd_neurovec))
#plt.yscale("log")  
plt.plot(np.log10(np.array(sorted_time_bruteforce)), color='red', label='Brute-force Search Time')
plt.plot(np.log10(np.array(sorted_comptime)), color='blue', label='Autograph Compile + Prediction Time')
plt.plot(np.log10(np.array(sorted_predtime)), color='green', label='Autograph Prediction Time')
plt.legend(fontsize=12, loc='upper left', frameon=False)
plt.xlabel("Kernel ID", fontsize=14)
plt.ylabel("log(Time)", fontsize=14)

#plt.xticks([])
plt.ylim(5, 12.5)
plt.savefig("time.png")

