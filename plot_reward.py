import json
import matplotlib.pyplot as plt
from scipy.stats import gmean
import numpy as np

with open('lore_autograph_reward.json') as f:
    autograph_reward = json.load(f)

with open('lore_neurovec_reward.json') as f:
    neurovec_reward = json.load(f)

autograph_rwd_sort = dict(sorted(autograph_reward.items(), key=lambda item: item[1]))
#print(autograph_rwd_sort.values())


sorted_files = list(autograph_rwd_sort.keys())
sorted_rwd_autograph = list(autograph_rwd_sort.values())
sorted_rwd_neurovec = []
for f in sorted_files:
    sorted_rwd_neurovec.append(neurovec_reward[f])

print(len(sorted_files))
print("autograph geometric mean = ", gmean(sorted_rwd_autograph))
print("neurovec geometric mean = ", gmean(sorted_rwd_neurovec))
print("autograph mean = ", np.mean(sorted_rwd_autograph))
print("neurovec mean = ", np.mean(sorted_rwd_neurovec))
plt.plot(sorted_rwd_autograph, color='red', label='autograph (geometric mean = '+"{:.2f}".format(gmean(sorted_rwd_autograph))+')')
plt.plot(sorted_rwd_neurovec, color='blue', label='neuro-vectorizer (geometric mean = '+"{:.2f}".format(gmean(sorted_rwd_neurovec))+')', alpha=0.5)
plt.legend(fontsize=12, loc='upper left', frameon=False)
plt.xlabel("Kernel ID", fontsize=14)
plt.ylabel("Reward", fontsize=14)
#plt.xticks([])

plt.savefig("reward.png")
