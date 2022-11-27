import json
from collections import OrderedDict
from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr

with open('lore_full_autograph.json', 'rb') as f:
    autograph = json.load(f)

with open('lore_full_neurovec.json', 'rb') as f:
    neurovec = json.load(f)

with open('lore_brute_force.json', 'rb') as f:
    bruteforce = json.load(f)

files = list(autograph.keys())
num = len(files)
auto = {}
neur = {}
bf = {}

for i in range(25):
    auto[i] = 0.000001
    neur[i] = 0.000001
    bf[i] = 0.000001

for f in files:
    auto_label = autograph[f]
    neur_label = neurovec[f]
    bf_label = bruteforce[f]
    if (auto_label not in auto):
        auto[auto_label] = 1
    else:
        auto[auto_label] += 1

    if (neur_label not in neur):
        neur[neur_label] = 1
    else:
        neur[neur_label] += 1

    if (bf_label not in bf):
        bf[bf_label] = 1
    else:
        bf[bf_label] += 1

auto = list(OrderedDict(sorted(auto.items())).values())
neur = list(OrderedDict(sorted(neur.items())).values())
bf = list(OrderedDict(sorted(bf.items())).values())

auto_acc = np.array(list(accumulate(auto))) / num
neur_acc = np.array(list(accumulate(neur))) / num
bf_acc = np.array(list(accumulate(bf))) / num


plt.plot(auto_acc, color='red', label='autograph')
plt.plot(neur_acc, color='blue', label='neuro-vectorizer')
plt.plot(bf_acc, color='green', label='brute-force')
plt.legend(fontsize=12, loc='upper left', frameon=False)
plt.xlabel("Class label", fontsize=14)
plt.ylabel("Cumulative distribution", fontsize=14)
#plt.xticks([])

plt.savefig("dist.png")

def kl_div(p, q):
	return sum(p[i] * np.log(p[i]/q[i]) for i in range(len(p)))

print(neur_acc)
print(bf_acc)

div1 = kl_div(auto, bf)#sum(rel_entr(auto_acc, bf_acc))
div2 = kl_div(neur, bf)#sum(rel_entr(neur_acc, bf_acc))


print(div1, div2)


