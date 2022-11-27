import pickle
import numpy as np
import json
import networkx as nx
import mfa
import matplotlib.pyplot as plt
import matplotlib as mpl

def one2two(label):
    VF_list = [1,2,4,8,16] # TODO: change this to match your hardware
    IF_list = [1,2,4,8,16] # TODO: change this to match your hardware
    y1 = VF_list[int(int(label) / 5)]
    y2 = IF_list[int(int(label) % 5)]
    return (y1, y2)

def two2one(VF, IF):
    VF_list = [1,2,4,8,16] # TODO: change this to match your hardware
    IF_list = [1,2,4,8,16] # TODO: change this to match your hardware
    VF_idx = VF_list.index(VF)
    IF_idx = IF_list.index(IF)
    return 5 * VF_idx + IF_idx


f = open('lore_runtimes.pickle', 'rb')
runtimes = pickle.load(f)    
f.close()
f = open('lore_runtimes_none_pragma.pickle', 'rb')
base_runtimes = pickle.load(f)    
f.close()

vf_if = {}
times = {}
files_VF_IF = runtimes.keys()
vf_list = []
if_list = []
baseline = {}
sp = {}
label2file = {}
for file_VF_IF in files_VF_IF:
    tmp = file_VF_IF.rpartition('.')
    fn = tmp[0]
    tmp = tmp[2].split('-')
    VF = int(tmp[0])
    IF = int(tmp[1])
    fn_c = fn
    label = two2one(VF, IF)

    rt_mean = np.mean(runtimes[file_VF_IF])
    base_mean = np.mean(base_runtimes[fn])
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

for f in baseline:
    sp[f] = baseline[f] / vf_if[f][0]



#sp_sort = dict(sorted(sp.items(), key=lambda item: item[1]))
#top5 = list(sp_sort.keys()][0:5]
#print("top = ", list(sp_sort.values())[0:5])
'''
print(label2file.keys())
benchmarks = ['spec']
methods = ['autograph']
for benchmark in benchmarks:
    f = open('lore_testing_autograph_800k_'+benchmark+'.json', 'rb')
    pred = json.load(f)
    f.close()
    files = list(pred.keys())
    sp_bench = {}
    for f in files:
        sp_bench[f] = sp[f]
    sp_bench_sort = dict(sorted(sp_bench.items(), key=lambda item: item[1], reverse=True))
    print("benchmark = ", benchmark)
    top5 = list(sp_bench_sort.keys())[0:5]
    print("top 5 = ", top5)
    print("top 5 = ", list(sp_bench_sort.values())[0:5])
    for method in methods:
        f = open('lore_testing_'+method+'_800k_'+benchmark+'.json', 'rb')
        pred = json.load(f)
        f.close()
        print("method  = ", method)
        #print("benchmark = ", benchmark, " method = ", method)
        for f in top5:
            VF, IF = one2two(pred[f])
            print("speedup = ", baseline[f] / times[f][VF][IF])
best_file = top5[2]
worst_file = top5[3]

print(best_file, worst_file)
'''


def plot_scatter(name, feats):
    feat = []
    target = []
    for i in range(len(feats)):
        ft = feats[i]
        size = len(ft)
        t = [i] * size
        feat.extend(ft)
        target.extend(t)

    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["figure.autolayout"] = True
    # Create figure object and store it in a variable called ', fig'
    fig = plt.figure()#figsize=(3, 3))

    # Add axes object to our figure that takes up entire figure
    ax = plt.gca()#fig.add_axes([0, 0, 1, 1])


    # Set the x/y-axis scaling to logarithmic
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Edit the major and minor ticks of the x and y axes
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')

    # Edit the major and minor tick locations of x and y axes
    ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))

    plt.scatter(feat, target, s=20, color='#00b3b3', label='Data')
    plt.xlabel(name, labelpad=10)
    plt.ylabel("Class label", labelpad=10)
    plt.savefig("figs/"+name+".png")
    plt.close()

json_obj = {}
with open('graph_data.json') as fp:
    json_obj = json.load(fp)

for feat in json_obj:
    plot_scatter(feat, json_obj[feat])

'''
node = json_obj['node']
edge = json_obj['edge']
apl = json_obj['apl']
diam = json_obj['diam']
den = json_obj['den']
dim = json_obj['dim']
spec = json_obj['spec']
'''
