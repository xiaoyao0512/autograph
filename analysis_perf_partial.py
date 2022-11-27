import pickle
import numpy as np
import json
import networkx as nx
import mfa

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
        vf_if[fn_c] = (rt_mean, label)
    else:
        rt_mean_pre = vf_if[fn_c][0]
        if rt_mean < rt_mean_pre:
            vf_if[fn_c] = (rt_mean, label)    
    if fn_c not in times.keys():
        times[fn_c] = {}
    if label not in times[fn_c].keys():
        times[fn_c][label] = {}
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


for f in vf_if:
    label = vf_if[f][1]
    if label not in label2file:
        label2file[label] = [f]
    else:
        label2file[label].append(f)

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
def metrics(graph_file):
    with open('lore_graphs/'+graph_file+'.json') as fh:
        g = nx.readwrite.json_graph.node_link_graph(json.load(fh))
    g = g.to_undirected()
    apl = nx.average_shortest_path_length(g)
    diam = nx.diameter(g)
    #ecc = nx.eccentricity(g)
    den = nx.density(g)
    nodes = g.number_of_nodes()
    edges = g.number_of_edges()
    
    Q = [q for q in range(-10,11,1)]
    ntau = mfa.NFD(g, Q)
    al_list, fal_list = mfa.nspectrum(ntau, Q)
    dim_list = mfa.ndimension(ntau, Q)


    return nodes, edges, apl, diam, den, dim_list[11], max(al_list)-min(al_list)
# investigate graph properties in best and worst

node_l = []
edge_l = []
apl_l = []
diam_l = []
den_l = []
dim_l = []
spec_l = []

for i in range(25):
    print(i)
    n_t = []
    e_t = []
    a_t = []
    dm_t = []
    den_t = []
    dim_t = []
    s_t = []
    print(len(label2file[i]))
    '''
    for f in label2file[i]:
        #print(f)
        n, e, a, dm, den, dim, s = metrics(f)
        n_t.append(n)
        e_t.append(e)
        a_t.append(a)
        dm_t.append(dm)
        den_t.append(den)
        dim_t.append(dim)
        s_t.append(s)
    node_l.append(n_t)
    edge_l.append(e_t)
    apl_l.append(a_t)
    diam_l.append(dm_t)
    den_l.append(den_t)
    dim_l.append(dim_t)
    spec_l.append(s_t)
    '''
json_obj = {
    'node': node_l,
    'edge': edge_l,
    'apl': apl_l,
    'diam': diam_l,
    'den': den_l,
    'dim': dim_l,
    'spec': spec_l
        }

with open('graph_data.json', 'w') as fp:
    json.dump(json_obj, fp)
