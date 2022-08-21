import json
import dgl, glob
import torch as th
import networkx as nx
from dgl.nn import SumPooling, AvgPooling, MaxPooling, SortPooling, GlobalAttentionPooling



#dims = [8, 16, 32, 64, 128]
dims = [32, 64, 128]
pool_types = ['sum', 'avg', 'max', 'att', 'sort']

#exec_time = {}
#with open('exec_time.json') as f:
#    exec_time = json.load(f)
#filenames = list(exec_time.keys())

for dim in dims:
    
    emb = {}
    with open('embeddings2_'+str(dim)+'.json') as f:
        emb = json.load(f)
   
    for pool_type in pool_types:
        print(dim, pool_type)
        emb_pooling = {}
        pool = None
        if (pool_type == 'sum'):
            pool = SumPooling()
        elif (pool_type == 'avg'):
            pool = AvgPooling()
        elif (pool_type == 'max'):
            pool = MaxPooling()
        elif (pool_type == 'sort'):
            pool = SortPooling(k=2)
        elif (pool_type == 'att'):
            gate_nn = th.nn.Linear(dim, 1)
            pool = GlobalAttentionPooling(gate_nn)



        for f in glob.glob("json-small/*.json"):
            fn = f.split('/')[-1].split('.')[0]
            fn_c = fn + '.c'
            with open(f) as fh:
                g = nx.readwrite.json_graph.node_link_graph(json.load(fh))
                # calculate the graph features
                #print(g.nodes)
                g = dgl.from_networkx(g)
            feat = emb[fn]
            feat = th.FloatTensor(feat)
            emb_pooling[fn_c] = pool(g, feat).tolist()


        with open('embeddings2_'+str(dim)+'_'+pool_type+'_pooling.json', 'w') as f:
            json.dump(emb_pooling, f) 
