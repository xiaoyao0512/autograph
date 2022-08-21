'''
from utility import measure_execution_time, vocal, vocal2, deep_walk_preprocess, mfa_preprocess
import glob, json
import networkx as nx

for f in glob.glob("json-small/*.json"):
    fn = f.split('/')[-1].split('.')[0]
    fn_c = fn + '.c'
    with open(f) as fh:
        g = nx.readwrite.json_graph.node_link_graph(json.load(fh))
        print(g.nodes(data=True))
        quit()
        #print(g.nodes)
        #feat = vocal(g.nodes(data=True))
        #feat = vocal2(g.nodes(data=True))
        feat = vocal(g.nodes(data=True), g.degree)
'''
import json

with open('features.json') as f:
    features = json.load(f)
feats = features["feat"]
labels = features["labels"]
files = features["files"]

features_new = {}
files_new = []
for f in files:
    files_new.append(f+".c")
features_new["feat"] = feats
features_new["labels"] = labels
features_new["files"] = files_new

with open('features.json', 'w') as f:
    json.dump(features_new, f) 

print(feats)
