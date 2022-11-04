#from utility import measure_execution_time, vocal, vocal2, deep_walk_preprocess, mfa_preprocess
import glob, json
import networkx as nx
import re
from collections import OrderedDict
import csv


def full_text_feat(nodes):
    feat_matrix = []
    out = []
    #with open('vocab.csv') as file:
    with open('lore_vocab.csv') as file:
        lines = file.readlines()

    for line in lines:
        out.append(line.rstrip().replace("\"",""))
    #print(len(out))
    for (node_id, node_dict) in nodes:
        feat = [0] * len(out)
        if 'features' in node_dict: 
            text = node_dict['features']['full_text'][0]
            #print("text = ", text)
            words = re.split('(\[\d+ x \[\d+ x i\d+\]\], \[\d+ x \[\d+ x i\d+\]\]\**)|(\[\d+ x \[\d+ x i\d+\]\]\**)|(\[\d+ x [a-z]\d+\]\**)|,| |\(|\)|=|inbounds|label|\.\.\.', text)
            words = list(filter(None, words))
            for word in words:
                #if re.search("^\d+$|\d+\.\d+e", word):
                #    idx = out.index("num")
                    #print("1, ", word, idx)
                    #print(word)
                #    feat[idx] = 1
                #elif re.search("^%\d+$", word):
                #    vocab.append("reg")
                #else:
                idx = out.index(word)
                    #print("2, ", word, idx)
                feat[out.index(word)] = 1
        feat_matrix.append(feat)        

    return feat_matrix

'''
vocab = []
cnt = 0
#for f in glob.glob("json-small/*.json"):
for f in glob.glob("json_lore/**/*.json", recursive = True):
    fn = f.split('/')[-1].split('.')[0]
    fn_c = fn + '.c'
    with open(f) as fh:
        g = nx.readwrite.json_graph.node_link_graph(json.load(fh))
    #print(nx.adjacency_matrix(g))
    for (node_id, node_dict) in g.nodes(data=True):
        if 'features' in node_dict: 
            text = node_dict['features']['full_text'][0]
            words = re.split('(\[\d+ x \[\d+ x i\d+\]\], \[\d+ x \[\d+ x i\d+\]\]\**)|(\[\d+ x \[\d+ x i\d+\]\]\**)|(\[\d+ x [a-z]\d+\]\**)|,| |\(|\)|=|inbounds|label|\.\.\.', text)
            words = list(filter(None, words))
            for word in words:
                #if re.search("^\d+$|\d+\.\d+e", word):
                    #print(word)
                #    vocab.append("num")
                #elif re.search("^%\d+$", word):
                #    vocab.append("reg")               
                #else:
                vocab.append(word)
            #print("text = ", text)
            #print("words = ", words)
            
            #cnt += 1
        #if (cnt == 10000):        
            #print("vocab = ", vocab)
            #quit()

vocab = list(OrderedDict.fromkeys(vocab))
voc_csv = []
for token in vocab:
    voc_csv.append([token])
header = ['token'] 
with open('lore_vocab.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    # write the header
    #writer.writerow(header)
    # write multiple rows
    writer.writerows(voc_csv)    
#print("vocab = ", voc_csv)
'''

#for f in glob.glob("json-small/*.json"):
#    fn = f.split('/')[-1].split('.')[0]
#    fn_c = fn + '.c'
#    with open(f) as fh:
#        g = nx.readwrite.json_graph.node_link_graph(json.load(fh))
#    feat = full_text_feat(g.nodes(data=True))
