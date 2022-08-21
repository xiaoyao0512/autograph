#from utility import measure_execution_time, vocal, vocal2, deep_walk_preprocess, mfa_preprocess
import glob, json
import networkx as nx
import csv


def dict_from_csv(filename):
    bag = {}
    with open(filename, 'r') as file:
        csvreader = csv.reader(file, delimiter="\t")
        header = next(csvreader)
        #print("header = ", header)
        for elem in header:
            bag[elem] = []
        for row in csvreader:
            for i in range(len(row)):
                #print("i = ", i)
                #print("row = ", row)
                val = row[i]
                key = header[i]
                bag[key].append(val)
    return bag

for f in glob.glob("nxgraphs/*.json"):
    fn = f.split('/')[-1].split('.')[0]
    fn_c = fn + '.c'
    bag_of_words = dict_from_csv("vocab/programl.csv")
    if (fn_c == "s8_128_0.c"):
        continue
    with open(f) as fh:
        g = nx.readwrite.json_graph.node_link_graph(json.load(fh))
    feat = g.nodes(data=True)
    print(fn_c)

    for (node_id, node_dict) in feat:
        val = 0
        if (node_dict['text'] in bag_of_words['text']):
            text_idx = bag_of_words['text'].index(node_dict['text'])
            val = text_idx + 1
        if (node_dict['idx'] != val):
            print("something's wrong!")
            print("mine, Guixiang = ", val, node_dict['idx'])
            print(f)
            quit()
        else:
            
        
