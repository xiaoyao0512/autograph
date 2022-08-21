from torch.utils.data import Dataset, DataLoader
import torch
import sys
import numpy as np
#import multifractal
import dgl
import os.path
from dgl.data.utils import load_graphs
import json
import networkx as nx
import math
from keras.utils import to_categorical

#glist, label_dict = load_graphs("./data.bin") # glist will be [g1, g2]
def nodeFeatures(g, types):
	#print("In nodeFeatures")
    if (types == 'simple'):
        #print("size = ", g.in_degrees().size(-1))
        return g.in_degrees().view(-1,1)
    elif (types == 'weight'):
        return dgl.khop_adj(g, 1)
    #elif (types == 'multifractal'):
    #   return multifractal.multifractal(g)

def vectorRead(filename):
	temp = []
	fr = open(filename, 'r')
	for line in fr:
		num = line.rstrip()
		temp.append(float(num))
	fr.close()
	return temp

def vectorWrite(lst, filename):
	assert len(lst) != 0
	fw = open(filename, 'w')
	for num in lst:
		fw.write("{}\n".format(num))
	fw.close()

def bins(exec_times):
    exec_min = min(exec_times)
    exec_max = max(exec_times)
    num = 10
    tmp = float((exec_max - exec_min) / num)


    labels = []
    label = 0
    for exec_time in exec_times:
        if (exec_time == exec_max):
            label = num - 1
        else:
            label = math.floor((exec_time - exec_min) / tmp)
        # convert it into one hot encoding 
        #label = to_categorical(label, num_classes=10)
        labels.append(label)
    return labels

def z(exec_times):
    exec_mean = np.mean(exec_times)
    exec_std = np.std(exec_times)
    exec_normalized = []
    for exec_time in exec_times:
        time = (exec_time - exec_mean) / exec_std
        exec_normalized.append(float(time))
    return exec_normalized

def scaling(exec_times, mode):
    if (mode == "bins"):
        # categorical prediction
        return bins(exec_times)
    elif (mode == "z"):
        # still continuous 
        # same as without scaling
        return z(exec_times)
    elif (mode == "no"):
        return exec_times
        

class GraphDataset(Dataset):
    def __init__(self):
        exec_time = {}
        with open('exec_time.json') as f:
            exec_time = json.load(f)
        #print(exec_time["s1_64_sub_0.c"][1])
        self.filenames = list(exec_time.keys())
        #print(self.filenames)
        self.exec_times = []
        self.labels = []
        self.graphs = []
        mode = "z"
        for fn in self.filenames:
            self.exec_times.append(exec_time[fn][1])
            g = nx.from_dict_of_dicts(exec_time[fn][2])
            g = dgl.from_networkx(g)
            #print(len(exec_time[i]))
            feat = exec_time[fn][3]
            #print(feat)
            # add structural embedding to semantic embedding
            #structure = g.in_degrees().view(-1,1)
            #for i in range(len(structure)):
            #    feat[i].append(float(structure[i]))
            #print("feat = ", feat)
            #feat = scaling(feat, mode)
            g.ndata["m"] = torch.FloatTensor(feat)
            g.filename = fn
            #print("filename = ", i)
            #feat = nodeFeatures(g, 'simple')
            #g.ndata["m"] = feat
            #print(g.ndata["m"].view(-1,1))
            
            self.graphs.append(g)
        self.labels = scaling(self.exec_times, mode)
        #print("labels = ", self.labels)
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.filenames[idx], self.graphs[idx], self.labels[idx]


