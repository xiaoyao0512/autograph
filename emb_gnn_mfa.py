import networkx as nx
import os, glob
import tensorflow as tf
import torch
from torch import nn
from os.path import exists
import json
from torch.utils.data import Dataset
import pickle
import numpy as np
from collections import Counter
from node import node_dimension
import dgl, torch
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GatedGraphConv
import torch.nn.functional as F
from random import randint 


def preprocess(vf_if):
    if exists ("multifractal.json"):
        with open("multifractal.json") as f:
            features = json.load(f)
        graphs = []
        for f in glob.glob("json-small/*.json"):
            fn = f.split('/')[-1].split('.')[0]
            fn_c = fn + '.c'
            with open(f) as fh:
                g = nx.readwrite.json_graph.node_link_graph(json.load(fh))
            g = dgl.from_networkx(g)
            g.ndata["m"] = torch.FloatTensor(features['feats'][fn_c])
            g.filename = fn  
            #print(fn)
            graphs.append(g)                
        return graphs, features["labels"], features["files"]   
    else: 

        files = []
        graphs = []
        labels = []
        feats = {}
        for f in glob.glob("json-small/*.json"):
            fn = f.split('/')[-1].split('.')[0]
            fn_c = fn + '.c'
            files.append(fn_c)
            with open(f) as fh:
                g = nx.readwrite.json_graph.node_link_graph(json.load(fh))
                # calculate the graph features
                #print(g.nodes)
            feat = node_dimension(g)  
            feats[fn_c] = feat         
            g = dgl.from_networkx(g)
            g.ndata["m"] = torch.FloatTensor(feat)
            g.filename = fn  
            #print(fn)
            graphs.append(g)
            labels.append(vf_if[fn][1])
        
        features = {}
        features['feats'] = feats
        features['labels'] = labels
        features['files'] = files
        with open('multifractal.json', 'w') as f:
            json.dump(features, f) 
        return graphs, labels, files
        
class GraphDataset(Dataset):
    def __init__(self, vf_if):        
        self.graphs, self.labels, self.fn = preprocess(vf_if)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.fn[idx]

def difference(lst1, lst2): 
    return list(set(lst1) - set(lst2))

class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier, self).__init__()
        #self.mode = mode
        self.conv1 = GraphConv(in_dim, hidden_dim)
        #self.conv2 = GraphConv(hidden_dim, hidden_dim) # graph attention network / gated GNN
        #self.conv3 = GraphConv(hidden_dim, hidden_dim) # graph attention network / gated GNN
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.log = nn.LogSoftmax(dim=1)

    def forward(self, g):
        # node feature
        #print(g.ndata['w'])
        #h = g.ndata['m'].view(-1,1).float()
        h = g.ndata['m'].float()
        g = dgl.add_self_loop(g)
        #em
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        #h = F.relu(self.conv2(g, h))
        #h = F.relu(self.conv3(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.max_nodes(g, 'h')
        return self.log(self.classify(hg))


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels, filenames = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels), filenames



f = open('runtimes.pickle', 'rb')
runtimes = pickle.load(f)    
f.close()

f = open('runtimes_none_pragma.pickle', 'rb')
base_runtimes = pickle.load(f)    
f.close()


VF_list = [1, 2, 4, 8, 16]
IF_list = [1, 2, 4, 8, 16]

vf_if = {}
times = {}
files_VF_IF = runtimes.keys()
vf_list = []
if_list = []
baseline = {}
for file_VF_IF in files_VF_IF:
    tmp = file_VF_IF.split('.')
    fn = tmp[0]
    tmp = tmp[1].split('-')
    VF = int(tmp[0])
    IF = int(tmp[1])
    label = VF_list.index(VF) * 5 + IF_list.index(IF)
    fn_c = fn + '.c'
    rt_mean = np.mean(runtimes[file_VF_IF])
    base_mean = np.mean(base_runtimes[fn])
    #print("filename = ", fn)
    #print("VF = ", VF)
    #print("IF = ", IF)
    #print("mean = ", rt_mean)
    if fn not in vf_if.keys():
        vf_if[fn] = (rt_mean, label)
    else:
        rt_mean_pre = vf_if[fn][0]
        if rt_mean < rt_mean_pre:
            vf_if[fn] = (rt_mean, label)    
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




kfold = 5
#if (num_instances % kfold != 0):
#    assert False, "Please select a new kfold value."
dataset = GraphDataset(vf_if)
num_instances = len(dataset)
test_ratio = 0.2
test_size = int(num_instances * test_ratio)
train_size = num_instances - test_size
num_per_fold = int(num_instances / kfold)
BATCH_SIZE = 1
NUM_EPOCHES = 200

#num_neurons = [8, 16, 32, 64, 128]
num_neurons = [25]



print(num_instances, train_size, test_size)
# Layer details for the neural network
input_size = 1
#hidden_sizes = [128, 64]
output_size1 = len(VF_list)
output_size2 = len(IF_list)
output_size = output_size1 * output_size2


files = {}
acc_list = []
exec_list1 = []
exec_list2 = []
exec_list3 = []
for num in num_neurons:
    total_acc = 0
    total_acc1 = 0
    #files[num] = []
    # for kf in range(1):
    for kf in range(kfold):
        test_set = range(kf*num_per_fold, (kf+1)*num_per_fold)
        train_set = difference(range(num_instances), test_set)
        #print("fold = ", kf)
        #print("test_set = ", test_set)
        #print("train_set = ", train_set)

        train_data = torch.utils.data.Subset(dataset, train_set)
        test_data = torch.utils.data.Subset(dataset, test_set)
        full_data = dataset
        # Build a feed-forward network
        '''
        model1 = nn.Sequential(nn.Linear(input_size, hidden_size),
                              nn.ReLU(),
                              nn.Linear(hidden_size, output_size1),
                              nn.LogSoftmax(dim=1))
        model2 = nn.Sequential(nn.Linear(input_size, hidden_size),
                              nn.ReLU(),
                              nn.Linear(hidden_size, output_size2),
                              nn.LogSoftmax(dim=1))
        '''
        # = nn.Sequential(nn.Linear(input_size, output_size), nn.LogSoftmax(dim=1))
        #model2 = nn.Sequential(
        #                      nn.Linear(input_size, output_size2))
        model = GCNClassifier(1, num, 25)
        loss_func = nn.NLLLoss()
        #loss_func2 = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        #optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.0005)
        #optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
        #train_data, test_data = torch.utils.data.random_split(dataset, (train_size, test_size))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle = True, collate_fn=collate)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle = False, collate_fn=collate)
        full_loader = torch.utils.data.DataLoader(full_data, shuffle = False)

        # Train the model    
        model.train()
        #model2.train()
        epoch_losses = []
        #epoch_losses2 = []
        for epoch in range(NUM_EPOCHES):
            epoch_loss = 0
            #epoch_loss2 = 0
            #print("epoch = ", epoch)
            for iter, (bg, label, fn) in enumerate(train_loader):
                #print(bg, label)
                #prediction = model(bg)
                #bg = dgl.add_self_loop(bg)
                #bg = torch.FloatTensor(bg)
                #print(bg)
                pred = model(bg)
                #pred2 = model2(bg)
                #if (epoch == NUM_EPOCHES - 1):                
                #    print("pred1 = ", pred1, "pred2 = ", pred2, ", label = ", label)
                #print("pred1 = ", pred1, "pred2 = ", pred2, ", label = ", label)
                #loss = loss_func(prediction, label)
                #print("loss = ", loss)
                #print("pred = ", pred, ", label = ", label)
                #quit()
                #print(label)
                #label = label.tolist()[0]
                #label1 = torch.LongTensor([VF_list.index(label[0])])
                #label2 = torch.LongTensor([IF_list.index(label[1])])

                #print(label1, label2)
                loss = loss_func(pred, label)
                #loss2 = loss_func2(pred2, label2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #optimizer2.zero_grad()
                #loss2.backward()
                #optimizer2.step()
                epoch_loss += loss.detach().item()
                #epoch_loss2 += loss2.detach().item()
            #print("a = ", a)
            epoch_loss /= (iter + 1)
            #epoch_loss2 /= (iter + 1)
            #print('model weights - ', model1[0].weight)
            print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss) 
            #epoch_losses2.append(epoch_loss2) 

        
        # Evaluate the testing dataset
        #print("start evaluating the testing dataset")
        print("epoch losses = ", epoch_losses)
        model.eval()
        #model2.eval()
        # Convert a list of tuples to two lists
        test_X, test_Y, test_fn = map(list, zip(*test_data))
        acc = 0
        exec1 = 0
        exec2 = 0
        exec3 = 0
        #continue
        pred_list = []
        test_list = []
        bad = []
        for idx in range(len(test_X)):
            x = test_X[idx]
            y = test_Y[idx]
            test_list.append(int(y))
            fn = test_fn[idx]
            #print("batch size = ", x)
            #print("#nodes = ", test_bg.batch_num_nodes())
            #test_bg = dgl.add_self_loop(test_bg)
            #print('y = ', y)
            #y = torch.tensor(y).float().view(-1, 1)
            #print('y = ', y)
            #print('y00 = ', y[1,0])
            #x = torch.FloatTensor(x)
            '''
            with torch.no_grad():
                logps1 = model1(x)
                logps2 = model2(x)

                sampled_y1 = VF_list[np.argmax(pred1.tolist())]
                sampled_y2 = IF_list[np.argmax(pred2.tolist())]            
            ps1 = torch.exp(logps1)
            ps2 = torch.exp(logps2)
            #print("ps = ", ps)
            probab1 = list(ps1.numpy()[0])
            probab2 = list(ps2.numpy()[0])
            #print("probab = ", probab)
            pred_label1 = VF_list[probab1.index(max(probab1))]
            pred_label2 = IF_list[probab2.index(max(probab2))]
            '''
            with torch.no_grad():
                pred = model(x)
            #pred2 = model2(x)
            ps = torch.exp(pred)
            probab = list(ps.numpy()[0])
            #pred2 = model2(x)
            #pred_label1 = VF_list[np.argmax(pred1.tolist())]
            #pred_label2 = IF_list[np.argmax(pred2.tolist())]    
            #print("predicted vf/if = ", pred_label1, pred_label2)
            #print("best vf/if = ", y[0], y[1])
            pred_label = probab.index(max(probab))
            print("fn/predicted/groundtruth = ", fn, pred_label, y)
            if (pred_label == y):
                acc += 1
                pred_list.append(pred_label)
                #files[num].append(fn)

            # measure the execution time of a program with a specific VF/IF
            #print("sample vf/if = ", y[0], y[1])
            sampled_y1 = VF_list[int(int(y) / 5)]
            sampled_y2 = IF_list[int(y) % 5]
            VF_pred = VF_list[int(int(pred_label) / 5)]
            IF_pred = IF_list[int(pred_label) % 5]
            print(VF_pred, IF_pred)
            print(sampled_y1, sampled_y2)
            t1 = times[fn][sampled_y1][sampled_y2]
            t2 = times[fn][VF_pred][IF_pred]
            t3 = baseline[fn]
            print("fn = ", fn, ", t1 = ", t1, ", t2 = ", t2, ", t3 = ", t3)
            exec1 += abs(t1 - t2)
            speedup_gt = ((t1 - t2) / t1)
            speedup_base = ((t3 - t2) / t3)
            exec2 += speedup_gt
            print("speedup compared to ground truth = ", speedup_gt)
            exec3 += speedup_base
            print("speedup compared to baseline O3 = ", speedup_base)
            if ((abs(speedup_gt) > 2) or (abs(speedup_base) > 2)):
                bad.append(fn)            
        
        acc = acc / (len(test_Y)) * 100
        exec1 = exec1 / len(test_Y)
        exec2 = exec2 / len(test_Y) * 100
        exec3 = exec3 / len(test_Y) * 100
        #acc1 = acc1 / len(test_Y) * 100
        print("In fold ", kf, ', Accuracy of sampled predictions on the test set: ', acc)
        #print("In fold ", kf, ', Accuracy of sampled predictions on the test set: ', acc1)
        #total_acc1 += acc1
        #total_acc += acc
        
        break
        
    #total_acc = total_acc / kfold
    #total_acc1 = total_acc1 / kfold
    acc_list.append(acc)
    exec_list1.append(exec1)
    exec_list2.append(exec2)
    exec_list3.append(exec3)
    #acc_list1.append(acc1)
#print(num_neurons)
pred_freq = Counter(pred_list)
test_freq = Counter(test_list)
acc_class = {}
for dim in test_freq:
    acc_per_class = pred_freq[dim] / test_freq[dim]
    acc_class[dim] = acc_per_class

print("pred_freq = ", pred_freq)
print("test_freq = ", test_freq)
print("acc_per_class = ", acc_class)
print("Total accuracy cross validation = ", acc_list)
print("exec1 = ", exec_list1)
print("exec2 = ", exec_list2)
print("exec3 = ", exec_list3)
print("bad files = ", bad)
#print("files = ", files)

