import pickle, json
import numpy as np
from torch.utils.data import Dataset
import networkx as nx
import dgl, torch
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GatedGraphConv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold as kfold
from collections import Counter


class ProgramDataset(Dataset):
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
        
        for fn in self.filenames:
            self.exec_times.append(exec_time[fn][1])
            g = nx.from_dict_of_dicts(exec_time[fn][2])
            g = dgl.from_networkx(g)
            #print(len(exec_time[i]))
            filename = fn.split('.')[0]
            feat = emb[fn]
            #print(feat)
            # add structural embedding to semantic embedding
            self.labels.append(vf_if[filename][1])
            g.ndata["m"] = torch.FloatTensor(feat)
            g.filename = fn
            #print("filename = ", i)
            #feat = nodeFeatures(g, 'simple')
            #g.ndata["m"] = feat
            #print(g.ndata["m"].view(-1,1))
            
            self.graphs.append(g)
        
        #print("labels = ", self.labels)
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
       

class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier, self).__init__()
        #self.mode = mode
        self.conv1 = GraphConv(in_dim, hidden_dim)
        #self.conv2 = GraphConv(hidden_dim, hidden_dim) # graph attention network / gated GNN
        #self.conv3 = GraphConv(hidden_dim, hidden_dim) # graph attention network / gated GNN
        self.classify1 = nn.Linear(hidden_dim, n_classes)
        self.classify2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # node feature
        #print(g.ndata['w'])
        #h = g.ndata['m'].view(-1,1).float()
        h = g.ndata['m'].float()
        g = dgl.add_self_loop(g)
        #em
        print("g = ", g.number_of_nodes())
        #g = dgl.from_networkx(g)
        print("h = ", h)
        print("h shape = ", h.shape)  
        quit()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        #h = F.relu(self.conv2(g, h))
        #h = F.relu(self.conv3(g, h))

        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify1(hg), self.classify2(hg)

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


def difference(lst1, lst2): 
    return list(set(lst1) - set(lst2))


dims = [8, 16, 32, 64, 128]
dim = 8

emb = {}
with open('embeddings_'+str(dim)+'.json') as f:
    emb = json.load(f)

f = open('runtimes.pickle', 'rb')
runtimes = pickle.load(f)    
f.close()

#print("emb keys = ", emb.keys())

vf_if = {}
files_VF_IF = runtimes.keys()
vf_list = []
if_list = []
for file_VF_IF in files_VF_IF:
    tmp = file_VF_IF.split('.')
    fn = tmp[0]
    tmp = tmp[1].split('-')
    VF = int(tmp[0])
    IF = int(tmp[1])
    vf_list.append(VF)
    if_list.append(IF)
    rt_mean = np.mean(runtimes[file_VF_IF])
    #print("filename = ", fn)
    #print("VF = ", VF)
    #print("IF = ", IF)
    #print("mean = ", rt_mean)
    if fn not in vf_if.keys():
        vf_if[fn] = (rt_mean, VF, IF)
    else:
        rt_mean_pre = vf_if[fn][0]
        if rt_mean < rt_mean_pre:
            vf_if[fn] = (rt_mean, VF, IF)

#print("vf_if = ", vf_if)
#print("max VF = ", np.max(vf_list))
#print("min VF = ", np.min(vf_list))
#print("max IF = ", np.max(if_list))
#print("min IF = ", np.min(if_list))



dataset = ProgramDataset()

num_instances = len(dataset)
test_ratio = 0.2
test_size = int(num_instances * test_ratio)
train_size = num_instances - test_size


print(num_instances, train_size, test_size)

kfold = 5
#if (num_instances % kfold != 0):
#    assert False, "Please select a new kfold value."
num_per_fold = int(num_instances / kfold)
BATCH_SIZE = 1
NUM_EPOCHES = 200

num_neurons = [8, 16, 32, 64, 128]
#num_neurons = [64]
acc_list = []
acc_list1 = []


for num in num_neurons:
    total_acc = 0
    total_acc1 = 0
    # for kf in range(1):
    for kf in range(kfold):
        test_set = range(kf*num_per_fold, (kf+1)*num_per_fold)
        train_set = difference(range(num_instances), test_set)
        print("fold = ", kf)
        print("test_set = ", test_set)
        #print("train_set = ", train_set)

        train_data = torch.utils.data.Subset(dataset, train_set)
        test_data = torch.utils.data.Subset(dataset, test_set)

        #if (mode == "multifractal"):
        #    inp = 6
        model = GCNClassifier(dim, num, 17)
        #model = GATClassifier(1, num, 1) 
        #model = GATEDClassifier(1, num, 10)
        #loss_func = torch.nn.MSELoss()
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        #train_data, test_data = torch.utils.data.random_split(dataset, (train_size, test_size))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle = True, collate_fn=collate)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle = False, collate_fn=collate)
        full_loader = torch.utils.data.DataLoader(dataset, shuffle = False, collate_fn=collate)
        # Train the model    
        model.train()
        epoch_losses = []
        
        for epoch in range(NUM_EPOCHES):
            epoch_loss = 0
            #print("epoch = ", epoch)
            for iter, (bg, label) in enumerate(train_loader):
                #print(iter, label)
                #prediction = model(bg)
                #bg = dgl.add_self_loop(bg)
                pred1, pred2 = model(bg)
                #if (epoch == NUM_EPOCHES - 1):                
                #    print("pred1 = ", pred1, "pred2 = ", pred2, ", label = ", label)
                #print("pred1 = ", pred1, "pred2 = ", pred2, ", label = ", label)
                #loss = loss_func(prediction, label)
                #print("loss = ", loss)
                #print("pred = ", prediction, ", label = ", label)
                #quit()
                label = label.tolist()[0]
                label1 = torch.LongTensor([label[0]])
                label2 = torch.LongTensor([label[1]])
                loss1 = loss_func(pred1, label1)
                loss2 = loss_func(pred2, label2)
                loss = loss1 + loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
            #print("a = ", a)
            epoch_loss /= (iter + 1)
            #print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss) 
      
        
        # Evaluate the testing dataset
        print("start evaluating the testing dataset")
        model.eval()
        # Convert a list of tuples to two lists
        test_X, test_Y = map(list, zip(*test_data))
        acc = 0
        acc1 = 0
        for idx in range(len(test_X)):
            x = test_X[idx]
            y = test_Y[idx]
            #print("#nodes = ", test_bg.batch_num_nodes())
            #test_bg = dgl.add_self_loop(test_bg)
            #y = torch.tensor(y).float().view(-1, 1)
            #print('ll = ', ll)
            pred = model(x)
            pred1 = pred[0]
            pred2 = pred[1]
            sampled_y1 = np.argmax(pred1.tolist())
            sampled_y2 = np.argmax(pred2.tolist())
            #print("y = ", y)
            #print("pred = ", pred)
            #print(pred, y)
            #acc += abs(pred - y)
            #probs_Y = torch.softmax(pred, 1)
            #print("In fold ", kf, ', y = ', y, ', probs_Y = ', probs_Y)
            #print("In fold ", kf, ', len = ', len(test_Y))
            #print("label = ", y)
            #probs_Y = probs_Y[0]
            #sampled_Y = torch.multinomial(probs_Y, 1)
            #argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
            #print("In fold ", kf, ', sampled = ', sampled_Y, ', argmax = ', argmax_Y)
            y1 = y[0]
            y2 = y[1]
            if (sampled_y1 == y1):
                acc += 1
            if (sampled_y2 == y2):
                acc += 1

        
        acc = acc / (len(test_Y)) * 100
        #acc1 = acc1 / len(test_Y) * 100
        print("In fold ", kf, ', Accuracy of sampled predictions on the test set: ', acc)
        #print("In fold ", kf, ', Accuracy of sampled predictions on the test set: ', acc1)
        #total_acc1 += acc1
        #total_acc += acc
        
        break
        
    #total_acc = total_acc / kfold
    #total_acc1 = total_acc1 / kfold
    acc_list.append(acc)
    #acc_list1.append(acc1)
print(num_neurons)
print("Total accuracy cross validation = ", acc_list)
#print("Total accuracy cross validation = ", acc_list1)

