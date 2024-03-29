from dataset import GraphDataset
#import matplotlib.pyplot as plt
import networkx as nx
import dgl
import torch
###############################################################################
# The return type of :func:`dgl.batch` is still a graph. In the same way, 
# a batch of tensors is still a tensor. This means that any code that works
# for one graph immediately works for a batch of graphs. More importantly,
# because DGL processes messages on all nodes and edges in parallel, this greatly
# improves efficiency.
#
# Graph classifier
# ----------------
# Graph classification proceeds as follows.
#
# .. image:: https://data.dgl.ai/tutorial/batch/graph_classifier.png
#
# From a batch of graphs, perform message passing and graph convolution
# for nodes to communicate with others. After message passing, compute a
# tensor for graph representation from node (and edge) attributes. This step might 
# be called readout or aggregation. Finally, the graph 
# representations are fed into a classifier :math:`g` to predict the graph labels.
#
# Graph convolution layer can be found in the ``dgl.nn.<backend>`` submodule.

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GatedGraphConv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold as kfold
import torch as th



def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    filenames, graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return filenames, batched_graph, torch.tensor(labels)

dataset = GraphDataset()
filename, graph, label = dataset[0]
print(filename, label)
#embed = nn.Embedding()


class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier, self).__init__()
        #self.mode = mode
        self.conv1 = GraphConv(in_dim, hidden_dim)
        #self.conv2 = GraphConv(hidden_dim, hidden_dim) # graph attention network / gated GNN
        #self.conv3 = GraphConv(hidden_dim, hidden_dim) # graph attention network / gated GNN
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # node feature
        #print(g.ndata['w'])
        #h = g.ndata['m'].view(-1,1).float()
        #print("ndata = ", g.ndata['m'])
        h = g.ndata['m'].float()
        g = dgl.add_self_loop(g)
        #em
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        #print("h = ", h)
        
        #h = F.relu(self.conv2(g, h))
        #h = F.relu(self.conv3(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        #print("classify hg = ", self.classify(hg))
        return hg, self.classify(hg)

class GATEDClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, t=5, e_type=1):
        super(GATEDClassifier, self).__init__()
        self.conv1 = GatedGraphConv(in_dim, hidden_dim, t, e_type)
        # self.conv2 = GatedGraphConv(hidden_dim, hidden_dim, t, e_type) # gated gnn
        # self.conv3 = GatedGraphConv(hidden_dim, hidden_dim, t, e_type) # gated gnn
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # node feature
        # feature = nodeFeatures(g, "multifractal")
        h = g.ndata["m"].float()
        g = dgl.add_self_loop(g)

        # h = g.in_degrees().view(-1, 1).float()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h, th.tensor([0 for i in range(dgl.DGLGraph.number_of_nodes(g))])))
        #h = F.relu(self.conv2(g, h))
        #h = F.relu(self.conv3(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return hg, self.classify(hg)

class GATClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, num_heads=1):
        super(GATClassifier, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads=num_heads)
        self.conv2 = GATConv(hidden_dim, hidden_dim, num_heads=num_heads) # graph attention network
        self.conv3 = GATConv(hidden_dim, hidden_dim, num_heads=num_heads) # graph attention network
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # node feature
        h = g.in_degrees().view(-1, 1).float()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)

def difference(lst1, lst2): 
    return list(set(lst1) - set(lst2)) 

num_instances = len(dataset)
test_ratio = 0.2
test_size = int(num_instances * test_ratio)
train_size = num_instances - test_size


print(num_instances, train_size, test_size)






#trainset = dataset
#data_loader = DataLoader(trainset, batch_size=4, shuffle=True,
#                         collate_fn=collate)


# Create model

kfold = 5
#if (num_instances % kfold != 0):
#    assert False, "Please select a new kfold value."
num_per_fold = int(num_instances / kfold)
BATCH_SIZE = 1
NUM_EPOCHES = 100

#num_neurons = [8, 16, 32, 64, 100, 128, 200, 256]
num_neurons = [128]
acc_list = []
acc_list1 = []

save_features = 1
#mode = "multifractal"
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
        model = GCNClassifier(3, num, 1)
        #model = GATClassifier(1, num, 1) 
        #model = GATEDClassifier(1, num, 1)
        loss_func = torch.nn.MSELoss()
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
            for iter, (filename, bg, label) in enumerate(train_loader):
                #print(iter, filename, label)
                #print("bg = ", bg)
                prediction = model(bg)
                #bg = dgl.add_self_loop(bg)
                feat, prediction = model(bg)
                if (epoch == 1):                
                    print("feat = ", feat)
                    quit()
                #print("pred = ", prediction[0], ", feat = ", feat)
                
                print("pred = ", prediction, ", label = ", label)
                #loss = loss_func(prediction, label)
                #print("loss = ", loss)
                loss = loss_func(prediction[0], label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
            #print("a = ", a)
            epoch_loss /= (iter + 1)
            print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss) 
      
        
        # Evaluate the testing dataset
        model.eval()
        # Convert a list of tuples to two lists
        _, test_X, test_Y = map(list, zip(*test_data))
        acc = 0
        acc1 = 0
        for idx in range(len(test_X)):
            x = test_X[idx]
            y = test_Y[idx]
            #print("batch size = ", x)
            #print("#nodes = ", test_bg.batch_num_nodes())
            #test_bg = dgl.add_self_loop(test_bg)
            y = torch.tensor(y).float().view(-1, 1)
            #print('ll = ', ll)
            _, pred = model(x)
            #print(pred, y)
            acc += (abs(pred - y) / abs(y))
            #probs_Y = torch.softmax(, 1)
            #print("In fold ", kf, ', sampled = ', y, ', argmax = ', probs_Y)
            #print("In fold ", kf, ', len = ', len(test_Y))
            #print("label = ", y)
            #probs_Y = probs_Y[0]
            #sampled_Y = torch.multinomial(probs_Y, 1)
            #argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
            #print("In fold ", kf, ', sampled = ', sampled_Y, ', argmax = ', argmax_Y)
            #if (sampled_Y == y):
            #    acc += 1
            #if (argmax_Y == y):
            #    acc1 += 1
        #acc = acc / len(test_Y) * 100
        #acc1 = acc1 / len(test_Y) * 100
        print("In fold ", kf, ', Accuracy of sampled predictions on the test set: ', acc)
        #print("In fold ", kf, ', Accuracy of sampled predictions on the test set: {:.4f}%'.format(acc1))
        #total_acc1 += acc1
        total_acc += acc
        
        feat_mat = {}
        #if (save_features == 1):
        #    for iter, (filename, bg, label) in enumerate(full_loader):
        #        feat, pred = model(bg)
        #        print(filename)
        #        print(label)
        #        print(feat)
                
        break
        
    #total_acc = total_acc / kfold
    #total_acc1 = total_acc1 / kfold
    acc_list.append(acc)
    #acc_list1.append(total_acc1)
print("Total accuracy cross validation = ", acc_list)
#print("Total accuracy cross validation = ", acc_list1)

'''
y_true, y_pred, y_prob  = [], [], []
with torch.no_grad():
  for x, y in test_loader:
    # ground truth
    y = list(y.numpy())
    y_true += y
    
    x = x.float().to(device)
    outputs = model(x)

    # predicted label
    _, predicted = torch.max(outputs.data, 1)
    predicted = list(predicted.cpu().numpy())
    y_pred += predicted
    
    # probability for each label
    prob = list(outputs.cpu().numpy())
    y_prob += prob


# calculating overall accuracy
num_correct = 0

for i in range(len(y_true)):
  if y_true[i] == y_pred[i]:
    num_correct += 1

print("Accuracy: ", num_correct/len(y_true))



model.eval()
# Convert a list of tuples to two lists
test_X, test_Y = map(list, zip(*test_data))
test_bg = dgl.batch(test_X)
test_Y = torch.tensor(test_Y).float().view(-1, 1)
probs_Y = torch.softmax(model(test_bg), 1)
sampled_Y = torch.multinomial(probs_Y, 1)
argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
    (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
    (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))

'''
