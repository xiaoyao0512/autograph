from dataset import GraphDataset
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import json

from model import SAGE, compute_acc_unsupervised as compute_acc
from negative_sampler import NegativeSampler


class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = torch.cat([pos_score, neg_score])
        label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss

def evaluate(model, g, nfeat, labels, train_nids, val_nids, test_nids, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        # single gpu
        if isinstance(model, SAGE):
            pred = model.inference(g, nfeat, device, args.batch_size, args.num_workers)
        # multi gpu
        else:
            pred = model.module.inference(g, nfeat, device, args.batch_size, args.num_workers)
    model.train()
    return compute_acc(pred, labels, train_nids, val_nids, test_nids)




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
graphs = dataset.graphs


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
batch_size = 64
num_epoches = 100

#num_neurons = [8, 16, 32, 64, 100, 128, 200, 256]
num_neurons = [16]
acc_list = []
acc_list1 = []

in_feats = 13
#num_hidden = 16
num_layers = 2
dropout = 0
avg = 0
iter_pos = []
iter_neg = []
iter_d = []
iter_t = []
best_eval_acc = 0
best_test_acc = 0
fan_out = '10,25'
num_negs = 1
neg_share = False
emb = {}
for num_hidden in num_neurons:
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

        for g in graphs:
            model = SAGE(in_feats, num_hidden, num_hidden, num_layers, F.relu, dropout)
            loss_fcn = CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.05)

            #train_data, test_data = torch.utils.data.random_split(dataset, (train_size, test_size))
            #train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle = True, collate_fn=collate)
            #test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle = False, collate_fn=collate)
            #full_loader = torch.utils.data.DataLoader(dataset, shuffle = False, collate_fn=collate)
            epoch_losses = []
            model.train()
        
            n_edges = g.num_edges()
            train_seeds = torch.arange(n_edges)
            nfeat = g.ndata['m']
            # Create sampler
            sampler = dgl.dataloading.MultiLayerNeighborSampler(
                [int(fanout) for fanout in fan_out.split(',')])
            dataloader = dgl.dataloading.EdgeDataLoader(
                g, train_seeds, sampler, exclude='reverse_id',
                # For each edge with ID e in Reddit dataset, the reverse edge is e ± |E|/2.
                reverse_eids=torch.cat([
                    torch.arange(n_edges // 2, n_edges),
                    torch.arange(0, n_edges // 2)]).to(train_seeds),
                negative_sampler=NegativeSampler(g, num_negs, neg_share),
                device='cpu',#device,
                #use_ddp = False,#n_gpus > 1,
                batch_size = batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0#args.num_workers
                )

            for epoch in range(num_epoches):
                epoch_loss = 0
                for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
                    batch_inputs = nfeat[input_nodes]
                    pos_graph = pos_graph
                    neg_graph = neg_graph
                    blocks = [block.int() for block in blocks]
                    batch_pred = model(blocks, batch_inputs)
                    loss = loss_fcn(batch_pred, pos_graph, neg_graph)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().item()
                #print("a = ", a)
                epoch_loss /= (step + 1)
                print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
                #epoch_losses.append(epoch_loss) 
            #break

            model.eval()
            nfeat = g.ndata['m']
            pred = model.inference(g, nfeat, batch_size)
            print("pred = ", pred)  
            emb[g.filename] = pred.tolist()         
        break
        
    #acc_list.append(acc)
    #acc_list1.append(total_acc1)
#print("Total accuracy cross validation = ", acc_list)
'''
model.eval()
emb = {}
for g in graphs:
    nfeat = g.ndata['m']
    pred = model.inference(g, nfeat, batch_size)
    print("pred = ", pred)
    emb[g.filename] = pred.tolist()
'''   
with open('embeddings.json', 'w') as f:
    json.dump(emb, f) 

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
