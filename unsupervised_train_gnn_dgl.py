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
        #print("block_outputs = ", block_outputs.shape)
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
batch_size = 512
num_epoches = 100

#num_neurons = [8, 16, 32, 64, 128]
num_neurons = [128]
acc_list = []
acc_list1 = []

in_feats = 8
#num_hidden = 16
num_layers = 1
dropout = 0.1
avg = 0
iter_pos = []
iter_neg = []
iter_d = []
iter_t = []
best_eval_acc = 0
best_test_acc = 0
fan_out = '10,10' if num_layers == 2 else '10'
num_negs = 1
neg_share = False
for num_hidden in num_neurons:
    print('num_hidden = ', num_hidden)
    total_acc = 0
    total_acc1 = 0
    # for kf in range(1):
    for kf in range(kfold):
        test_set = range(kf*num_per_fold, (kf+1)*num_per_fold)
        train_set = difference(range(num_instances), test_set)
        #print("fold = ", kf)
        #print("test_set = ", test_set)
        #print("train_set = ", train_set)

        train_data = torch.utils.data.Subset(dataset, train_set)
        test_data = torch.utils.data.Subset(dataset, test_set)


        model = SAGE(in_feats, num_hidden, num_hidden, num_layers, F.relu, dropout)

        for name, param in model.named_parameters():
            print("name = ", name)
            print("param = ", param)

        loss_fcn = CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.05)

        #train_data, test_data = torch.utils.data.random_split(dataset, (train_size, test_size))
        #train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle = True, collate_fn=collate)
        #test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle = False, collate_fn=collate)
        #full_loader = torch.utils.data.DataLoader(dataset, shuffle = False, collate_fn=collate)
        epoch_losses = []
        
        for g in graphs:
            
            n_edges = g.num_edges()
            n_nodes = g.num_nodes()
            #print("number of nodes = ", n_nodes)
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
# check weights in a model
# check different graphs or sane gr
# compute the difference between graph embeddings
            for epoch in range(num_epoches):
                epoch_loss = 0
                for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
                    model.train()
                    batch_inputs = nfeat[input_nodes]
                    pos_graph = pos_graph
                    neg_graph = neg_graph
                    blocks = [block.int() for block in blocks]
                    #print("batch_inputs shape = ", batch_inputs.shape)
                    #print("blocks len = ", len(blocks))
                    #print("blocks = ", blocks)
                    batch_pred = model(blocks, batch_inputs)
                    #print("batch_inputs = ", batch_inputs)
                    #print("batch_pred shape = ", batch_pred.shape)
                    #print("pos_graph nodes = ", pos_graph.num_nodes())
                    #print("neg_graph nodes = ", neg_graph.num_nodes())
                    loss = loss_fcn(batch_pred, pos_graph, neg_graph)
                    #quit()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().item()
                #print("a = ", a)
                epoch_loss /= (step + 1)
                print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
                #epoch_losses.append(epoch_loss) 
            #break
        break
        
    #acc_list.append(acc)
    #acc_list1.append(total_acc1)
#print("Total accuracy cross validation = ", acc_list)


    print('start evaluation')
    model.eval()


    for name, param in model.named_parameters():
        print("name = ", name)
        print("param = ", param)


    emb = {}

    #cnt = 0
    for g in graphs:
        nfeat = g.ndata['m']
        pred = model.inference(g, nfeat, batch_size)
        #print("edges = ", g.number_of_edges())
        #print("nfeat = ", nfeat)
        #print("pred = ", pred)
        #print("pred shape = ", pred.shape)
        emb[g.filename] = pred.tolist()
        #cnt = cnt + 1
        #if (cnt > 3):
        #    break
        
    with open('embeddings_'+str(num_hidden)+'.json', 'w') as f:
        json.dump(emb, f) 
    
    torch.save(model, 'gnn_model_'+str(num_hidden)+'.pt')
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
