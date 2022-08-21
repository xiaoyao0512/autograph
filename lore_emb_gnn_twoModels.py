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
#from utility import measure_execution_time
import glob
from collections import Counter
from random import randint 
from sklearn import preprocessing

# s2_1024_2_2_x_b.c 118 171
def preprocess(vf_if, emb):
#for _ in range(1):
    graphs = []
    labels = []
    #VF = []
    #IF = []
    filenames = []
    for f in glob.glob("json_lore/**/*.json", recursive=True):
        fn = f.split('/', 2)[-1][:-5]
        fn_c = fn
        filenames.append(fn_c)
        #print(fn_c)、
        if fn not in vf_if:
            continue
        with open(f) as fh:
            g = nx.readwrite.json_graph.node_link_graph(json.load(fh))
            # calculate the graph features
            g = dgl.from_networkx(g)
            feat = emb[fn]
            #feat.extend(loop_count[fn_c])
            #feat = min_max_scaler.fit_transform(feat)
            g.ndata["m"] = torch.FloatTensor(feat)
            g.filename = fn            
        graphs.append(g)
        #print(fn)
        labels.append(vf_if[fn][1])
        #VF.append(vf_if[fn][1])
        #IF.append(vf_if[fn][2])
    #print("VF max = ", np.max(VF))
    #print("IF max = ", np.max(IF))
    return filenames, graphs, labels
        
class GraphDataset(Dataset):
    def __init__(self, vf_if, emb):        
        self.filenames, self.graphs, self.labels = preprocess(vf_if, emb)
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.filenames[idx], self.graphs[idx], self.labels[idx]
       
class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, filenames):
        super(GCNClassifier, self).__init__()
        #self.mode = mode
        self.conv1 = GraphConv(in_dim, hidden_dim)
        #self.conv2 = GraphConv(hidden_dim, hidden_dim) # graph attention network / gated GNN
        #self.conv3 = GraphConv(hidden_dim, hidden_dim) # graph attention network / gated GNN
        #self.conv1 = GatedGraphConv(in_dim, hidden_dim, 5, 1)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, n_classes)
        self.log = nn.LogSoftmax(dim=1)
        self.filenames = filenames



    def forward(self, g, name):
        # node feature
        #print(g.ndata['w'])
        #h = g.ndata['m'].view(-1,1).float()
        #name = g.filename
        h = g.ndata['m'].float()
        g = dgl.add_self_loop(g)
        #em
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        #h = F.relu(self.conv1(g, h, torch.tensor([0 for i in range(dgl.DGLGraph.number_of_nodes(g))])))
        #h = F.relu(self.conv2(g, h))
        #h = F.relu(self.conv3(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        # introduce loop count in the neural network
        #print("hg = ", hg)
        #quit()
        #print("files = ", self.filenames)
        #print("name = ", name)
        hg = hg.tolist()
        #print(hg)
        #quit()

        linear1 = self.linear1(torch.FloatTensor(hg))
        relu = self.relu(linear1)
        out = self.linear2(relu)
        #out = self.log(out)
        return out
        #return self.log(self.classify(hg))

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    filenames, graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return filenames, batched_graph, torch.tensor(labels)


def difference(lst1, lst2): 
    return list(set(lst1) - set(lst2))




#print("emb keys = ", emb.keys())

VF_list = [1, 2, 4, 8, 16]
IF_list = [1, 2, 4, 8, 16]



f = open('lore_runtimes_none_pragma2.pickle', 'rb')
base_runtimes = pickle.load(f)    
f.close()


f = open('lore_runtimes2.pickle', 'rb')
runtimes = pickle.load(f)    
f.close()

vf_if = {}
times = {}
files_VF_IF = runtimes.keys()
vf_list = []
if_list = []
baseline = {}

for file_VF_IF in files_VF_IF:
    tmp = file_VF_IF.rpartition('.')
    fn = tmp[0]
    fn_c = fn
    tmp = tmp[2].split('-')
    VF = int(tmp[0])
    IF = int(tmp[1])
    label = VF_list.index(VF) * 5 + IF_list.index(IF)
    rt_mean = np.median(runtimes[file_VF_IF])
    base_mean = np.median(base_runtimes[fn])
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

#print("vf_if = ", vf_if)
#print("max VF = ", np.max(vf_list))
#print("min VF = ", np.min(vf_list))
#print("max IF = ", np.max(if_list))
#print("min IF = ", np.min(if_list))

#dims = [8, 16, 32, 64, 128]
dims = [128]
#dim = 8
for dim in dims:
    emb = {}
    with open('loregcc_embeddings2_'+str(dim)+'.json') as f:
        emb = json.load(f)
    
    dataset = GraphDataset(vf_if, emb)
    g_filenames = dataset.filenames
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

    #num_neurons = [8, 16, 32, 64, 128]
    num_neurons = [256]

    print("dim = ", dim)
    acc_list = []
    acc_list1 = []

    exec_list1 = []
    exec_list2 = []
    exec_list3 = []
    exec_list4 = []
    exec_list5 = []
    files = {}
    for num in num_neurons:
        total_acc = 0
        total_acc1 = 0
        files[num] = []
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
            #if (mode == "multifractal"):
            #    inp = 6
            model = GCNClassifier(dim, num, 25, g_filenames)
            #model2 = GCNClassifier(dim, num, len(IF_list))
            #model = GATClassifier(1, num, 1) 
            #model = GATEDClassifier(1, num, 10)
            loss_func = nn.CrossEntropyLoss()
            #loss_func2 = nn.CrossEntropyLoss()
            #loss_func = nn.NLLLoss()
            #loss_func2 = nn.NLLLoss()
            #loss_func1 = nn.HingeEmbeddingLoss()
            #loss_func2 = nn.HingeEmbeddingLoss()
            #loss_func1 = nn.MultiLabelMarginLoss()
            #loss_func2 = nn.MultiLabelMarginLoss()
            #loss_func1 = nn.CTCLoss()
            #loss_func2 = nn.CTCLoss()
            #loss_func1 = nn.BCELoss()
            #loss_func2 = nn.BCELoss()            
            #optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            #train_data, test_data = torch.utils.data.random_split(dataset, (train_size, test_size))
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle = True, collate_fn=collate)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle = False, collate_fn=collate)
            full_loader = torch.utils.data.DataLoader(full_data, shuffle = False, collate_fn=collate)
            # Train the model    
            model.train()
            #model2.train()
            epoch_losses = []
            #epoch_losses2 = []
            for epoch in range(NUM_EPOCHES):
                epoch_loss = 0
                #epoch_loss2 = 0
                #print("epoch = ", epoch)
                for iter, (fn, bg, label) in enumerate(train_loader):
                    #print(iter, label)
                    #prediction = model(bg)
                    #bg = dgl.add_self_loop(bg)
                    #label = label.tolist()[0]
                    #label1 = torch.LongTensor([VF_list.index(label[0])])
                    #label2 = torch.LongTensor([IF_list.index(label[1])])
                    pred = model(bg, fn[0])
                    #pred2 = model2(bg)
                    #if (epoch == 10):                
                    #    print("pred1 = ", pred1, ", label1 = ", label1, ", loss1 = ", loss1)
                    #print("pred1 = ", pred1, "pred2 = ", pred2, ", label = ", label1, label2)
                    #loss = loss_func(prediction, label)
                    #print("loss = ", loss)
                    #print("pred = ", pred, ", label = ", label)

                    loss = loss_func(pred, label)
                    #print("pred1 = ", pred1, ", label1 = ", label1, ", loss1 = ", loss1)
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
                print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
                epoch_losses.append(epoch_loss) 
                #epoch_losses2.append(epoch_loss2) 
            #print("epoch losses = ", epoch_losses)
            #torch.save(model1, 'predictor1_'+str(dim)+'_'+str(num)+'.pt')
            #torch.save(model2, 'predictor2_'+str(dim)+'_'+str(num)+'.pt')
            # Evaluate the testing dataset
            #print("start evaluating the testing dataset")
            model.eval()
            #model2.eval()
            # Convert a list of tuples to two lists
            full_fn, full_X, full_Y = map(list, zip(*full_data))
            acc = 0
            acc1 = 0
            exec1 = 0
            exec2 = 0
            exec3 = 0
            exec4 = 0
            exec5 = 0
            pred_list = []
            test_list = []
            bad = []
            for idx in range(len(full_X)):
                x = full_X[idx]
                y = full_Y[idx]
                test_list.append(int(y))
                fn = x.filename
                print("f = ", fn)
                #print("batch size = ", x)
                #print("#nodes = ", test_bg.batch_num_nodes())
                #test_bg = dgl.add_self_loop(test_bg)
                #y = torch.tensor(y).float().view(-1, 1)
                #print('y00 = ', int(y[0,0]), ', y10 = ', int(y[1,0]))
                with torch.no_grad():
                    pred = model(x, full_fn[idx])
                #pred2 = model2(x)
                ps = torch.exp(pred)
                probab = list(ps.numpy()[0])
                #print("probab = ", probab)
                pred_label = probab.index(max(probab))
                #pred2 = model2(x)
                #sampled_y1 = VF_list[np.argmax(pred1.tolist())]
                #sampled_y2 = IF_list[np.argmax(pred2.tolist())]
                #print("predicted vf/if = ", sampled_y1, sampled_y2)
                #print("best vf/if = ", int(y[0,0]), int(y[1,0]))
                #print("y = ", y)
                #print("pred1 = ", sampled_y1, " , pred2 = ", sampled_y2)
                #print("pred2 = ", sampled_y2)
                #print(pred, y)
                #acc += abs(pred - y)
                #probs_Y = torch.softmax(pred1, 1)
                #print("In fold ", kf, ', y = ', y, ', probs_Y = ', probs_Y)
                #print("In fold ", kf, ', len = ', len(test_Y))
                #print("label = ", y)
                #probs_Y = probs_Y[0]
                #sampled_Y = torch.multinomial(probs_Y, 1)
                #print("probs_Y = ", probs_Y, " sampled_Y = ", sampled_Y)
                #argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
                #print("In fold ", kf, ', sampled = ', sampled_Y, ', argmax = ', argmax_Y)
                #pred_label = np.argmax(pred.tolist())
                #print("fn/predicted/groundtruth = ", fn, pred_label, y)
                print("label = ", y, (int(pred_label/5), int(pred_label%5)))
                if (pred_label == y):
                    acc += 1
                    pred_list.append(pred_label)
                    #files[num].append(fn)

                # measure the execution time of a program with a specific VF/IF
                #t1 = measure_execution_time(save_dir, run_dir, fn, int(y[0,0]), int(y[1,0]) , 10)
                #t2 = measure_execution_time(save_dir, run_dir, fn, sampled_y1, sampled_y2, 10)
                sampled_y1 = VF_list[int(int(y) / 5)]
                sampled_y2 = IF_list[int(y) % 5]
                VF_pred = VF_list[int(int(pred_label) / 5)]
                IF_pred = IF_list[int(pred_label) % 5]
                VF_rand = VF_list[randint(0, 4)]
                IF_rand = IF_list[randint(0, 4)]
                #print(VF_pred, IF_pred)
                #print(sampled_y1, sampled_y2)
                t1 = times[fn][sampled_y1][sampled_y2]
                t2 = times[fn][VF_pred][IF_pred]
                t3 = baseline[fn]
                t4 = times[fn][VF_rand][IF_rand]
                print("t1 = ", t1, ", t2 = ", t2, ", t3 = ", t3, ", t4 = ", t4)
                exec1 += abs(t1 - t2)
                speedup_gt = ((t1 - t2) / t1)
                speedup_base = ((t3 - t2) / t3)
                speedup_gt_rand = ((t1 - t4) / t1)
                speedup_base_rand = ((t3 - t4) / t3)
                exec2 += speedup_gt
                print("speedup compared to ground truth = ", speedup_gt)
                exec3 += speedup_base
                print("speedup compared to baseline O3 = ", speedup_base)
                exec4 += speedup_gt_rand
                exec5 += speedup_base_rand
                if ((abs(speedup_gt) > 2) or (abs(speedup_base) > 2)):
                    bad.append(fn)

            full_fn, full_X, full_Y = map(list, zip(*full_data))
            emb = {}
            for idx in range(len(full_X)):
                g = full_X[idx]
                name = full_fn[idx]
                    
                h = g.ndata['m'].float()
                g = dgl.add_self_loop(g)
                h = F.relu(model.conv1(g, h))
                #h = F.relu(model.conv1(g, h, torch.tensor([0 for i in range(dgl.DGLGraph.number_of_nodes(g))])))
                g.ndata['h'] = h
                hg = dgl.mean_nodes(g, 'h')
                #hg = [hg[0].tolist() + model.loops[model.filenames.index(aname)].tolist()]  

                emb[name] = hg.tolist()
            #with open('embeddings2_graphsage_gcn'+str(dim)+'.json', 'w') as f:
            with open('lore_embeddings2_graphsage_gcn'+str(dim)+'.json', 'w') as f:
                json.dump(emb, f) 
            
            acc = acc / (len(full_Y)) * 100
            exec1 = exec1 / len(full_Y)
            exec2 = exec2 / len(full_Y) * 100
            exec3 = exec3 / len(full_Y) * 100
            exec4 = exec4 / len(full_Y) * 100
            exec5 = exec5 / len(full_Y) * 100
            #acc1 = acc1 / len(test_Y) * 100
            #print("In fold ", kf, ', Accuracy of sampled predictions on the test set: ', acc)
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
        exec_list4.append(exec4)
        exec_list5.append(exec5)
        #acc_list1.append(acc1)
    #pred_freq = Counter(pred_list)
    #test_freq = Counter(test_list)
    #acc_class = {}
    #for dim in test_freq:
    #    acc_per_class = pred_freq[dim] / test_freq[dim]
    #    acc_class[dim] = acc_per_class

    #print("pred_freq = ", pred_freq)
    #print("test_freq = ", test_freq)
    #print("acc_per_class = ", acc_class)
    print("accuracy_list = ", acc_list)
    print("exec1 = ", exec_list1)
    print("exec2 = ", exec_list2)
    print("exec3 = ", exec_list3)
    print("exec4 = ", exec_list4)
    print("exec5 = ", exec_list5)
    #print("bad files = ", bad)
    #print("files = ", files)
#print("Total accuracy cross validation = ", acc_list1)

