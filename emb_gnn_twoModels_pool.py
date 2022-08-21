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
from utility import measure_execution_time
from collections import Counter

class ProgramDataset(Dataset):
    def __init__(self, emb, vf_if):

        self.emb = emb
        self.vf_if = vf_if
        self.feat = []
        self.labels = []
        self.fn = []
        for fn in self.emb.keys():
            self.feat.append(torch.FloatTensor(self.emb[fn]))
            self.labels.append((self.vf_if[fn.split('.')[0]][1], self.vf_if[fn.split('.')[0]][2]))
            self.fn.append(fn)


    def __len__(self):
        return len(self.feat)
    def __getitem__(self, idx):
        return self.feat[idx], self.labels[idx], self.fn[idx]
       


def difference(lst1, lst2): 
    return list(set(lst1) - set(lst2))



dims = [8, 16, 32, 64, 128]
input_size = 32
pool = 'max'
emb = {}
with open('embeddings_'+str(input_size)+'_'+pool+'_pooling.json') as f:
    emb = json.load(f)

f = open('runtimes.pickle', 'rb')
runtimes = pickle.load(f)    
f.close()

f = open('runtimes_none_pragma.pickle', 'rb')
base_runtimes = pickle.load(f)    
f.close()




vf_if = {}
times = {}
files_VF_IF = runtimes.keys()

baseline = {}
for file_VF_IF in files_VF_IF:
    tmp = file_VF_IF.split('.')
    fn = tmp[0]
    tmp = tmp[1].split('-')
    VF = int(tmp[0])
    IF = int(tmp[1])
    fn_c = fn + '.c'
    rt_mean = np.mean(runtimes[file_VF_IF])
    base_mean = np.mean(base_runtimes[fn])
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

vf_list = []
if_list = []
for f in vf_if.keys():
    vf_list.append(vf_if[f][1])
    if_list.append(vf_if[f][2])
#print("vf_list = ", vf_list)
#print("if_list = ", if_list)

vf_freq = Counter(vf_list)
if_freq = Counter(if_list)

print("vf_freq = ", vf_freq)
print("if_freq = ", if_freq)


dataset = ProgramDataset(emb, vf_if)

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
num_neurons = [32]
acc_list = []
exec_list1 = []
exec_list2 = []
exec_list3 = []
run_dir = "training_data_default"
save_dir = "training_data_vec"

VF_list = [1, 2, 4, 8, 16]
IF_list = [1, 2, 4, 8, 16]
output_size1 = len(VF_list)
output_size2 = len(IF_list)

files = {}
for num in num_neurons:
    vf = []
    _if = []
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

        #if (mode == "multifractal"):
        #    inp = 6
        #input_size = 2*dim if pool == 'sort' else dim
        #hidden_size = num


        model1 = nn.Sequential(nn.Linear(input_size, output_size1))
        model2 = nn.Sequential(
                              nn.Linear(input_size, output_size2))


        #print('Initial weights - ', model1[0].weight)
        loss_func1 = nn.CrossEntropyLoss() 
        loss_func2 = nn.CrossEntropyLoss() 
        #loss_func1 = nn.NLLLoss()
        #loss_func2 = nn.NLLLoss()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.0005)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.0005)
        #optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
        #train_data, test_data = torch.utils.data.random_split(dataset, (train_size, test_size))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle = True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle = False)
        full_loader = torch.utils.data.DataLoader(dataset, shuffle = False)
        # Train the model    
        model1.train()
        model2.train()
        epoch_losses1 = []
        epoch_losses2 = []
        for epoch in range(NUM_EPOCHES):
            epoch_loss1 = 0
            epoch_loss2 = 0
            pred_vf_list = []
            ground_vf_list = []
            pred_if_list = []
            ground_if_list = []
            #print("epoch = ", epoch)
            for iter, (bg, label, fn) in enumerate(train_loader):
                #print(bg, label)
                #prediction = model(bg)
                #bg = dgl.add_self_loop(bg)
                #print("bg = ", bg)
                pred1 = model1(bg[0])
                pred2 = model2(bg[0])
                #if (epoch == NUM_EPOCHES - 1):                
                #    print("pred1 = ", pred1, "pred2 = ", pred2, ", label = ", label)
                #print("pred1 = ", pred1, "pred2 = ", pred2, ", label = ", label)
                #loss = loss_func(prediction, label)
                #print("loss = ", loss)
                #print("pred = ", prediction, ", label = ", label)
                #quit()
                #print(label)
                #label = label.tolist()[0]
                label1 = torch.LongTensor([VF_list.index(label[0])])
                vf.append(int(label[0]))
                _if.append(int(label[1]))
                pred_vf_list.append(VF_list[np.argmax(pred1.tolist())])
                ground_vf_list.append(int(label[0]))
                pred_if_list.append(IF_list[np.argmax(pred2.tolist())])
                ground_if_list.append(int(label[1]))
                label2 = torch.LongTensor([IF_list.index(label[1])])
                #print(pred1, label1)
                loss1 = loss_func1(pred1, label1)
                loss2 = loss_func2(pred2, label2)
                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()
                optimizer2.zero_grad()
                loss2.backward()
                optimizer2.step()
                epoch_loss1 += loss1.detach().item()
                epoch_loss2 += loss2.detach().item()
            #print("a = ", a)
            epoch_loss1 /= (iter + 1)
            print("In epoch ", epoch, " vf pred freq = ", Counter(pred_vf_list))
            print("vf ground true freq = ", Counter(ground_vf_list))
            print("In epoch ", epoch, " if pred freq = ", Counter(pred_if_list))
            print("if ground true freq = ", Counter(ground_if_list))
            epoch_loss2 /= (iter + 1)
            #print('model weights - ', model1[0].weight)
            #print('Epoch {}, loss1 {:.4f} loss2 {:.4f}'.format(epoch, epoch_loss1, epoch_loss2))
            epoch_losses1.append(epoch_loss1) 
            epoch_losses2.append(epoch_loss2) 
        '''
        print("model1")
        for name, param in model1.named_parameters():
            print("name = ", name)
            print("param = ", param)      
        print("model2")
        for name, param in model2.named_parameters():
            print("name = ", name)
            print("param = ", param)  
        '''
        # Evaluate the testing dataset
        #print("start evaluating the testing dataset")
        vf_train_freq = Counter(vf)
        print("During training, VF freq = ", vf_train_freq)
        if_train_freq = Counter(_if)
        print("During training, IF freq = ", if_train_freq)
        model1.eval()
        #model2.eval()
        # Convert a list of tuples to two lists
        test_X, test_Y, test_fn = map(list, zip(*test_data))
        acc = 0
        exec1 = 0
        exec2 = 0
        exec3 = 0
        #continue
        for idx in range(len(test_X)):
            x = test_X[idx]
            y = test_Y[idx]
            fn = test_fn[idx]
            #print("batch size = ", x)
            #print("#nodes = ", test_bg.batch_num_nodes())
            #test_bg = dgl.add_self_loop(test_bg)
            #print('y = ', y)
            #y = torch.tensor(y).float().view(-1, 1)
            #print('y = ', y)
            #print('y00 = ', y[1,0])
            x = torch.FloatTensor(x)
            pred1 = model1(x)
            #pred2 = model2(x)
            pred_label1 = VF_list[np.argmax(pred1.tolist())]
            #pred_label2 = IF_list[np.argmax(pred2.tolist())]    
            print("predicted vf/if = ", pred_label1)
            #print("best vf/if = ", y[0], y[1])
            #print("y = ", y)
            #if (pred_label1 == y[0] and pred_label2 == y[1]):
            #    acc += 1
            #    files[num].append(fn)

            # measure the execution time of a program with a specific VF/IF
            #print("sample vf/if = ", y[0], y[1])
            #t1 = times[fn][y[0]][y[1]]
            #t2 = times[fn][pred_label1][pred_label2]
            #t3 = baseline[fn]
            #print("fn = ", fn, ", t1 = ", t1, ", t2 = ", t2, ", t3 = ", t3)
            #exec1 += abs(t1 - t2)
            #slowdown = ((t2 - t1) / t1)
            #speedup = ((t3 - t2) / t3)
            #exec2 += slowdown
            #print("slowdown = ", slowdown)
            #exec3 += speedup
            #print("speedup = ", speedup)
        
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
print(num_neurons)
print("Total accuracy cross validation = ", acc_list)
print("exec1 = ", exec_list1)
print("exec2 = ", exec_list2)
print("exec3 = ", exec_list3)
#print("files = ", files)

#print("Total accuracy cross validation = ", acc_list1)

