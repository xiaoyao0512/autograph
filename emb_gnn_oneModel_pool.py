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
from random import randint 

class ProgramDataset(Dataset):
    def __init__(self, emb, vf_if, loop_count):

        self.emb = emb
        self.vf_if = vf_if
        self.feat = []
        self.labels = []
        self.fn = []
        for fn in self.emb.keys():
            feat_node = self.emb[fn][0]
            #print("feat_node = ", feat_node)
            feat_node.extend(loop_count[fn])
            self.feat.append(torch.FloatTensor([feat_node]))
            self.labels.append(self.vf_if[fn.split('.')[0]][1])
            self.fn.append(fn)


    def __len__(self):
        return len(self.feat)
    def __getitem__(self, idx):
        return self.feat[idx], self.labels[idx], self.fn[idx]
       


def difference(lst1, lst2): 
    return list(set(lst1) - set(lst2))

VF_list = [1, 2, 4, 8, 16]
IF_list = [1, 2, 4, 8, 16]

dims = [8, 16, 32, 64, 128]
input_size = 32
pool = 'avg'
emb = {}
with open('embeddings2_'+str(input_size)+'_'+pool+'_pooling.json') as f:
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
vf_list = []
if_list = []
baseline = {}
num_loops = {
    "s1": 1, "s2": 1, "s6": 1, "s6n": 1,
    "s7": 1, "s7n": 1, "s8": 1, "s8n": 1,
    "s9": 2, "s9n": 2, "s10": 1, "s11": 1,
    "s12": 1, "s12n": 1, "s12nn": 1, "s13": 1, 
    "s15": 2
}
loop_count = {}
for file_VF_IF in files_VF_IF:
    tmp = file_VF_IF.split('.')
    fn = tmp[0]
    fn_tmp = fn.split('_')
    fn_c = fn + '.c'
    if fn_c not in loop_count.keys():
        loop_count[fn_c] = [0, 0]
        for i in range(num_loops[fn_tmp[0]]):
            loop_count[fn_c][i] = int(fn_tmp[i+1])

    tmp = tmp[1].split('-')
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
label_list = []

for f in vf_if.keys():
    label_list.append(vf_if[f][1])
#print("vf_list = ", vf_list)
#print("if_list = ", if_list)

label_freq = Counter(label_list)

print("vf_freq = ", label_freq)

dataset = ProgramDataset(emb, vf_if, loop_count)

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
exec_list4 = []
exec_list5 = []
run_dir = "training_data_default"
save_dir = "training_data_vec"


output_size1 = len(VF_list)
output_size2 = len(IF_list)
output_size = output_size1 * output_size2

files = {}
dim = 32
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


        #model = nn.Sequential(nn.Linear(input_size+2, output_size))#, nn.LogSoftmax(dim=1))
        model = nn.Sequential(nn.Linear(input_size+2, dim),
                              nn.ReLU(),
                              nn.Linear(dim, output_size),
                              #nn.LogSoftmax(dim=1),
                              )

        #print('Initial weights - ', model1[0].weight)
        loss_func = nn.CrossEntropyLoss() 
        #loss_func = nn.NLLLoss()
        #loss_func2 = nn.NLLLoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        #train_data, test_data = torch.utils.data.random_split(dataset, (train_size, test_size))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle = True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle = False)
        full_loader = torch.utils.data.DataLoader(dataset, shuffle = False)
        # Train the model    
        model.train()
        epoch_losses = []
        for epoch in range(NUM_EPOCHES):
            epoch_loss = 0
            pred_list = []
            ground_list = []
            #print("epoch = ", epoch)
            for iter, (bg, label, fn) in enumerate(train_loader):
                #print(bg, label)
                #prediction = model(bg)
                #bg = dgl.add_self_loop(bg)
                #print("bg = ", bg)
                pred = model(bg[0])
                #if (epoch == NUM_EPOCHES - 1):                
                #    print("pred1 = ", pred1, "pred2 = ", pred2, ", label = ", label)
                #print("pred1 = ", pred1, "pred2 = ", pred2, ", label = ", label)
                #loss = loss_func(prediction, label)
                #print("loss = ", loss)
                #print("pred = ", pred, ", label = ", label)
                #quit()
                #print("label = ", label)
                #label = label.tolist()[0]
                #l = torch.LongTensor([VF_list.index(label)])
                loss = loss_func(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()

            #print("a = ", a)
            epoch_loss /= (iter + 1)

            #print('model weights - ', model1[0].weight)
            #print('Epoch {}, loss1 {:.4f} loss2 {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss) 
        print("epoch losses = ", epoch_losses)
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

        model.eval()
        #model2.eval()
        # Convert a list of tuples to two lists
        test_X, test_Y, test_fn = map(list, zip(*test_data))
        acc = 0
        exec1 = 0
        exec2 = 0
        exec3 = 0
        exec4 = 0
        exec5 = 0
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
            x = torch.FloatTensor(x)
            with torch.no_grad():
                pred = model(x)
            #pred2 = model2(x)
            ps = torch.exp(pred)
            probab = list(ps.numpy()[0])
            #print("probab = ", probab)
            pred_label = probab.index(max(probab))
            #pred_label2 = IF_list[np.argmax(pred2.tolist())]    
            #print("predicted vf/if = ", pred_label)
            #print("best vf/if = ", y[0], y[1])
            #print("y = ", y)
            print("fn/predicted/groundtruth = ", fn, pred_label, y)
            if (pred_label == y):
                acc += 1
                pred_list.append(pred_label)
                files[num].append(fn)

            # measure the execution time of a program with a specific VF/IF
            #print("sample vf/if = ", y[0], y[1])
            sampled_y1 = VF_list[int(int(y) / 5)]
            sampled_y2 = IF_list[int(y) % 5]
            VF_pred = VF_list[int(int(pred_label) / 5)]
            IF_pred = IF_list[int(pred_label) % 5]
            VF_rand = VF_list[randint(0, 4)]
            IF_rand = IF_list[randint(0, 4)]
            print(VF_pred, IF_pred)
            print(sampled_y1, sampled_y2)
            t1 = times[fn][sampled_y1][sampled_y2]
            t2 = times[fn][VF_pred][IF_pred]
            t3 = baseline[fn]
            t4 = times[fn][VF_rand][IF_rand]
            print("fn = ", fn, ", t1 = ", t1, ", t2 = ", t2, ", t3 = ", t3, ", t4 = ", t4)
            exec1 += abs(t1 - t2)
            speedup_gt = ((t1 - t2) / t1)
            speedup_base = ((t3 - t2) / t3)
            exec2 += speedup_gt
            speedup_gt_rand = ((t1 - t4) / t1)
            speedup_base_rand = ((t3 - t4) / t3)
            print("speedup compared to ground truth = ", speedup_gt)
            exec3 += speedup_base
            print("speedup compared to baseline O3 = ", speedup_base)
            if ((abs(speedup_gt) > 2) or (abs(speedup_base) > 2)):
                bad.append(fn)
                exec4 += speedup_gt_rand
                exec5 += speedup_base_rand
        acc = acc / (len(test_Y)) * 100
        exec1 = exec1 / len(test_Y)
        exec2 = exec2 / len(test_Y) * 100
        exec3 = exec3 / len(test_Y) * 100
        exec4 = exec4 / len(test_Y) * 100
        exec5 = exec5 / len(test_Y) * 100
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
    exec_list4.append(exec4)
    exec_list5.append(exec5)
    #acc_list1.append(acc1)

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
print("exec4 = ", exec_list4)
print("exec5 = ", exec_list5)
print("bad files = ", bad)
#print("files = ", files)

#print("Total accuracy cross validation = ", acc_list1)

