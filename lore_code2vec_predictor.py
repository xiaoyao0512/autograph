from extractor_c import CExtractor
from config import Config
from my_model import Code2VecModel
from path_context_reader import EstimatorAction
from lore_utility import get_snapshot_from_code, get_vectorized_codes
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
from random import randint

def preprocess(vf_if):
    if exists ("lore_features.json"):
        with open("lore_features.json") as f:
            features = json.load(f)
        return features["feat"], features["labels"], features["files"]   
    else: 
        CLANG_PATH = "/usr/lib/llvm-6.0/lib/libclang.so.1"
        MAX_LEAF_NODES = 320
        new_rundir = "lore-src"
        ''' Parse the training data. '''
        orig_train_files = [os.path.join(root, name)
            for root, dirs, files in os.walk(new_rundir)
            for name in files
            if name.endswith(".c") and not name.startswith('header.c') 
            and not name.startswith('aux_AST_embedding_code.c')]
        # copy testfiles
        new_testfiles = list(orig_train_files)
        #print(orig_train_files)
        # parse the code to detect loops and inject commented pragmas.  
        #loops_idxs_in_orig,pragmas_idxs,const_new_codes,num_loops,const_orig_codes = get_vectorized_codes(orig_train_files,new_testfiles)
        # to operate only on files that have for loops.
        #new_testfiles = list(pragmas_idxs.keys())

        '''Config the AST tree parser.'''
        config = Config(set_defaults=True, load_from_args=False, verify=True)
        code2vec = Code2VecModel(config)
        path_extractor = CExtractor(config,clang_path=CLANG_PATH,max_leaves=MAX_LEAF_NODES)
        train_input_reader = code2vec._create_data_reader(estimator_action=EstimatorAction.Train)

        input_full_path_filename = os.path.join('aux_AST_embedding_code.c')

        #print(const_orig_codes.keys())
        files = []
        feat = []
        labels = []
        for current_filename in glob.glob(new_rundir + "/**/*.c", recursive=True):
            if ((current_filename == new_rundir + "/header.c") or (current_filename == new_rundir + "/aux_AST_embedding_code.c")):
                continue
            file_dir = current_filename.split('/', 1)[-1].rpartition('/')[0]
            #print(file_dir)
            
            if (file_dir not in vf_if):
                continue
            files.append(file_dir)
            labels.append(vf_if[file_dir][1])
            f = open(current_filename)
            code = f.readlines()
            f.close()
            #print("code = ", code)
            #print("input_full_path_filename = ", input_full_path_filename)
            code = get_snapshot_from_code(code)
            #print("new code = ", code)
            loop_file = open(input_full_path_filename,'w')
            loop_file.write(''.join(code))
            loop_file.close()
            try:
                train_lines, hash_to_string_dict = path_extractor.extract_paths(input_full_path_filename)
            except:
                print('Could not parse file',current_filename, '. Try removing it.')
                raise 
            dataset = train_input_reader.process_and_iterate_input_from_data_lines(train_lines)
            obs = []
            tensors = list(dataset)[0][0]
            
            for tensor in tensors:
                #with tf.compat.v1.Session() as sess: 
                #    sess.run(tf.compat.v1.tables_initializer())
                obs.append(tf.squeeze(tensor).numpy().tolist())
            obs = list(np.concatenate(obs).flat)         
            assert len(obs) == 800, "ERROR!"
            feat.append([obs])
        
        features = {}
        features['feat'] = feat
        features['labels'] = labels
        features['files'] = files
        with open('lore_features.json', 'w') as f:
            json.dump(features, f) 
        return feat, labels, files
        
class GraphDataset(Dataset):
    def __init__(self, vf_if):        
        self.feat, self.labels, self.fn = preprocess(vf_if)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.feat[idx], self.labels[idx], self.fn[idx]

def difference(lst1, lst2): 
    return list(set(lst1) - set(lst2))

f = open('lore_runtimes2.pickle', 'rb')
runtimes = pickle.load(f)    
f.close()

f = open('lore_runtimes_none_pragma2.pickle', 'rb')
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
    #print(file_VF_IF)
    tmp = file_VF_IF.rpartition('.')
    fn = tmp[0]
    tmp = tmp[2].split('-')
    VF = int(tmp[0])
    IF = int(tmp[1])
    label = VF_list.index(VF) * 5 + IF_list.index(IF)
    fn_c = fn
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
num_neurons = [16]



print(num_instances, train_size, test_size)
# Layer details for the neural network
input_size = 800
#hidden_sizes = [128, 64]
output_size1 = len(VF_list)
output_size2 = len(IF_list)
output_size = output_size1 * output_size2


files = {}
acc_list = []
exec_list1 = []
exec_list2 = []
exec_list3 = []
for hidden_size in num_neurons:
    total_acc = 0
    total_acc1 = 0
    files[hidden_size] = []
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
        model = nn.Sequential(nn.Linear(input_size, output_size), nn.LogSoftmax(dim=1))
        #model2 = nn.Sequential(
        #                      nn.Linear(input_size, output_size2))
        loss_func = nn.NLLLoss()
        #loss_func = nn.CrossEntropyLoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        #optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
        #train_data, test_data = torch.utils.data.random_split(dataset, (train_size, test_size))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle = True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle = False)
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
                bg = torch.FloatTensor(bg)
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
        #print("epoch losses = ", epoch_losses)
        model.eval()
        #model2.eval()
        # Convert a list of tuples to two lists
        full_X, full_Y, full_fn = map(list, zip(*full_data))
        acc = 0
        exec1 = 0
        exec2 = 0
        exec3 = 0
        #continue
        pred_list = []
        test_list = []
        bad = []
        for idx in range(len(full_X)):
            x = full_X[idx]
            y = full_Y[idx]
            test_list.append(int(y))
            fn = full_fn[idx]
            print("f = ", fn)
            #print("#nodes = ", test_bg.batch_num_nodes())
            #test_bg = dgl.add_self_loop(test_bg)
            #print('y = ', y)
            #y = torch.tensor(y).float().view(-1, 1)
            #print('y = ', y)
            #print('y00 = ', y[1,0])
            x = torch.FloatTensor(x)
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
            print("label = ", y, (int(pred_label/5), int(pred_label%5)))
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
            VF_rand = VF_list[randint(0, 4)]
            IF_rand = IF_list[randint(0, 4)]
            #print(VF_pred, IF_pred)
            #print(sampled_y1, sampled_y2)
            t1 = times[fn][sampled_y1][sampled_y2] # ground truth
            t2 = times[fn][VF_pred][IF_pred] # prediction
            t3 = baseline[fn] # baseline
            t4 = times[fn][VF_rand][IF_rand]
            print( "t1 = ", t1, ", t2 = ", t2, ", t3 = ", t3, ", t4 = ", t4)
            exec1 += abs(t1 - t2)
            speedup_gt = ((t1 - t2) / t1)
            speedup_base = ((t3 - t2) / t3)
            print("reward = ", speedup_base)
            exec2 += speedup_gt
            print("speedup compared to ground truth = ", speedup_gt)
            exec3 += speedup_base
            print("speedup compared to baseline O3 = ", speedup_base)
                
        
        acc = acc / (len(full_Y)) * 100
        exec1 = exec1 / len(full_Y)
        exec2 = exec2 / len(full_Y) * 100
        exec3 = exec3 / len(full_Y) * 100
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
    #acc_list1.append(acc1)
#print(num_neurons)
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
print("exec3 = ", exec_list2)
print("exec3 = ", exec_list3)
#print("bad files = ", bad)
#print("files = ", files)

