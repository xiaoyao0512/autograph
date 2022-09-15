'''
Copyright (c) 2019, Ameer Haj Ali (UC Berkeley), and Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import os
import re
import pickle
import subprocess
import copy
import networkx as nx
import json
import dgl, torch
from os.path import exists
from sklearn.decomposition import PCA
from torch import nn
import csv
import glob
import numpy as np
from node import node_dimension
from sklearn import preprocessing

SAVE_DIR = "training_data_vec"

#MAX_LEAF_NODES = os.environ['MAX_LEAF_NODES']
#TEST_SHELL_COMMAND_TIMEOUT = os.environ['TEST_SHELL_COMMAND_TIMEOUT']
#the maximum number of leafs in the LLVM abstract sytnax tree
###MAX_LEAF_NODES = os.environ['MAX_LEAF_NODES']
###TEST_SHELL_COMMAND_TIMEOUT = os.environ['TEST_SHELL_COMMAND_TIMEOUT']
# pragma line injected for each loop
pragma_line = '#pragma clang loop vectorize_width({0}) interleave_count({1})\n'

def read_feature_from_files(embed_path):
    assert os.path.isfile(embed_path)
    feat = {}
    cnt = 0
    feat_matrix = []
    with open(embed_path) as file_in:
        for line in file_in:
            #print(line)
            cnt += 1
            if cnt == 1:
                continue 
            line = line.rstrip()
            embed = line.split(' ')
            #print("embed = ", embed)
            node = embed.pop(0)
            feat[node] = list(map(float, embed))
            
    #print("feat = ", feat)
    for k in sorted(feat.keys()):
        #print("k = ", k, " d = ", feat[k])
        feat_matrix.append(feat[k])
    return feat_matrix

def deep_walk_preprocess(vf_if):
    if exists ("deepwalk.json"):
        with open("deepwalk.json") as f:
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
        return features["files"], graphs, features["labels"]  
  
    else: 

        files = []
        graphs = []
        labels = []
        feats = {}
        adjlist_dir = "training_adjlist"
        min_max_scaler = preprocessing.MinMaxScaler()
        for f in glob.glob("json-small/*.json"):
            fn = f.split('/')[-1].split('.')[0]
            fn_c = fn + '.c'
            full_path = adjlist_dir+'/'+fn_c+'.adjlist'
            embed_path = adjlist_dir+'/'+fn_c+'.embeddings'
            files.append(fn_c)
            with open(f) as fh:
                g = nx.readwrite.json_graph.node_link_graph(json.load(fh))
                # calculate the graph features
                #print(g.nodes)
            #os.chdir("deepwalk")
            #os.system("deepwalk --input "+full_path+" --output "+embed_path+" --representation-size 8")
            feat = read_feature_from_files(embed_path)
            feat = min_max_scaler.fit_transform(feat)
            feats[fn_c] = feat.tolist()
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
        with open('deepwalk.json', 'w') as f:
            json.dump(features, f) 
        return files, graphs, labels


def mfa_preprocess(vf_if):
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
        return features["files"], graphs, features["labels"] 
    else: 

        files = []
        graphs = []
        labels = []
        feats = {}
        min_max_scaler = preprocessing.MinMaxScaler()
        for f in glob.glob("json-small/*.json"):
            fn = f.split('/')[-1].split('.')[0]
            fn_c = fn + '.c'
            files.append(fn_c)
            with open(f) as fh:
                g = nx.readwrite.json_graph.node_link_graph(json.load(fh))
                # calculate the graph features
                #print(g.nodes)
            feat = node_dimension(g)  
            feat = min_max_scaler.fit_transform(feat)
            feats[fn_c] = feat.tolist()      
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
        return files, graphs, labels


def preprocess(run_dir, filepath, vf_len, if_len):

    graphs = {}
    emb = {}
    emb_temp = {}

    with open(filepath) as f:
        emb_temp = json.load(f)

    for f in glob.glob("json-small/*.json"):
        fn = f.split('/')[-1].split('.')[0]
        fn_c = fn + '.c'
        with open(f) as fh:
            g = nx.readwrite.json_graph.node_link_graph(json.load(fh))
            # calculate the graph features
            #print(g.nodes)
            g = dgl.from_networkx(g)
        graphs[fn_c] = {}
        graphs[fn_c][0] = [[None for _ in range(if_len)] for _ in range(vf_len)]
        graphs[fn_c][0][0][0] = g

    
        # check if a file exists because I deleted some programs that have over 10,000 nodes
        # just to check whether or not this prototype could run.
        #file_exists = exists(run_dir + '/' + fn)
        #if not file_exists:
        #    continue
        loops = find_loops(run_dir, fn_c)
        emb[fn_c] = {}
        for i in range(loops):
            emb[fn_c][i] = [[None for _ in range(if_len)] for _ in range(vf_len)]
            emb[fn_c][i][0][0] = emb_temp[fn]

    f = open('runtimes_icx7_omp_orig.pickle', 'rb')
    runtimes = pickle.load(f)    
    f.close()

    vf_if = {}
    for fn in runtimes.keys(): 
        fn_c = fn + '.c'
        full_path = os.path.join(rundir, fn_c)
        rt_mean = np.median(runtimes[fn])
        #print("filename = ", fn)
        #print("VF = ", VF)
        #print("IF = ", IF)
        #print("mean = ", rt_mean)
        if full_path not in vf_if.keys():
            vf_if[full_path] = rt_mean
        else:
            rt_mean_pre = vf_if[full_path]
            if rt_mean < rt_mean_pre:
                vf_if[full_path] = rt_mean
        
    return graphs, emb, vf_if

def measure_execution_time(save_dir, run_dir, file_name, VF, IF, repeat):

    insert_pragma(save_dir, run_dir, file_name, VF, IF, 1)

    os.system("clang -O3 " + save_dir+'/'+file_name + ' ' + run_dir+'/header.c')
    time_slots = []
    for i in range(repeat):
        output = subprocess.check_output("./a.out", shell=True)
        time = float(output.split()[3])
        time_slots.append(time)

    average = sum(time_slots) / len(time_slots)

    return average

def delete_zeros(name, mylist):

    new_list = []    
    if name == "filename":
        for num in mylist:
            if num:
                new_list.append(int(num))
    elif name == "emb":
        for row in mylist:
            if row[0] != 0.0:
                new_list.append(row.tolist()) 
    return new_list

def dict_from_csv(filename):
    bag = {}
    with open(filename, 'r') as file:
        csvreader = csv.reader(file, delimiter="\t")
        header = next(csvreader)
        #print("header = ", header)
        for elem in header:
            bag[elem] = []
        for row in csvreader:
            for i in range(len(row)):
                #print("i = ", i)
                #print("row = ", row)
                val = row[i]
                key = header[i]
                bag[key].append(val)
    return bag



def vocal2(nodes):
    #print(nodes)
    #quit()
    feat_matrix = []
    bag_of_words = dict_from_csv("vocab/programl.csv")
    dim = 6 #int(pow(2230, 0.25))
    emb = nn.Embedding(2230, dim)
    for (node_id, node_dict) in nodes:
        #print(node_id)
        #print(node_dict)
        # check if a keyword is in the vocabulary
        val = 0
        if (node_dict['text'] in bag_of_words['text']):
            text_idx = bag_of_words['text'].index(node_dict['text'])
            val = text_idx + 1
            '''            
            if (text_idx != 0):
                val = float(bag_of_words['cumulative_node_frequency'][text_idx]) - float(bag_of_words['cumulative_node_frequency'][text_idx-1])
            else:
                val = float(bag_of_words['cumulative_node_frequency'][text_idx])
            '''
        #print(node_dict)
        #print("dict type = ", node_dict['type'])
        #idx_emb = [val]
        #type_emb = [node_dict['type']]
        idx_emb = emb(torch.IntTensor([val])).flatten().tolist()
        type_emb = emb(torch.IntTensor([node_dict['type']])).flatten().tolist()
        
        #feat = []
        #feat.append(idx)
        #feat.append(type)
        #print(feat)
        #quit()
        feat_matrix.append(feat)
    #print("feat_matrix = ", feat_matrix)
    return feat_matrix#normalize(feat_matrix)

def vocal(nodes, degree):
    #print(nodes)
    feat_matrix = []
    bag_of_words = dict_from_csv("vocab/programl.csv")
    dim1 = 6 #int(pow(2230, 0.25))
    #dim2 = 4
    emb1 = nn.Embedding(2230, dim1)
    emb2 = nn.Embedding(2230, dim1)
    #emb3 = nn.Embedding(300, dim2)
    for (node_id, node_dict) in nodes:
        #print(node_id)
        #print(node_dict)
        # check if a keyword is in the vocabulary
        val = 0
        if (node_dict['text'] in bag_of_words['text']):
            text_idx = bag_of_words['text'].index(node_dict['text'])
            val = text_idx + 1
            '''            
            if (text_idx != 0):
                val = float(bag_of_words['cumulative_node_frequency'][text_idx]) - float(bag_of_words['cumulative_node_frequency'][text_idx-1])
            else:
                val = float(bag_of_words['cumulative_node_frequency'][text_idx])
            '''
        #print(node_dict)
        #print("dict type = ", node_dict['type'])
        #idx_emb = [val]
        #type_emb = [node_dict['type']]
        idx_emb = emb1(torch.IntTensor([val])).flatten().tolist()
        type_emb = emb2(torch.IntTensor([node_dict['type']])).flatten().tolist()
        #deg_emb = emb3(torch.IntTensor([degree[node_id]])).flatten().tolist()
        #idx_emb = [val]
        #type_emb = [node_dict['type']]
        feat = []
        feat.extend(idx_emb)
        feat.extend(type_emb)
        #feat.extend(deg_emb)
        #print(feat)
        #quit()
        feat_matrix.append(feat)
    #print("feat_matrix = ", feat_matrix)
    return feat_matrix#normalize(feat_matrix)




def pca_analysis(feature, n_comp):
    #feat = torch.FloatTensor(feature)
    pca = PCA(n_components=n_comp, random_state=2022)
    pca.fit(feature)
    #return pca.explained_variance_ratio_ * 100
    pca_8 = pca.transform(feature)
    return pca_8.tolist()
    #print("pca ratio = ", pca.explained_variance_ratio_ * 100)

def str2intlist(string):
    characters = list(string)
    numbers = []
    #print(characters)
    for ch in characters:
        numbers.append(ord(ch))
    return numbers

def intlist2str(numbers):
    chs = []
    for num in numbers:
        chs.append(chr(num))
    #print(chs)
    return ''.join(chs)

def init_runtimes_dict(files,num_loops,VF_len):
    '''Used to initialize runtimes dict that stores
    runtimes for all the files and loops for
    different VF/IF during training to save time.'''

    f = open('runtimes_omp_icx_8classes.pickle', 'rb')
    runtimes = pickle.load(f)
    f.close()

    times = {}
    files_VF_IF = runtimes.keys()
    vf_list = [0,1,2,3,4,5,6]

    for file_VF_IF in files_VF_IF:
        fn_c = file_VF_IF
        if fn_c not in num_loops:
            continue
        for i in range(num_loops[fn_c]):
            times[fn_c] = {}
            times[fn_c][i] = [np.mean(runtimes[file_VF_IF][VF]) for VF in range(VF_len)]
    return times

def find_loops(run_dir, fname):
    num = 0
    with open(run_dir+'/'+fname) as fr:
        for line in fr:
            #print(line.rstrip())
            if re.match(r'^\s*for\s*\(',line) or re.match(r'^\s*while\s*\(',line):
                num = num + 1
    return num

def load_observations(run_dir, filepath, vf_len):
    emb_temp = {}
    emb = {}
    with open(filepath) as f:
        emb_temp = json.load(f)
    for fn in emb_temp.keys():
        # check if a file exists because I deleted some programs that have over 10,000 nodes
        # just to check whether or not this prototype could run.
        file_exists = exists(run_dir + '/' + fn)
        if not file_exists:
            continue
        loops = find_loops(run_dir, fn)
        emb[fn] = {}
        for i in range(loops):
            emb[fn][i] = [None for _ in range(vf_len)]
            emb[fn][i][0] = emb_temp[fn]
            #print("fn = ", fn) 
            #print("emb_temp[fn] = ", emb_temp[fn])       
    return emb

def load_observations_pooling(run_dir, filepath):
    emb_temp = {}
    emb = {}
    with open(filepath) as f:
        emb_temp = json.load(f)
    
    #print("emb_temp = ", emb_temp)
    # emb is a dict:
    #   key - pragma_idx (loop idx)
    #   value - list of different VFs and IFs of features in a graph
    #print("11111")
    for fn in emb_temp.keys():
        loops = find_loops(run_dir, fn)
        emb[fn] = {}
        for i in range(loops):
            emb[fn][i] = []
            emb[fn][i].extend(emb_temp[fn][0])
            #print("fn = ", fn) 
            #print("emb_temp[fn] = ", emb_temp[fn])       
    return emb

def load_observations_dict(run_dir, filepath):
    emb_temp = {}
    emb = {}
    with open(filepath) as f:
        emb_temp = json.load(f)
    files = list(emb_temp.keys())

    for fn in files:
        loops = 1#find_loops(run_dir, fn)
        fn_root = fn#run_dir + '/' + fn
        emb[fn_root] = {}
        for i in range(loops):
            emb[fn_root][i] = np.array(emb_temp[fn_root], dtype=np.float32)
            # emb[fn_root][i] = np.reshape(np.array(emb_temp[fn_root], dtype=np.float32),(256,))
            emb[fn_root][i] = np.reshape(emb[fn_root][i],(256,))

    return emb
        
def load_graphs(exec_time, vf_len, if_len):

    graphs = {}
    for fn in exec_time.keys():
        graphs[fn] = {}
        graphs[fn][0] = [None for _ in range(vf_len)]
        graphs[fn][0][0] = dgl.from_networkx(nx.from_dict_of_dicts(exec_time[fn][2]))
    return graphs
    
def get_bruteforce_runtimes(rundir,files,vec_action_meaning):
    ''' get all runtimes with bruteforce seach and -O3 
    assuming a single loop per file!'''
    opt_runtimes = {}
    opt_factors = {}
    all_program_runtimes = {}
    one_program_runtimes = [0 for vf in range(len(vec_action_meaning))]
    full_path_header = os.path.join(rundir,'header.c')
    for filename in files:
        opt_runtime = 1e+9
        opt_factor = (1,1)
        for i,VF in enumerate(vec_action_meaning):
            for j,IF in enumerate(interleave_action_meaning):
                rm_cmd = 'rm ' + filename[:-1]+'o '
                if os.path.exists(filename[:-1]+'o'):
                    os.system(rm_cmd)
                cmd1 = 'timeout 4s ' + os.environ['CLANG_BIN_PATH'] + ' -O3 -lm '+full_path_header
                +' ' +filename+' -Rpass=loop-vectorize -mllvm -force-vector-width='
                +str(VF)+' -mllvm -force-vector-interleave='+str(IF)
                +' -o ' +filename[:-1]+'o'
                os.system(cmd1)
                cmd2 = filename[:-1]+'o '
                try:
                    runtime=int(subprocess.Popen(cmd2, executable='/bin/bash', shell=True,
                            stdout=subprocess.PIPE).stdout.read())
                except:
                    runtime = None #None if fails
                    logger.warning('Could not compile ' + filename + 
                                   ' due to time out. Setting runtime to: '+str(runtime)+'.' +
                                   ' Consider increasing the timeout, which is set to 4 seconds.')
                one_program_runtimes[i] = runtime
                if runtime<opt_runtime:
                    opt_runtime = runtime
                    opt_factor = (VF,IF)
        opt_runtimes[filename] = opt_runtime
        opt_factors[filename] = opt_factor
        all_program_runtimes[filename]=copy.deepcopy(one_program_runtimes)
    data={'opt_runtimes':opt_runtimes,'opt_factors':opt_factors,'all_program_runtimes':all_program_runtimes}
    output = open(os.path.join(rundir,'bruteforce_runtimes.pkl'), 'wb')
    pickle.dump(data, output)
    output.close()

def rename_contents(rundir, contents):
    '''Takes in a run directory, and the contents of the pkl file, renames the directory of the contents
    of the pkl file based on the new rundir specified. It is useful when the user reuses the provided pkl
    file with new rundir.'''
    new_contents = {} 
    for key in contents.keys():
        value = contents[key] 
        suffix_filename = key.split('/')[-1]  # extracts the file name 
        new_path = os.path.join(rundir, suffix_filename)
        new_contents[new_path] = value
    return new_contents 

def get_O3_runtimes(rundir,files):
    f = open('runtimes_icx7_omp_orig.pickle', 'rb')
    runtimes = pickle.load(f)    
    f.close()

    vf_if = {}
    for fn in runtimes.keys():
        rt_mean = np.mean(runtimes[fn])
        if fn not in vf_if.keys():
            vf_if[fn] = rt_mean
        else:
            rt_mean_pre = vf_if[fn]
            if rt_mean < rt_mean_pre:
                vf_if[fn] = rt_mean
    return vf_if

def get_snapshot_from_code(code,loop_idx=None):
    ''' take snapshot of the loop code and encapsulate
     in a function declaration so the parser can output
     AST tree.'''
    found = False
    new_code = []
    for line in code:
        if 'void loop()' in line:
            found = True
            line = "void loop(ret)\n"
        if found:
            new_code.append(line)
    return new_code

def get_encodings_from_local(rundir):
    '''returns encodings from obs_encodings.pkl if 
    file exists in trainig directory.'''
    encodings = {}
    print('Checking if local obs_encodings.pkl file exists.') 
    if os.path.exists(os.path.join(rundir,'obs_encodings.pkl')):
        print('found local obs_encodings.pkl.')
        with open(os.path.join(rundir,'obs_encodings.pkl'), 'rb') as f:
            return rename_contents(rundir, pickle.load(f))
    return encodings

def run_llvm_test_shell_command(rundir,filename):
    '''runs the file after the pragma is injected 
    and returns runtime.'''
    full_path_header = os.path.join(rundir, 'header.c')
    cmd1 = 'timeout ' + TEST_SHELL_COMMAND_TIMEOUT + ' ' + os.environ['CLANG_BIN_PATH'] + ' -O3 -lm '+full_path_header \
    +' ' +filename+' -o ' +filename[:-1]+'o'
    cmd2 = filename[:-1]+'o '
    os.system(cmd1)
    try:
        runtime=float(subprocess.Popen(cmd2, executable='/bin/bash', shell=True, stdout=subprocess.PIPE).stdout.read())
    except:
        runtime = None #None if fails
        print('Could not compile ' + filename +  
                       ' due to time out. Setting runtime to: ' + 
                       str(runtime)+'. Considering increasing the TEST_SHELL_COMMAND_TIMEOUT,'+ 
                       ' which is currently set to ' + TEST_SHELL_COMMAND_TIMEOUT)
    return runtime

def get_runtime(rundir,current_filename, VF, IF):
    '''produces the new file with the pragma and 
    compiles to get runtime.'''
    #runtime=run_llvm_test_shell_command(rundir,current_filename)
    file_name = current_filename.split('/')[-1]
    insert_pragma(SAVE_DIR, rundir, file_name, VF, IF, 1)
    os.system("clang -O3 " + SAVE_DIR+'/'+file_name + ' ' + rundir+'/header.c')
    output = subprocess.check_output("./a.out", shell=True)
    runtime = float(output.split()[3])
    return runtime


def get_block(i,code):
    j = i
    cnt = 0
    while(True):
        line = code[j]
        if re.match(r'^\s*//',line) or re.match(r'^\s*$',line):
            j += 1
            continue
        if '{' in line:
            cnt += line.count('{')
        if '}' in line:
            cnt -= line.count('}')
        if cnt == 0 and not (re.match(r'^\s*for\s*\(',line) or re.match(r'^\s*while\s*\(',line)):
            return (i,j)
        if cnt == 0 and line.endswith(';\n'):
            return (i,j)
        if (re.match(r'^\s*for\s*\(',line) or re.match(r'^\s*while\s*\(',line)) and i != j:
            return get_block(j,code)
        j=j+1

def get_vectorized_code(code):
    '''Used by get_vectorized_codes function to do the parsing 
    of a single code to detect the loops, inject commented pragmas,
    and collect data.''' 
    #print(code)
    new_code = []
    for_loops_indices = []
    i=0
    pragma_indices = []
    num_elems_in_new_code=0
    while i < len(code):
        line=code[i]
        if re.match(r'^\s*for\s*\(',line) or re.match(r'^\s*while\s*\(',line):
            begining,ending = get_block(i,code)
            orig_i=i
            while(i<ending+1):
                if i==begining:
                    new_code.append('//'+pragma_line.format(64,16))#start with -O3 vectorization
                    num_elems_in_new_code += 1
                    pragma_indices.append(num_elems_in_new_code-1)
                new_code.append(code[i])
                num_elems_in_new_code += 1
                i = i+1
            # to pick the index of the most innner loop    
            #for_loops_indices.append((orig_i,ending))
            for_loops_indices.append((begining,ending))
            i=ending+1
            continue
        new_code.append(line)
        num_elems_in_new_code += 1
        i += 1

    return for_loops_indices,pragma_indices,new_code

def get_vectorized_codes(orig_trainfiles, new_trainfiles):
    '''parses the original training files to detect loops.
    Then copies the files to the new directory with
    commented pragmas.'''
    loops_idxs_in_orig = {}
    pragmas_idxs = {}
    const_new_codes ={}
    num_loops = {}
    const_orig_codes={}
    for o_fn,n_fn in zip(orig_trainfiles,new_trainfiles):
        f = open(o_fn,'r')
        try:
            code = f.readlines()
        except:
            f.close()
            continue
        #print(o_fn)
        loops_idx, pragmas_idx, new_code = get_vectorized_code(code)
        if not pragmas_idx:
            f.close()
            continue
        const_orig_codes[n_fn] = list(code)
        loops_idxs_in_orig[n_fn]=list(loops_idx)
        pragmas_idxs[n_fn] = list(pragmas_idx)
        const_new_codes[n_fn] = list(new_code)
        num_loops[n_fn] = len(pragmas_idx)
        #logger.info('writing file... ' + n_fn)
        #print('writing file... ' + n_fn)
        nf = open(n_fn,'w')
        nf.write(''.join(new_code))
        nf.close()
        f.close()
    return loops_idxs_in_orig, pragmas_idxs, const_new_codes,num_loops,const_orig_codes





def insert_pragma(save_dir, run_dir, fname, VF, IF, insert_timer):
    file = fname.split('.')[0]
    new_file = save_dir + '/' + file + ".c"
    fw = open(new_file, 'w')
    with open(run_dir+'/'+fname) as fr:
        for line in fr:
            #print(line.rstrip())
            if re.match(r'^\s*for\s*\(',line) or re.match(r'^\s*while\s*\(',line):
                fw.write("#pragma clang loop vectorize_width("+str(VF)+") interleave_count("+str(IF)+")\n")         
            if (insert_timer):
                if (re.match(r'\s+return 0;', line)):
                    fw.write("gettimeofday(&end, NULL);\n") 
                    fw.write("double time_taken = end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6;\n")
                    fw.write("printf(\"time program took %f seconds to execute\\n\", time_taken);\n")
            fw.write(line)
            if (insert_timer):
                if (re.match(r'int\s*main\(', line)): 
                    fw.write("struct timeval start, end;\ngettimeofday(&start, NULL);\n") 
    fw.close()
