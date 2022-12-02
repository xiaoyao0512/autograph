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

    f = open('lore_runtimes.pickle', 'rb')
    runtimes = pickle.load(f)    
    f.close()
    vf_if = {}
    times = {}
    files_VF_IF = runtimes.keys()
    vf_list = [1, 2, 4, 8, 16]
    if_list = [1, 2, 4, 8, 16]
    baseline = {}
    for file_VF_IF in files_VF_IF:
        tmp = file_VF_IF.rpartition('.')
        fn = tmp[0]
        tmp = tmp[2].split('-')
        VF = vf_list.index(int(tmp[0]))
        IF = if_list.index(int(tmp[1]))
        label = 5 * VF + IF
        fn_c = fn
        rt_mean = np.median(runtimes[file_VF_IF])
        if fn_c not in vf_if.keys():
            vf_if[fn_c] = (rt_mean, label)
        else:
            rt_mean_pre = vf_if[fn_c][0]
            if rt_mean < rt_mean_pre:
                vf_if[fn_c] = (rt_mean, label)   

    if (file_dir not in vf_if):
        continue
    print(file_dir)
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
        #print(input_full_path_filename)
        train_lines, hash_to_string_dict = path_extractor.extract_paths(input_full_path_filename)
    except:
        print('Could not parse file',current_filename, '. Try removing it.')
        raise 
    #results, code_vectors = code2vec.predict(train_lines)
    dataset  = train_input_reader.process_and_iterate_input_from_data_lines(train_lines)
    obs = []
    tensors = list(dataset)[0][0]
    for tensor in tensors:
        #with tf.compat.v1.Session() as sess: 
        #    sess.run(tf.compat.v1.tables_initializer())
        obs.append(tf.squeeze(tensor).numpy().tolist())
    obs = list(np.concatenate(obs).flat)
    assert len(obs) == 800, "ERROR!"
    #print(obs)
    feat.append([obs])

features = {}
features['feat'] = feat
features['labels'] = labels
features['files'] = files
with open('lore_features.json', 'w') as f:
    json.dump(features, f) 
