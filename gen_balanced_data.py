import json, pickle
from m2m_data_loader import smote
import numpy as np
import glob


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
VF_list = [1, 2, 4, 8, 16]
IF_list = [1, 2, 4, 8, 16]
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
    if fn not in vf_if.keys():
        vf_if[fn] = (rt_mean, label)
    else:
        rt_mean_pre = vf_if[fn][0]
        if rt_mean < rt_mean_pre:
            vf_if[fn] = (rt_mean, label)
    if fn_c not in times.keys():
        times[fn_c] = {}
    if label not in times[fn_c].keys():
        times[fn_c][label] = rt_mean
    else:
        rt_mean_pre = times[fn][label]
        if (rt_mean < rt_mean_pre):
            times[fn_c][label] = rt_mean

    if fn_c not in baseline.keys():
        baseline[fn_c] = base_mean
    else:
        base_mean_pre = baseline[fn_c]
        if base_mean < base_mean_pre:
            baseline[fn_c] = base_mean


benchmark = "autograph"

file_train = []
label_train = []
feat_train = []
if benchmark == "autograph":
    with open('lore_features_training.json') as f:
        features = json.load(f)
    feat_train = features["feat"]
    label_train = features["labels"]
    file_train = features["files"]
elif benchmark == "neurovec":
    with open('lore_features_training_neurovec.json') as f:
        features = json.load(f)
    feat_train = features["feat"]
    label_train = features["labels"]
    file_train = features["files"]
elif benchmark == "code2vec":
    with open("lore_features.json") as f:
        features = json.load(f)
    feats = features["feat"]
    labels = features["labels"]
    files = features["files"]
    for f in glob.glob("lore/json_lore_spec/**/*.json", recursive=True):
        fn = f.split('/', 2)[-1][:-5]
        if fn not in files:
            continue
        fidx = files.index(fn)
        file_train.append(fn)
        label_train.append(labels[fidx])
        feat_train.append(feats[fidx])

exec_train = []
for idx in range(len(file_train)):
    label = label_train[idx]
    fn = file_train[idx]
    exec_train.append(times[fn][label])


file_train = np.array(file_train)
label_train = np.array(label_train)
feat_train = np.array(feat_train)

print(feat_train)

num_sample_per_class = [0] * 25

for label in label_train:
    num_sample_per_class[label] += 1
class_max = max(num_sample_per_class)
nb_classes = 25
print("before smote, ", num_sample_per_class)
aug_data, aug_label, aug_exec, aug_file = smote(feat_train, label_train, nb_classes, class_max, exec_train, file_train)
#aug_file = ["file"+str(i) for i in range(len(aug_data))]
full_feat = []
full_label = []
full_file = []
full_exec = []

feat_train = feat_train.tolist()
label_train = label_train.tolist()
file_train = file_train.tolist()
aug_data = aug_data.tolist()
aug_label = aug_label.tolist()

for i in range(len(feat_train)):
    full_feat.append(feat_train[i])
    full_label.append(label_train[i])
    full_file.append(file_train[i])
    full_exec.append(exec_train[i])

for i in range(len(aug_data)):
    full_feat.append(aug_data[i])
    full_label.append(aug_label[i])
    full_file.append(aug_file[i])
    full_exec.append(aug_exec[i])


print(len(full_label))

dist = {}
for l in full_label:
    if l not in dist:
        dist[l] = 1
    else:
        dist[l] += 1

print(dist.values())

full_features = {}
full_features["feat"] = full_feat
full_features["labels"] = full_label
full_features["files"] = full_file
full_features["exec"] = full_exec

if benchmark == "autograph":
    with open('lore_bal_features_training.json', 'w') as f:
        json.dump(full_features, f)
elif benchmark == "neurovec":
    with open('lore_bal_features_training_neurovec.json', 'w') as f:
        json.dump(full_features, f)
elif benchmark == "code2vec":
    with open("lore_bal_features.json", 'w') as f:
        json.dump(full_features, f)
