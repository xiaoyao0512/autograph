import json
import glob


file_list = ['spec2006_v1.1', 'NPB_2.3-OpenACC-C', 'polybench_4.1']
#with open('lore_features2_graphsage_gcn128.json') as f:
with open('lore_features.json') as f:
    features = json.load(f)

feats = features["feat"]
labels = features["labels"]
files = features["files"]

num = len(files)
training = {}
testing_spec = {}
testing_npb = {}
testing_poly = {}
training["feat"] = []
training["labels"] = []
training["files"] = []
testing_spec["feat"] = []
testing_spec["labels"] = []
testing_spec["files"] = []

testing_npb["feat"] = []
testing_npb["labels"] = []
testing_npb["files"] = []

testing_poly["feat"] = []
testing_poly["labels"] = []
testing_poly["files"] = []

for i in range(num):
    f = files[i]
    #print(f)
    benchmark = f.split('/')[0]
    if benchmark == "spec2006_v1.1":
        #print(f)
        #print(max(feats[i][0]))
        testing_spec["feat"].append(feats[i])
        testing_spec["labels"].append(labels[i])
        testing_spec["files"].append(files[i])
    elif benchmark == "NPB_2.3-OpenACC-C":
        print(len(feats[i][0]))
        testing_npb["feat"].append(feats[i])
        testing_npb["labels"].append(labels[i])
        testing_npb["files"].append(files[i])
    elif benchmark == "polybench_4.1":
        #print(f)
        testing_poly["feat"].append(feats[i])
        testing_poly["labels"].append(labels[i])
        testing_poly["files"].append(files[i])
    else:
        #print(f)
        #print()
        training["feat"].append(feats[i])
        training["labels"].append(labels[i])
        training["files"].append(files[i])


#with open("lore_features_training.json", "w") as outfile:
with open("lore_features_training_neurovec.json", "w") as outfile:
    json.dump(training, outfile)


#with open("lore_features_testing_poly.json", "w") as outfile:
with open("lore_features_testing_neurovec_poly.json", "w") as outfile:
    json.dump(testing_poly, outfile)

#with open("lore_features_testing_npb.json", "w") as outfile:
with open("lore_features_testing_neurovec_npb.json", "w") as outfile:
    json.dump(testing_npb, outfile)
#with open("lore_features_testing_spec.json", "w") as outfile:
with open("lore_features_testing_neurovec_spec.json", "w") as outfile:
    json.dump(testing_spec, outfile)


'''
for f in glob.glob('json_lore_testing/**/*.json', recursive=True):
    f_name = f.split('/', 1)[-1].rpartition('.')[0]
    print(f_name)
'''
