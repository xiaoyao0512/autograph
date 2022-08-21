import re
from ast import literal_eval
import matplotlib.pyplot as plt
import numpy as np
import json
import itertools
from scipy.stats import pearsonr




def parsing(filename):
    files = []
    pred = {}
    pred2 = {}
    baseline = {}
    baseline2 = {}
    reward = {}
    reward2 = {}
    flag = True
    sp1 = {}
    sp12 = {}
    sp2 = {}
    sp22 = {}
    base_l = []
    pred_l = []
    rewd_l = []
    spgt_l = []
    spo3_l = []
    acc = 0
    spgt_avg = 0
    spo3_avg = 0
    with open(filename) as file:
        f = ''
        for line in file:
            info = line.rstrip()
            #print("info = ", info)
            #if ((re.search("Training completed.", info)) and flag):
            #    flag = False
                
            if (re.search("f = ", info)):
                tmp = info.split(' ')
                f = tmp[-1]
                files.append(tmp[-1])
            elif (re.search("label = ", info)):
                #print("3")
                assert f != ''
                tmp = info.split(' ')
                str_set = tmp[-2] + ' ' + tmp[-1]
                lit_set = literal_eval(str_set)
                VF = lit_set[0]
                IF = lit_set[1]
                val1 = int(tmp[-3])
                val2 = VF*5 + IF
                baseline[f] = [val1]
                pred[f] = [val2]
                #print(val1, val2)
                if val1 not in baseline2:
                    baseline2[val1] = [f]
                else:
                    baseline2[val1].append(f)
                if val2 not in pred2:
                    pred2[val2] = [f]
                else:
                    pred2[val2].append(f)
                base_l.append(val1)
                pred_l.append(val2)
            elif (re.search("reward = ", info)):
                #print("4")
                assert f != ''
                tmp = info.split(' ')
                #print(float(tmp[-1]))
                val = float(tmp[-1])
                reward[f] = [val]
                if val not in reward2:
                    reward2[val] = [f]
                else:
                    reward2[val].append(f)
                rewd_l.append(val)
            elif (re.search("ground truth", info)):
                tmp = info.split(' ')
                val = float(tmp[-1])
                sp1[f] = [val]
                if val not in sp12:
                    sp12[val] = [f]
                else:
                    sp12[val].append(f)
                spgt_l.append(val)
            elif (re.search("baseline O3", info)):
                tmp = info.split(' ')
                val = float(tmp[-1])
                sp2[f] = [val]
                if val not in sp22:
                    sp22[val] = [f]
                else:
                    sp22[val].append(f)
                spo3_l.append(val)
            elif (re.search("accuracy_list", info)):
                tmp = info.split(' ')
                lit_set = literal_eval(tmp[-1])
                acc = lit_set[0]
            elif (re.search("exec2_list", info)):
                tmp = info.split(' ')
                lit_set = literal_eval(tmp[-1])
                spgt_avg = lit_set[0]
            elif (re.search("exec3_list", info)):
                tmp = info.split(' ')
                lit_set = literal_eval(tmp[-1])
                spo3_avg = lit_set[0]
               
    return files, (baseline, baseline2), (pred, pred2), (reward, reward2), (sp1,sp12), (sp2, sp22), base_l, pred_l, rewd_l, spgt_l, spo3_l, acc, spgt_avg, spo3_avg


def convert(list1, dict1, dict2):
    list2 = []    
    for item in list1:
        for f in dict1[item]:
            list2.append(dict2[f][0])
    return list2

def get_mean(list2d):
    size = len(list2d[0])
    sum_l = [0] * size
    for lst in list2d:
        for i in range(size):
            sum_l[i] += lst[i]
    mean = [x / 5 for x in sum_l]
    return mean

def get_min(list2d):
    size = len(list2d[0])
    min_l = [1000] * size
    for lst in list2d:
        for i in range(size):
            min_l[i] = min(min_l[i], lst[i])
    return min_l

def get_max(list2d):
    size = len(list2d[0])
    max_l = [-1000] * size
    for lst in list2d:
        for i in range(size):
            max_l[i] = max(max_l[i], lst[i])
    return max_l

def parsing2(f1, f2, reps):

    mean1 = {"rewd": [], "spgt": [], "spo3": []}
    upper1 = {"rewd": [], "spgt": [], "spo3": []}
    lower1 = {"rewd": [], "spgt": [], "spo3": []}
    mean2 = {"rewd": [], "spgt": [], "spo3": []}
    upper2 = {"rewd": [], "spgt": [], "spo3": []}
    lower2 = {"rewd": [], "spgt": [], "spo3": []}
    total1 = {"base": [], "pred": [], "rewd": [], "spgt": [], "spo3": [], "acc": [], "spgt_avg": [], "spo3_avg": []}
    total2 = {"base": [], "pred": [], "rewd": [], "spgt": [], "spo3": [], "acc": [], "spgt_avg": [], "spo3_avg": []}
    for i in range(reps):
        fn1 = f1 + str(i+1)
        fn2 = f2 + str(i+1)
        file1, (base1_1, base1_2), (pred1_1, pred1_2), (rewd1_1, rewd1_2), (speedup_gt1_1, speedup_gt1_2), (speedup_o1_1, speedup_o1_2), base_l1, pred_l1, rewd_l1, spgt_l1, spo3_l1, acc1, spgt_avg1, spo3_avg1 = parsing(fn1)  
        file2, (base2_1, _), (pred2_1, _), (rewd2_1, _), (speedup_gt2_1, _), (speedup_o2_1, _), base_l2, pred_l2, rewd_l2, spgt_l2, spo3_l2, acc2, spgt_avg2, spo3_avg2 = parsing(fn2)
        #print(base_l1)
        print("action (neuro): ", pearsonr(base_l1, pred_l1))
        print("action (graphsage): ", pearsonr(base_l2, pred_l2))
        assert file1.sort() == file2.sort()
        sorted_base1 = sorted(list(itertools.chain(*base1_1.values())))
        sorted_pred1 = sorted(list(itertools.chain(*pred1_1.values())))
        sorted_rewd1 = sorted(list(itertools.chain(*rewd1_1.values())))
        sorted_spgt1 = sorted(list(itertools.chain(*speedup_gt1_1.values())))
        sorted_spo31 = sorted(list(itertools.chain(*speedup_o1_1.values())))  
        sorted_base2 = convert(sorted(set(sorted_base1)), base1_2, base2_1)
        sorted_pred2 = convert(sorted(set(sorted_pred1)), pred1_2, pred2_1)
        sorted_rewd2 = convert(sorted(set(sorted_rewd1)), rewd1_2, rewd2_1)
        sorted_spgt2 = convert(sorted(set(sorted_spgt1)), speedup_gt1_2, speedup_gt2_1)
        sorted_spo32 = convert(sorted(set(sorted_spo31)), speedup_o1_2, speedup_o2_1)
        #print("sorted action (neuro): ", pearsonr(sorted_base1, sorted_pred1))
        #print("sorted action (graphsage): ", pearsonr(sorted_base2, sorted_pred2))
        total1["base"].append(sorted_base1)
        total1["pred"].append(sorted_pred1)
        total1["rewd"].append(sorted_rewd1)
        total1["spgt"].append(sorted_spgt1)
        total1["spo3"].append(sorted_spo31)
        total1["acc"].append(acc1)
        total1["spgt_avg"].append(spgt_avg1)
        total1["spo3_avg"].append(spo3_avg1)
        total2["base"].append(sorted_base2)
        total2["pred"].append(sorted_pred2)
        total2["rewd"].append(sorted_rewd2)
        total2["spgt"].append(sorted_spgt2)
        total2["spo3"].append(sorted_spo32)
        total2["acc"].append(acc2)
        total2["spgt_avg"].append(spgt_avg2)
        total2["spo3_avg"].append(spo3_avg2)
    mean1["rewd"] = get_mean(total1["rewd"])
    mean1["spgt"] = get_mean(total1["spgt"])
    mean1["spo3"] = get_mean(total1["spo3"])
    #print(total2["rewd"])
    mean2["rewd"] = get_mean(total2["rewd"])
    mean2["spgt"] = get_mean(total2["spgt"])
    mean2["spo3"] = get_mean(total2["spo3"]) 
    # measure correlation between neuro and ours 
    # scipy.stats.pearsonr(x, y)

    #print("Correlation (reward): ", pearsonr(mean1["rewd"], mean2["rewd"]))  
    #print("Correlation (spgt): ", pearsonr(mean1["spgt"], mean2["spgt"]))  
    #print("Correlation (spo3): ", pearsonr(mean1["spo3"], mean2["spo3"]))
    X = np.stack((mean1["rewd"], mean2["rewd"]), axis=0)  
    cov1 = np.cov(X)
    X = np.stack((mean1["spgt"], mean2["spgt"]), axis=0)  
    cov2 = np.cov(X)
    X = np.stack((mean1["spo3"], mean2["spo3"]), axis=0)  
    cov3 = np.cov(X)
    print("Covariance matrix (reward): ", cov1)
    print("Covariance matrix (spgt): ", cov2)
    print("Covariance matrix (spo3): ", cov3)
    print("Accuracy (neurovec): ", total1["acc"])
    print("Speedup compared to optimal VF/IF (neurovec): ", total1["spgt_avg"])
    print("Speedup compared to O3 (neurovec): ", total1["spo3_avg"])
    print("Accuracy (graphsage): ", total2["acc"])
    print("Speedup compared to optimal VF/IF (graphsage): ", total2["spgt_avg"])
    print("Speedup compared to O3 (graphsage): ", total2["spo3_avg"])
      
    lower1["rewd"] = get_min(total1["rewd"])
    lower1["spgt"] = get_min(total1["spgt"])
    lower1["spo3"] = get_min(total1["spo3"])
    lower2["rewd"] = get_min(total2["rewd"])
    lower2["spgt"] = get_min(total2["spgt"])
    lower2["spo3"] = get_min(total2["spo3"])    
    upper1["rewd"] = get_max(total1["rewd"])
    upper1["spgt"] = get_max(total1["spgt"])
    upper1["spo3"] = get_max(total1["spo3"])
    upper2["rewd"] = get_max(total2["rewd"])
    upper2["spgt"] = get_max(total2["spgt"])    
    upper2["spo3"] = get_max(total2["spo3"])   
    
    return total1, total2, mean1, mean2, lower1, lower2, upper1, upper2


def scatter_plots(total1, total2):
    plt.figure(0)
    plt.scatter(total1["pred"][0], total2["pred"][0])
    plt.xlabel('Neuro-vectorizer action')
    plt.ylabel('GraphSage action')
    #plt.legend("Action space(VF/IF)")
    plt.savefig("figure/scatter_action.png")

    plt.figure(1)
    plt.scatter(total1["rewd"][0], total2["rewd"][0])
    plt.xlabel('Neuro-vectorizer reward')
    plt.ylabel('GraphSage reward')
    #plt.legend("Action space(VF/IF)")
    plt.savefig("figure/scatter_reward.png")

    plt.figure(2)
    plt.scatter(total1["spgt"][0], total2["spgt"][0])
    plt.xlabel('Neuro-vectorizer speedup (gt)')
    plt.ylabel('GraphSage speedup (gt)')
    #plt.legend("Action space(VF/IF)")
    plt.savefig("figure/scatter_spgt.png")

    plt.figure(3)
    plt.scatter(total1["spo3"][0], total2["spo3"][0])
    plt.xlabel('Neuro-vectorizer speedup (o3)')
    plt.ylabel('GraphSage speedup (o3)')
    #plt.legend("Action space(VF/IF)")
    plt.savefig("figure/scatter_spo3.png")

def fillbetween(mean1, mean2, lower1, lower2, upper1, upper2):
    plt.figure(4)
    plt.plot(list(range(len(lower1["rewd"]))), mean1["rewd"], 'r', linewidth=0.5)
    #plt.plot(list(range(len(lower1["rewd"]))), lower1["rewd"], 'r', linewidth=0.5)
    #plt.plot(list(range(len(lower1["rewd"]))), upper1["rewd"], 'r', linewidth=0.5)
    #print(upper1["rewd"])
    plt.fill_between(list(range(len(lower1["rewd"]))), lower1["rewd"], upper1["rewd"], step="pre", alpha=0.8, label="Neuro-vectorizer")
    plt.plot(list(range(len(lower1["rewd"]))), mean2["rewd"], 'g', linewidth=0.5)
    plt.fill_between(list(range(len(lower2["rewd"]))), lower2["rewd"], upper2["rewd"], step="pre", alpha=0.2, label="GraphSage")   
    plt.xlabel('Program') 
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig("figure/fillbetween_rewd.png")

    plt.figure(5)
    plt.plot(list(range(len(lower1["spgt"]))), mean1["spgt"], 'r', linewidth=0.5)
    #plt.plot(list(range(len(lower1["rewd"]))), lower1["rewd"], 'r', linewidth=0.5)
    #plt.plot(list(range(len(lower1["rewd"]))), upper1["rewd"], 'r', linewidth=0.5)
    #print(upper1["rewd"])
    plt.fill_between(list(range(len(lower1["spgt"]))), lower1["spgt"], upper1["spgt"], step="pre", alpha=0.8, label="Neuro-vectorizer")
    plt.plot(list(range(len(lower1["spgt"]))), mean2["spgt"], 'g', linewidth=0.5)
    plt.fill_between(list(range(len(lower2["spgt"]))), lower2["spgt"], upper2["spgt"], step="pre", alpha=0.2, label="GraphSage")   
    plt.xlabel('Program') 
    plt.ylabel('spgt')
    plt.legend()
    plt.savefig("figure/fillbetween_spgt.png")

    plt.figure(6)
    plt.plot(list(range(len(lower1["spo3"]))), mean1["spo3"], 'r', linewidth=0.5)
    #plt.plot(list(range(len(lower1["rewd"]))), lower1["rewd"], 'r', linewidth=0.5)
    #plt.plot(list(range(len(lower1["rewd"]))), upper1["rewd"], 'r', linewidth=0.5)
    #print(upper1["rewd"])
    plt.fill_between(list(range(len(lower1["spo3"]))), lower1["spo3"], upper1["spo3"], step="pre", alpha=0.8, label="Neuro-vectorizer")
    plt.plot(list(range(len(lower1["spo3"]))), mean2["spo3"], 'g', linewidth=0.5)
    plt.fill_between(list(range(len(lower2["spo3"]))), lower2["spo3"], upper2["spo3"], step="pre", alpha=0.2, label="GraphSage")   
    plt.xlabel('Program') 
    plt.ylabel('spo3')
    plt.legend()
    plt.savefig("figure/fillbetween_spo3.png")

def find_files()

total1, total2, mean1, mean2, lower1, lower2, upper1, upper2 = parsing2("results/neurovec", "results/graphsagegcn128-256-", 5)
# scatter plot (reward/action)

scatter_plots(total1, total2)

# reward / speedup line plot with confidence bands
# fill_between
# ax.fill_between(x, y1, y2, alpha=0.2)

fillbetween(mean1, mean2, lower1, lower2, upper1, upper2)

#print(len(sorted_spgt1))







'''
sorted_files = []
for item in sorted_spo31:
    sorted_files.append(speedup_o1_2[item])
x = list(range(len(sorted_files)))  
#print(x)
plt.figure(0)
plt.plot(x, sorted_rewd1, label = filename1 + " reward")
plt.plot(x, sorted_rewd2, label = filename2 + " reward")
plt.legend()
plt.savefig("figure/reward4.png")

plt.plot(x, sorted_spgt1, label = filename1 + " speedupgt")
plt.plot(x, sorted_spgt2, label = filename2 + " speedupgt")
plt.legend()
plt.savefig("figure/speedupgt.png")
plt.figure(1)
plt.plot(x, sorted_spo31, label = filename1 + " speedupO3")
plt.plot(x, sorted_spo32, label = filename2 + " speedupO3")
plt.legend()
plt.savefig("figure/speedupO34.png")

#plt.figure(3)
#plt.plot(x, sorted_base1, label = "baseline action")
#plt.plot(x, sorted_pred1, label = filename1 + " action")
#plt.plot(x, sorted_pred2, label = filename2 + " action")
#plt.legend()
#plt.savefig("figure/action.png")






'''




'''
graphsage_ggnn32 = {
    "file": file1,
    "base": base1,
    "pred": pred1,
    "rewd": rewd1,
    "speedup_gt": speedup_gt1,
    "speedup_o3": speedup_o1
}

neurovectorizer = {
    "file": file2,
    "base": base2,
    "pred": pred2,
    "rewd": rewd2,
    "speedup_gt": speedup_gt2,
    "speedup_o3": speedup_o2
}
analysis = {
    "graphsage_ggnn32": graphsage_ggnn32,
    "neurovectorizer": neurovectorizer
}

with open('analysis.json', 'w') as f:
    json.dump(analysis, f) 

'''
'''
n_bins = 25

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(pred, bins=n_bins)
axs[1].hist(baseline, bins=n_bins)


plt.savefig("action_space.png")
sort_files = []
all_files = []
data = {}
wrong = {}
for i in range(len(files)):
    #if (pred[i] == 15 or pred[i] == 16):
    f_type = files[i].split('_')[0]
    if f_type not in data.keys():
        data[f_type] = 1
    else:
        data[f_type] += 1
    #print(f_type)
    if ((pred[i] == 15 or pred[i] == 16) and (baseline[i] != pred[i])):
        if f_type not in wrong.keys():
            wrong[f_type] = 1
        else:
            wrong[f_type] += 1        
        if (reward[i] < -1):
            print("actual = ", baseline[i])
            print("f = ", files[i])
            sort_files.append(files[i])
            print("reward = ", reward[i])
            #print(exec_time[i])

perc = {}
for i in wrong.keys():
    perc[i] = wrong[i] / data[i]

print("files = ", sorted(sort_files))
print("data = ", data)
print("wrong = ", wrong)
print("percentage = ", perc)
'''
