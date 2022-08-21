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

    total1 = {"base": {}, "pred": {}, "rewd": {}, "spgt": {}, "spo3": {}, "acc": [], "spgt_avg": [], "spo3_avg": []}
    total2 = {"base": {}, "pred": {}, "rewd": {}, "spgt": {}, "spo3": {}, "acc": [], "spgt_avg": [], "spo3_avg": []}
    for i in range(reps):
        fn1 = f1 + str(i+1)
        fn2 = f2 + str(i+1)
        file1, (base1_1, base1_2), (pred1_1, pred1_2), (rewd1_1, rewd1_2), (speedup_gt1_1, speedup_gt1_2), (speedup_o1_1, speedup_o1_2), base_l1, pred_l1, rewd_l1, spgt_l1, spo3_l1, acc1, spgt_avg1, spo3_avg1 = parsing(fn1)  
        file2, (base2_1, _), (pred2_1, _), (rewd2_1, _), (speedup_gt2_1, _), (speedup_o2_1, _), base_l2, pred_l2, rewd_l2, spgt_l2, spo3_l2, acc2, spgt_avg2, spo3_avg2 = parsing(fn2)
        #print(base_l1)
        for f in file1:
            if f not in total1["base"].keys():
                total1["base"][f] = [base1_1[f][0]]
                total1["pred"][f] = [pred1_1[f][0]]
                total1["rewd"][f] = [rewd1_1[f][0]]
                total1["spgt"][f] = [speedup_gt1_1[f][0]]
                total1["spo3"][f] = [speedup_o1_1[f][0]]
                total2["base"][f] = [base2_1[f][0]]
                total2["pred"][f] = [pred2_1[f][0]]
                total2["rewd"][f] = [rewd2_1[f][0]]
                total2["spgt"][f] = [speedup_gt2_1[f][0]]
                total2["spo3"][f] = [speedup_o2_1[f][0]]
            else:
                total1["base"][f].append(base1_1[f][0])
                total1["pred"][f].append(pred1_1[f][0])
                total1["rewd"][f].append(rewd1_1[f][0])
                total1["spgt"][f].append(speedup_gt1_1[f][0])
                total1["spo3"][f].append(speedup_o1_1[f][0])
                total2["base"][f].append(base2_1[f][0])
                total2["pred"][f].append(pred2_1[f][0])
                total2["rewd"][f].append(rewd2_1[f][0])
                total2["spgt"][f].append(speedup_gt2_1[f][0])
                total2["spo3"][f].append(speedup_o2_1[f][0])
        total1["acc"].append(acc1)
        total1["spgt_avg"].append(spgt_avg1)
        total1["spo3_avg"].append(spo3_avg1)
        total2["acc"].append(acc2)
        total2["spgt_avg"].append(spgt_avg2)
        total2["spo3_avg"].append(spo3_avg2)



    return total1, total2


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

def get_list(files, dict1):
    val = []
    for f in files:
        val.append(dict1[f])
    return val

def fillbetween(mean1, mean2, lower1, lower2, upper1, upper2):
    mean1_sorted = dict(sorted(mean1["rewd"].items(), key=lambda item: item[1]))
    mean1_sorted_gt = dict(sorted(mean1["spgt"].items(), key=lambda item: item[1]))
    files = list(mean1_sorted.keys())
    files_gt = list(mean1_sorted_gt.keys())
    file_idx = list(range(len(mean1_sorted.keys())))
    file_idx_gt = list(range(len(mean1_sorted_gt.keys())))
    y1_values = list(mean1_sorted.values())
    y1_values_gt = list(mean1_sorted_gt.values())
    plt.figure(0)
    plt.plot(file_idx, y1_values, 'r', linewidth=0.5)
    #plt.plot(list(range(len(lower1["rewd"]))), lower1["rewd"], 'r', linewidth=0.5)
    #plt.plot(list(range(len(lower1["rewd"]))), upper1["rewd"], 'r', linewidth=0.5)
    #print(upper1["rewd"])
    plt.fill_between(file_idx, get_list(files, lower1["rewd"]), get_list(files, upper1["rewd"]), step="pre", alpha=0.8, label="Neuro-vectorizer")
    plt.plot(file_idx, get_list(files, mean2["rewd"]), 'g', linewidth=0.5)
    plt.fill_between(file_idx, get_list(files, lower2["rewd"]), get_list(files, upper2["rewd"]), step="pre", alpha=0.2, label="GraphSage")   
    plt.xlabel('Program') 
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig("figure/fillbetween_rewd.png")

    plt.figure(1)
    plt.plot(file_idx_gt, y1_values_gt, 'r', linewidth=0.5)
    #plt.plot(list(range(len(lower1["rewd"]))), lower1["rewd"], 'r', linewidth=0.5)
    #plt.plot(list(range(len(lower1["rewd"]))), upper1["rewd"], 'r', linewidth=0.5)
    #print(upper1["rewd"])
    plt.fill_between(file_idx_gt, get_list(files_gt, lower1["spgt"]), get_list(files_gt, upper1["spgt"]), step="pre", alpha=0.8, label="Neuro-vectorizer")
    plt.plot(file_idx_gt, get_list(files_gt, mean2["spgt"]), 'g', linewidth=0.5)
    plt.fill_between(file_idx_gt, get_list(files_gt, lower2["spgt"]), get_list(files_gt, upper2["spgt"]), step="pre", alpha=0.2, label="GraphSage")   
    plt.xlabel('Program') 
    plt.ylabel('Speedup compared to ground truth')
    plt.legend()
    plt.savefig("figure/fillbetween_spgt.png")


def find_files(mean1, mean2, lower1, lower2, upper1, upper2, total1, total2):
    f_var = []
    f_mean = []
    # print best 3 actions / speedup / reward
    print("Some files with large variance:")
    for f in lower1["rewd"]:
        gap1 = upper1["rewd"][f] - lower1["rewd"][f]
        gap2 = upper2["rewd"][f] - lower2["rewd"][f]
        if (gap2 - 100*gap1 > 0.3):
            f_var.append(f)
            print("--", f, gap2-gap1, total1["pred"][f], total2["pred"][f])
            print("---- neurovec reward: ", total1["rewd"][f])
            print("---- neurovec spgt: ", total1["spgt"][f])
            print("---- graphsage reward: ", total2["rewd"][f])
            print("---- graphsage reward: ", total2["spgt"][f])
    print("Some files with large mean difference:")
    for f in lower1["rewd"]:
        gap1 = abs(mean1["rewd"][f])
        gap2 = abs(mean2["rewd"][f])
        if (gap1 - gap2 > 0.3):
            f_mean.append(f)
            print("**", f, gap1-gap2, total1["pred"][f], total2["pred"][f])
            print("**** neurovec reward: ", total1["rewd"][f])
            print("**** neurovec spgt: ", total1["spgt"][f])
            print("**** graphsage reward: ", total2["rewd"][f])
            print("**** graphsage reward: ", total2["spgt"][f])

    print("files with large variance and large mean difference: ", list(set(f_var).intersection(f_mean)))
    pred1 = []
    pred2 = []
    rewd1 = []
    rewd2 = []
    spgt1 = []
    spgt2 = []
    for f in list(set(f_var).intersection(f_mean)):
        print(f, ": ", total1["pred"][f], total2["pred"][f], total1["base"][f])
        print("--** neurovec reward: ", total1["rewd"][f])
        print("--** neurovec spgt: ", total1["spgt"][f])
        print("--** graphsage reward: ", total2["rewd"][f])
        print("--** graphsage reward: ", total2["spgt"][f])
        pred1.append(total1["pred"][f][0])
        pred2.append(total2["pred"][f][0])
        rewd1.append(np.mean(total1["rewd"][f]))
        rewd2.append(np.mean(total2["rewd"][f]))
        spgt1.append(np.mean(total1["spgt"][f]))
        spgt2.append(np.mean(total2["spgt"][f]))


    x_pos = list(set(f_var).intersection(f_mean))
    print(spgt1)
    br1 = np.arange(len(x_pos))
    barWidth = 0.25
    br2 = [x + barWidth for x in br1]

    # Make the plot
    plt.figure(2)
    plt.bar(br1, pred1, color ='r', width = barWidth, edgecolor ='grey', label ='neurovec')
    plt.bar(br2, pred2, color ='g', width = barWidth, edgecolor ='grey', label ='grpahsage')
    plt.xlabel('files')
    plt.ylabel('action')
    #plt.legend("Action space(VF/IF)")
    plt.savefig("figure/bar_action.png")

    plt.figure(3)
    plt.bar(br1, rewd1, color ='r', width = barWidth, edgecolor ='grey', label ='neurovec')
    plt.bar(br2, rewd2, color ='g', width = barWidth, edgecolor ='grey', label ='grpahsage')
    plt.xlabel('files')
    plt.ylabel('reward')
    #plt.legend("Action space(VF/IF)")
    plt.savefig("figure/bar_reward.png")

    plt.figure(4)
    plt.bar(br1, spgt1, color ='r', width = barWidth, edgecolor ='grey', label ='neurovec')
    plt.bar(br2, spgt2, color ='g', width = barWidth, edgecolor ='grey', label ='grpahsage')
    plt.xlabel('files')
    plt.ylabel('speedup (ground truth)')
    #plt.legend("Action space(VF/IF)")
    plt.savefig("figure/bar_spgt.png")

def find_statistics(total1, total2):
    mean1 = {"rewd": {}, "spgt": {}, "spo3": {}}
    upper1 = {"rewd": {}, "spgt": {}, "spo3": {}}
    lower1 = {"rewd": {}, "spgt": {}, "spo3": {}}
    mean2 = {"rewd": {}, "spgt": {}, "spo3": {}}
    upper2 = {"rewd": {}, "spgt": {}, "spo3": {}}
    lower2 = {"rewd": {}, "spgt": {}, "spo3": {}}
    for sta in total1.keys():
        if (sta == "rewd" or sta == "spgt" or sta == "spo3"):
            for f in total1[sta]:
                mean1[sta][f] = np.mean(total1[sta][f])
                lower1[sta][f] = np.min(total1[sta][f])
                upper1[sta][f] = np.max(total1[sta][f])
                mean2[sta][f] = np.mean(total2[sta][f])
                lower2[sta][f] = np.min(total2[sta][f])
                upper2[sta][f] = np.max(total2[sta][f])

    return mean1, mean2, lower1, lower2, upper1, upper2

# "results/graphsagegcn128-256-"
# "results/graphsage_"
total1, total2 = parsing2("results/neurovec", "results/graphsage_1m_10k_", 4)

mean1, mean2, lower1, lower2, upper1, upper2 = find_statistics(total1, total2)

# scatter plot (reward/action)

#scatter_plots(total1, total2)

# reward / speedup line plot with confidence bands
# fill_between
# ax.fill_between(x, y1, y2, alpha=0.2)

fillbetween(mean1, mean2, lower1, lower2, upper1, upper2)

#print(len(sorted_spgt1))

find_files(mean1, mean2, lower1, lower2, upper1, upper2, total1, total2)






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
