import os

for i in range(5):
    os.system("python3 emb_gcn_oneModel.py > results/supervised_graphsage_"+str(i))
