import os

for i in range(5):
    os.system("python3 lore_emb_gnn_twoModels.py > results/lore_supervised_graphsage_"+str(i))
