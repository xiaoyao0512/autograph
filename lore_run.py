import os

l = [3, 4]
for i in l:
    os.system("python3 lore_autovec_graphsage.py > results/lore_graphsage_1m_"+str(i))


