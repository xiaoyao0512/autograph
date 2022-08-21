import os

l = [3, 4]
for i in l:
    os.system("python3 lore_autovec_neurovec.py > results/lore_neurovec_1m_"+str(i))
