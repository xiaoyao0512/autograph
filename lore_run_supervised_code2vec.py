import os

for i in range(5):
    os.system("python3 lore_code2vec_predictor.py > lore_supervised_code2vec_"+str(i))
