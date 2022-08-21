import os

for i in range(5):
    os.system("python3 code2vec_predictor.py > results/supervised_code2vec_"+str(i))
