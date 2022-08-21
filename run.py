import os


for i in range(5):
    os.system("python3 autovec_test.py > results/neurovec_1m5_"+str(i))
