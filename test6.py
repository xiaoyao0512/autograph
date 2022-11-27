import pickle
import json


f = open('lore_runtimes.pickle', 'rb')
runtimes = pickle.load(f)    
f.close()
f = open('lore_runtimes_none_pragma.pickle', 'rb')
base_runtimes = pickle.load(f)    
f.close()

f = open('embeddings2_graphsage_gcn128.json', 'rb')
exeT = json.load(f)
f.close()

print(exeT)

cnt = 0
#exeT = times["feat"]
for f in exeT:
    print(len(f))

print(cnt)
