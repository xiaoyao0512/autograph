import json
from collections import OrderedDict
from itertools import accumulate
import numpy as np
import matplotlib.pyplot as plt

with open('lore_full_autograph.json', 'rb') as f:
    autograph = json.load(f)

with open('lore_full_neurovec.json', 'rb') as f:
    neurovec = json.load(f)

with open('lore_brute_force.json', 'rb') as f:
    bruteforce = json.load(f)




