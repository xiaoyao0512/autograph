from umap.parametric_umap import ParametricUMAP
import matplotlib.pyplot as plt
import json
from random import shuffle
import numpy as np
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

input_size = 64
#with open("features.json") as f:
#with open('features_graphsage_semantics_128max.json') as f:
#with open('features_gcn_degree.json') as f:
#with open('features2_graphsage_gcn'+str(input_size)+'.json') as f:
#with open('features2_gcn_nn.json') as f:
with open('features2_graphsage_gcn128.json') as f:
    features = json.load(f)

feats = features["feat"]
labels = np.array(features["labels"])

obs = []
for feat in feats:
    obs.append(feat[0])

component_size = 2
embedder = ParametricUMAP(n_epochs = 300, verbose=True, n_components=component_size)
embedding = embedder.fit_transform(obs)


if component_size == 2:
    fig, ax = plt.subplots( figsize=(8, 8))
    sc = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels.astype(int),
        cmap="Spectral",
        #s=0.1,
        #alpha=0.5,
        rasterized=True,
    )
    ax.axis('equal')
elif component_size == 3:
    ax = plt.axes(projection='3d')
    sc = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        embedding[:, 2],
        c=labels.astype(int),
        cmap="Spectral",
        s=1,
        #alpha=0.5,
        rasterized=True,
    ) 
    ax.axis('auto')   
ax.set_title("UMAP in Tensorflow embedding", fontsize=20)
plt.colorbar(sc, ax=ax);
plt.savefig("test_umap.png")


#embedder._history.keys()

#fig, ax = plt.subplots()
#ax.plot(embedder._history['loss'])
#ax.set_ylabel('Cross Entropy')
#ax.set_xlabel('Epoch')

'''
embedder = ParametricUMAP(n_epochs = 200, verbose=True)

embedding = embedder.fit_transform(train_images)

fig, ax = plt.subplots( figsize=(8, 8))
sc = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=Y_train.astype(int),
    cmap="tab10",
    s=0.1,
    alpha=0.5,
    rasterized=True,
)
ax.axis('equal')
ax.set_title("UMAP in Tensorflow embedding", fontsize=20)
plt.colorbar(sc, ax=ax);fig, ax = plt.subplots( figsize=(8, 8))
sc = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=Y_train.astype(int),
    cmap="tab10",
    s=0.1,
    alpha=0.5,
    rasterized=True,
)
ax.axis('equal')
ax.set_title("UMAP in Tensorflow embedding", fontsize=20)
plt.colorbar(sc, ax=ax);
'''
