import networkx as nx
import pandas as pd
import numpy as np
import os
import random
import dgl
import json

import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from sklearn.model_selection import train_test_split

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score

from stellargraph import globalvar
from dataset import GraphDataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph import StellarGraph
from stellargraph import IndexedArray


dataset = GraphDataset()
filename, graph, label = dataset[0]
print(filename, label)


graphs = dataset.graphs
filenames = dataset.filenames

from stellargraph import datasets
dataset = datasets.Cora()
G, node_subjects = dataset.load()
print("node_subjects = ", node_subjects)
print("node_subjects index = ", node_subjects)


number_of_walks = 1
length = 5
batch_size = 50
epochs = 30
num_samples = [10, 10]
layer_sizes = [50, 50]
feature_embedding = {}

for graph_idx in range(len(graphs)):
    graph = graphs[graph_idx]
    filename = filenames[graph_idx]
    #print("G = ", graph.ndata["m"][1])
    nx_g = dgl.to_networkx(graph, node_attrs=["m"])
    nxg = nx.Graph()
    src = []
    dst = []
    nodes = nx_g.nodes
    for (u, v, w) in nx_g.edges:
        src.append(u)
        dst.append(v)
    edges = pd.DataFrame(
        {"source": src, "target": dst}    
    )
    feature_array = graph.ndata["m"].numpy()
    nodes = IndexedArray(feature_array, index=nodes)

    G = StellarGraph(nodes, edges)
    
    nodes = list(G.nodes())

    unsupervised_samples = UnsupervisedSampler(
        G, nodes=nodes, length=length, number_of_walks=number_of_walks
    )

    generator = GraphSAGELinkGenerator(G, batch_size, num_samples)
    train_gen = generator.flow(unsupervised_samples)

    graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2"
    )

    x_inp, x_out = graphsage.in_out_tensors()

    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)


    model = keras.Model(inputs=x_inp, outputs=prediction)

    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-3),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy],
    )

    history = model.fit(
        train_gen,
        epochs=epochs,
        verbose=1,
        use_multiprocessing=False,
        workers=4,
        shuffle=True,
    )

    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

    node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(nodes)

    node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)
    feature_embedding[filename] = node_embeddings
    print("node_embeddings = ", node_embeddings)

with open('embeddings.json', 'w') as f:
    json.dump(feature_embedding, f) 


