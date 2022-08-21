import pickle, json
import numpy as np
from torch.utils.data import Dataset
import networkx as nx
import dgl, torch
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GatedGraphConv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold as kfold

class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier, self).__init__()
        #self.mode = mode
        self.conv1 = GraphConv(in_dim, hidden_dim)
        #self.conv2 = GraphConv(hidden_dim, hidden_dim) # graph attention network / gated GNN
        #self.conv3 = GraphConv(hidden_dim, hidden_dim) # graph attention network / gated GNN
        self.classify1 = nn.Linear(hidden_dim, n_classes)
        self.classify2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # node feature
        #print(g.ndata['w'])
        #h = g.ndata['m'].view(-1,1).float()
        h = g.ndata['m'].float()
        g = dgl.add_self_loop(g)
        #em
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        #h = F.relu(self.conv2(g, h))
        #h = F.relu(self.conv3(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify1(hg), self.classify2(hg)

