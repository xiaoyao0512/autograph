#from dataset import GraphDataset
#import matplotlib.pyplot as plt
import networkx as nx
import dgl
import torch
###############################################################################
# The return type of :func:`dgl.batch` is still a graph. In the same way, 
# a batch of tensors is still a tensor. This means that any code that works
# for one graph immediately works for a batch of graphs. More importantly,
# because DGL processes messages on all nodes and edges in parallel, this greatly
# improves efficiency.
#
# Graph classifier
# ----------------
# Graph classification proceeds as follows.
#
# .. image:: https://data.dgl.ai/tutorial/batch/graph_classifier.png
#
# From a batch of graphs, perform message passing and graph convolution
# for nodes to communicate with others. After message passing, compute a
# tensor for graph representation from node (and edge) attributes. This step might 
# be called readout or aggregation. Finally, the graph 
# representations are fed into a classifier :math:`g` to predict the graph labels.
#
# Graph convolution layer can be found in the ``dgl.nn.<backend>`` submodule.

from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GatedGraphConv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.utils.data import DataLoader
#from sklearn.model_selection import KFold as kfold
#import torch as th
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
#from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.annotations import override
import gym
import numpy as np
from ray.rllib.utils import try_import_tf
#from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
#from utility import load_graphs
from envs.neurovec_gnn import EMBED
import pickle
from ray.rllib.utils.typing import Dict, TensorType, List
import os
from utility import delete_zeros, intlist2str


tf = try_import_tf()



class GCNClassifier(TorchModelV2, nn.Module):
    def __init__(self, obs_space,
                 action_space, num_outputs: int,
                 model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        #self.mode = mode
        self.in_dim = EMBED#model_config.get("in_dim")
        #print("action_space = ", action_space)
        #print("obs_space = ", obs_space)
        #print("num_outputs = ", num_outputs)
        #print("name = ", name)
        
        graphs = {}
        for f in glob.glob("/home/yao/Simulator/drl-vec/json-small/*.json"):
            fn = f.split('/')[-1].split('.')[0]
            fn_c = fn + '.c'
            with open(f) as fh:
                g = nx.readwrite.json_graph.node_link_graph(json.load(fh))
                # calculate the graph features
                #print(g.nodes)
                g = dgl.from_networkx(g)
            graphs[fn_c] = g

        custom_config = model_config.get("custom_model_config")
        #print("custom_config = ", custom_config)
        self.hidden_dim = custom_config.get("hidden_dim")
        self.num_outputs = num_outputs
        self.num_layers = custom_config.get("num_layers")
        #print("model_config = ", model_config)
        #print("num_layers = ", self.num_layers)
        self.conv = []
        self._logits = None
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

        if (self.num_layers == 1):
            self.conv.append(GraphConv(self.in_dim, num_outputs))
        else:
            for i in range(self.num_layers):
                if (i == 0):
                    self.conv.append(GraphConv(self.in_dim, self.hidden_dim))
                else:
                    self.conv.append(GraphConv(self.hidden_dim, num_outputs))
        #self.conv1 = GraphConv(in_dim, hidden_dim)
        #self.conv2 = GraphConv(hidden_dim, hidden_dim) # graph attention network / gated GNN
        #self.conv3 = GraphConv(hidden_dim, hidden_dim) # graph attention network / gated GNN
        self.value_branch = nn.Linear(num_outputs, 1)


    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        # 
        obs_dict = input_dict["obs"] 
        filename = delete_zeros("filename", np.argmax(obs_dict["filename"].values[0], axis=1))
        filename = intlist2str(filename)
        #print("filename = ", filename)
        if (filename == ""):
            logits = torch.zeros(1, self.num_outputs)            
            self._features = logits
            return logits, state
        #print("after resetting in GNN")
        #print("in forward, input_dict = ", input_dict)
        #print("state = ", state)
        #print("in forward, obs = ", input_dict['obs'])
        #print("in forward, filename = ", input_dict["obs"]['filename'].values[0,0])
        #print("in forward, obs [dim] = ", input_dict['obs']['current_pragma_idx'].shape)
        #print("in forward, obs [filename] = ", input_dict['obs']['filename'])
        #print("in forward, obs [vf_idx] = ", input_dict['obs']['vf_idx'].shape)
        #print("in forward, obs [if_idx] = ", input_dict['obs']['if_idx'].shape)
        #print("in forward, obs [obs_flat] = ", input_dict['obs_flat'])
       
        vf_idx = np.argmax(obs_dict['vf_idx'], axis=1)
        vf_idx = int(vf_idx[0])
        if_idx = np.argmax(obs_dict['if_idx'], axis=1)
        if_idx = int(if_idx[0])  
        current_pragma_idx = np.argmax(obs_dict['current_pragma_idx'], axis=1)  
        current_pragma_idx = int(current_pragma_idx[0])  
        
        #print("in forward, filename values = ", obs_dict["emb"].values)
        #print("in forward, filename values values", obs_dict["emb"].values.values)
        #print("in forward, filename lengths", obs_dict["emb"].lengths)
        #print("in forward, filename values lengths", obs_dict["emb"].values.shape)
        emb = torch.tensor(delete_zeros("emb", obs_dict["emb"].values[0]))
        #print("in forward, emb values = ", emb)


        
        #pragma_idx = obs_dict['current_pragma_idx']
        #filename = obs_dict['filename']
        
        #feat = obs_dict['emb']
        #print("obs_dict = ", obs_dict)
        #print("vf_idx = ", vf_idx)
        #print("pragma_idx = ", pragma_idx)
        #print("filename = ", filename.values[0,0,0])
        #print("filename = ", filename)
        #print("current_pragma_idx = ", current_pragma_idx)
        #print("vf_idx = ", vf_idx)
        #print("if_idx = ", if_idx)
        #g = self.graphs[filename][current_pragma_idx][vf_idx][if_idx]
        g = self.graphs[filename]
        #print("g = ", g.number_of_nodes())
        #g = dgl.from_networkx(g)
        #print("emb = ", emb)
        #print("emb shape = ", emb.shape)
        h = emb
        #print("h = ", h)
        #print("h shape = ", h.shape)        
        g = dgl.add_self_loop(g)
        
        # Perform graph convolution and activation function.
        for conv in self.conv:
            h = F.relu(conv(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        logits = dgl.mean_nodes(g, 'h')        
        self._features = logits
        return logits, state


    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return self.value_branch(self._features).squeeze(1)

