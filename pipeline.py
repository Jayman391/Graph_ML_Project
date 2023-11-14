import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from scipy import sparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

groups = pd.read_json("babynamesDB_groups.json")
groups = groups.query("num_users_stored > 3")
group_ids = groups["_id"].to_list()

# Efficiently handle embeddings
embeddings = pd.read_csv('data/full_users_embeddings.csv')
# keep only the last 768 columns
embeddings = embeddings.iloc[:, -768:]
print('embeddings shape: ', embeddings.shape)
# Using sparse matrix for embeddings
sparse_embeddings = sparse.lil_matrix((len(embeddings) + len(group_ids), embeddings.shape[1] + len(group_ids)))
print('sparse embeddings shape: ', sparse_embeddings.shape)
# Populate the matrix
sparse_embeddings[:len(embeddings), :embeddings.shape[1]] = embeddings.values

# One-hot encoding for groups
group_one_hot = sparse.eye(len(group_ids))
sparse_embeddings[len(embeddings):, -len(group_ids):] = group_one_hot
print(group_one_hot.shape)

embeddings = pd.DataFrame(sparse_embeddings.todense(), columns=embeddings.columns.to_list() + group_ids)
# set column names to range of integers
embeddings.rename(columns={col: i for i, col in enumerate(embeddings.columns)}, inplace=True)
G = nx.read_edgelist('graph.edgelist')

attrs = {}
for node in G.nodes():
    data = embeddings.iloc[int(node)].to_dict()
    # if there are fields in embeddings but not in data,add and fill them with 0
    for field in embeddings.columns.to_list():
        if field not in data:
            data[field] = 0
    attrs[int(node)] = data 

nx.set_node_attributes(G, attrs)    
print('node attributes set')

pyg_graph = from_networkx(G)


print('graph converted to pytorch geometric graph')

torch.save(pyg_graph, 'pyg_graph.pt')
data = train_test_split_edges(pyg_graph)

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_groups, num_layers, dropout):
        super(GCN, self).__init__()
        
        # Convolutional layers
        self.convs = torch.nn.ModuleList([GCNConv(input_dim, hidden_dim)])
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, num_groups))  # Output layer for group prediction

        # Batch normalization layers
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout

    def encode(self, x, edge_index):
        x_hat = x
        for i in range(len(self.convs)-1):
            x_hat = self.convs[i](x_hat, edge_index)
            x_hat = self.bns[i](x_hat)
            x_hat = F.relu(x_hat)
            x_hat = F.dropout(x_hat, self.dropout, training=self.training)
        return self.convs[-1](x_hat, edge_index)

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def forward(self, data):
        z = self.encode(data.x, data.train_pos_edge_index)
        link_logits = self.decode(z, data.train_pos_edge_index)
        return link_logits

def train(model, data, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    link_logits = model(data)
    link_labels = torch.cat([torch.ones(data.train_pos_edge_index.size(1)), 
                             torch.zeros(data.train_neg_edge_index.size(1))], dim=0)
    loss = loss_fn(link_logits, link_labels)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data):
    model.eval()
    pos_link_logits = model.encode(data.x, data.test_pos_edge_index)
    neg_link_logits = model.encode(data.x, data.test_neg_edge_index)
    pos_link_probs = F.softmax(pos_link_logits, dim=1)
    neg_link_probs = F.softmax(neg_link_logits, dim=1)
    # Here you can calculate the evaluation metrics like AUC, Accuracy, etc.
    return pos_link_probs, neg_link_probs

input_dim = embeddings.shape[1] - 4  # Minus 4 to exclude '_id', 'one_hot', and two additional columns
hidden_dim = 64  # Example value, you may need to tune this
num_groups = len(group_ids)  # Assuming this is the number of groups
num_layers = 2   # Number of layers in GCN
dropout = 0.5    # Dropout rate

model = GCN(input_dim, hidden_dim, num_groups, num_layers, dropout)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Learning rate may need tuning
num_epochs = 100  # Number of epochs to train

for epoch in range(num_epochs):
    loss = train(model, data, optimizer, loss_fn)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}')

pos_link_probs, neg_link_probs = test(model, data)
