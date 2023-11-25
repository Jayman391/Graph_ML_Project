import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# use cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pyg_graph = torch.load('pyg_graph_with_features.pt')

# only keep edge_intex, x, and num_nodes
graph = Data(x=pyg_graph.x, edge_index=pyg_graph.edge_index, num_nodes=pyg_graph.num_nodes)

transform = RandomLinkSplit(is_undirected=True)
train_data, val_data, test_data = transform(graph)

print('data preprocessed')

# Define the GCN Model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_groups, num_layers, dropout):
        super(GCN, self).__init__()
        # Initialize convolutional layers
        self.convs = torch.nn.ModuleList([GCNConv(input_dim, hidden_dim)])
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        # Output layer for group prediction
        self.convs.append(GCNConv(hidden_dim, num_groups))  

        # Initialize batch normalization layers
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout

    # Encoding function to generate node embeddings
    def encode(self, x, edge_index):
        x_hat = x
        x_hat = self.convs[0](x_hat, edge_index)  # First convolutional layer
        for i in range(1, len(self.convs) - 1):  # Adjusted loop
            x_hat = self.bns[i](x_hat)  # Apply batch normalization
            x_hat = F.relu(x_hat)  # Apply ReLU
            x_hat = F.dropout(x_hat, self.dropout, training=self.training)  # Apply dropout
            x_hat = self.convs[i](x_hat, edge_index)  # Apply next convolutional layer
        return x_hat  # Return the transformed features


    # Decoding function to compute edge scores
    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    # Forward pass
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        link_logits = self.decode(z, edge_index)
        return link_logits

# Function to train the model
def train(model, data, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()

    pos_edge_index = data.edge_index
    neg_edge_index = negative_sampling(edge_index=pos_edge_index, num_nodes=data.x.size(0))

    link_logits = model(data.x, data.edge_index)
    link_labels = torch.cat([torch.ones(pos_edge_index.size(1)), 
                             torch.zeros(neg_edge_index.size(1))], dim=0).to(device)

    loss = loss_fn(link_logits, link_labels)
    loss.backward()
    optimizer.step()
    return loss.item()

# Function to evaluate the model
@torch.no_grad()
def evaluate(model, data):
    model.eval()
    total_auc_roc, total_precision, total_recall, total_f1 = 0, 0, 0, 0

    pos_edge_index = data.edge_index
    neg_edge_index = negative_sampling(edge_index=pos_edge_index, num_nodes=data.x.size(0))

    pos_link_logits = model.encode(data.x, pos_edge_index)
    neg_link_logits = model.encode(data.x, neg_edge_index)
    pos_probs = torch.sigmoid(pos_link_logits).numpy()
    neg_probs = torch.sigmoid(neg_link_logits).numpy()

    probs = np.concatenate([pos_probs, neg_probs])
    labels = np.concatenate([np.ones(pos_probs.shape[0]), np.zeros(neg_probs.shape[0])])

    auc_roc = roc_auc_score(labels, probs)
    preds = (probs > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    
    total_auc_roc += auc_roc
    total_precision += precision
    total_recall += recall
    total_f1 += f1

    return total_auc_roc, total_precision , total_recall , total_f1 


input_dim = graph.x.shape[1]
print(f'input_dim : {input_dim}')
hidden_dim = 64

groups = pd.read_json("babynamesDB_groups.json")
groups = groups.query("num_users_stored > 3")
group_ids = groups["_id"].to_list()
num_groups = len(group_ids)
num_layers = 3
dropout = 0.5


# Initialize the model
model = GCN(input_dim, hidden_dim, num_groups, num_layers, dropout)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print('GCN initialized')

num_epochs = 100

# train on GPU if available

model = model.to(device)
loss_fn = loss_fn.to(device)

loss = train(model, train_data, optimizer, loss_fn)

print(loss)

# Training loop with error handling
for epoch in range(num_epochs):
    try:
        loss = train(model, train_data, optimizer, loss_fn)
        print(f'Epoch {epoch}: Loss: {loss:.4f}')
    except KeyError as e:
        print(f"KeyError encountered: {e}")
        # Optional: Add debugging or logging statements here
        break

# Evaluate the model
auc_roc, precision, recall, f1 = evaluate(model, test_data)
print(f"AUC-ROC: {auc_roc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Writing metrics to a file
with open("evaluation_metrics.txt", "w") as file:
    file.write(f"AUC-ROC: {auc_roc:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1-Score: {f1:.4f}\n")