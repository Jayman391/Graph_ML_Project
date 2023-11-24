import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# use cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load embeddings and set model parameters
embeddings = pd.read_csv('data/full_users_embeddings_2.csv')
input_dim = embeddings.shape[1]
hidden_dim = 64
groups = pd.read_json("babynamesDB_groups.json")
groups = groups.query("num_users_stored > 3")
group_ids = groups["_id"].to_list()
num_groups = len(group_ids)
num_layers = 2
dropout = 0.5

# Load your graph and perform train/test split
pyg_graph = torch.load('pyg_graph.pt')

assert pyg_graph.num_nodes == embeddings.shape[0], "Mismatch in the number of nodes and embeddings rows"

pyg_graph.x = torch.tensor(embeddings.values, dtype=torch.float)

transform = RandomLinkSplit(is_undirected=True)
train_data, val_data, test_data = transform(pyg_graph)


train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


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
        for i in range(len(self.convs)-1):
            x_hat = self.convs[i](x_hat, edge_index)
            x_hat = self.bns[i](x_hat)
            x_hat = F.relu(x_hat)
            x_hat = F.dropout(x_hat, self.dropout, training=self.training)
        return self.convs[-1](x_hat, edge_index)

    # Decoding function to compute edge scores
    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    # Forward pass
    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        link_logits = self.decode(z, data.edge_index)
        return link_logits

# Function to train the model
def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()

        # Positive and Negative sampling for each batch
        pos_edge_index = batch.edge_index
        neg_edge_index = negative_sampling(edge_index=pos_edge_index, num_nodes=batch.x.size(0))

        # Forward pass
        link_logits = model(batch)
        link_labels = torch.cat([torch.ones(pos_edge_index.size(1)), 
                                 torch.zeros(neg_edge_index.size(1))], dim=0).to(device)

        # Loss calculation
        loss = loss_fn(link_logits, link_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Function to evaluate the model
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_auc_roc, total_precision, total_recall, total_f1 = 0, 0, 0, 0
    for batch in loader:
        # Positive and Negative sampling for each batch
        pos_edge_index = batch.edge_index
        neg_edge_index = negative_sampling(edge_index=pos_edge_index, num_nodes=batch.x.size(0))

        pos_link_logits = model.encode(batch.x, pos_edge_index)
        neg_link_logits = model.encode(batch.x, neg_edge_index)
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
    
    num_batches = len(loader)
    return total_auc_roc / num_batches, total_precision / num_batches, total_recall / num_batches, total_f1 / num_batches



# Initialize the model
model = GCN(input_dim, hidden_dim, num_groups, num_layers, dropout)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100

# train on GPU if available

model = model.to(device)
loss_fn = loss_fn.to(device)


# Training loop with error handling
for epoch in range(num_epochs):
    try:
        loss = train(model, train_loader, optimizer, loss_fn)
        print(f'Epoch {epoch}: Loss: {loss:.4f}')
    except KeyError as e:
        print(f"KeyError encountered: {e}")
        # Optional: Add debugging or logging statements here
        break

# Evaluate the model
auc_roc, precision, recall, f1 = evaluate(model, test_loader)
print(f"AUC-ROC: {auc_roc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Writing metrics to a file
with open("evaluation_metrics.txt", "w") as file:
    file.write(f"AUC-ROC: {auc_roc:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1-Score: {f1:.4f}\n")