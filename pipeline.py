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
            # x_hat = F.dropout(x_hat, self.dropout, training=self.training)  # Apply dropout
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
    neg_edge_index = negative_sampling(edge_index=pos_edge_index, num_nodes=data.num_nodes)

    # Concatenate positive and negative edges
    total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

    link_logits = model(data.x, total_edge_index)
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
    pos_edge_index = data.edge_index
    neg_edge_index = negative_sampling(edge_index=pos_edge_index, num_nodes=data.num_nodes)

    total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    link_logits = model(data.x, total_edge_index)

    link_labels = torch.cat([torch.ones(pos_edge_index.size(1)), 
                             torch.zeros(neg_edge_index.size(1))], dim=0).to(device)

    probs = torch.sigmoid(link_logits).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    labels = link_labels.cpu().numpy()

    auc_roc = roc_auc_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    
    return auc_roc, precision, recall, f1

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

from torch.optim.lr_scheduler import StepLR

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

print('GCN initialized')

num_epochs = 1000

# train on GPU if available

model = model.to(device)
loss_fn = loss_fn.to(device)


# Training loop with error handling and metric logging
train_losses, val_metrics = [], []
for epoch in range(num_epochs):
    try:
        train_loss = train(model, train_data, optimizer, loss_fn)
        auc_roc, precision, recall, f1 = evaluate(model, val_data)
        train_losses.append(train_loss)
        val_metrics.append((auc_roc, precision, recall, f1))
        print(f'Epoch {epoch}: Loss: {train_loss:.4f}, AUC-ROC: {auc_roc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
        scheduler.step()

    except Exception as e:
        print(f"Exception encountered: {e}")
        break

# Writing training loss and validation metrics to a file
with open("training_validation_metrics.txt", "w") as file:
    file.write("Epoch, Training Loss, AUC-ROC, Precision, Recall, F1-Score\n")
    for i, epoch in enumerate(num_epochs):
        train_loss = train_losses[i]
        auc_roc, precision, recall, f1 = val_metrics[i]
        file.write(f"{epoch}, {train_loss:.4f}, {auc_roc:.4f}, {precision:.4f}, {recall:.4f}, {f1:.4f}\n")

# Plot training loss and validation metrics
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
val_metrics = np.array(val_metrics)
metrics_labels = ['AUC-ROC', 'Precision', 'Recall', 'F1-Score']
for i, label in enumerate(metrics_labels):
    plt.plot(epochs, val_metrics[:, i], label=label)
plt.title('Validation Metrics Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.legend()
plt.show()