import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load your graph and perform train/test split
pyg_graph = torch.load('pyg_graph.pt')
transform = RandomLinkSplit(is_undirected=True)
train_data, val_data, test_data = transform(pyg_graph)

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
def train(model, data, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()

    # Positive samples from the graph
    pos_edge_index = train_data.edge_index

    # Negative sampling for training
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.edge_index.shape[1],
        num_neg_samples=pyg_graph.num_edges - train_data.num_edges
        )

    link_logits = model(data)
    link_labels = torch.cat([torch.ones(pos_edge_index.size(1)), 
                             torch.zeros(neg_edge_index.size(1))], dim=0)
    loss = loss_fn(link_logits, link_labels)
    loss.backward()
    optimizer.step()
    return loss.item()

# Function to evaluate the model
@torch.no_grad()
def evaluate(model, data):
    model.eval()

    # Positive and negative samples for testing
    pos_edge_index = test_data.edge_index
    neg_edge_index = negative_sampling(
        edge_index=test_data.edge_index,
        num_nodes=test_data.edge_index.shape[1],
        num_neg_samples=pyg_graph.num_edges - test_data.num_edges
        )


    pos_link_logits = model.encode(data.x, pos_edge_index)
    neg_link_logits = model.encode(data.x, neg_edge_index)
    pos_probs = torch.sigmoid(pos_link_logits).cpu().numpy()
    neg_probs = torch.sigmoid(neg_link_logits).cpu().numpy()

    # Combine positive and negative predictions
    probs = np.concatenate([pos_probs, neg_probs])
    labels = np.concatenate([np.ones(pos_probs.shape[0]), np.zeros(neg_probs.shape[0])])
    
    # Calculate evaluation metrics
    auc_roc = roc_auc_score(labels, probs)
    preds = (probs > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    conf_matrix = confusion_matrix(labels, preds)

    return auc_roc, precision, recall, f1, conf_matrix

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

# Initialize the model
model = GCN(input_dim, hidden_dim, num_groups, num_layers, dropout)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    loss = train(model, train_data, optimizer, loss_fn)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}')

# Evaluate the model
auc_roc, precision, recall, f1, conf_matrix = evaluate(model, test_data)
print(f"AUC-ROC: {auc_roc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Visualize the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
