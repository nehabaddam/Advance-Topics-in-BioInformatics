import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load Data with Dask and optimize memory usage
data = dd.read_parquet('sequence_features_val.parquet')

# Drop 'ID' column if present and optimize data types
data = data.drop(columns=['ID'], errors='ignore').astype('float32')
data = data.compute()  # Convert to a Pandas DataFrame

data = data.dropna(axis=1, thresh=int(0.6 * len(data))) 
data = data.fillna(data.mean())
# Assign labels directly
num_positive_samples = 1249857
# data = pd.concat([data.head(10000), data.tail(10000)]).reset_index(drop=True)
# num_positive_samples =10000
labels = pd.Series([1] * num_positive_samples + [0] * (len(data) - num_positive_samples), name='label')

# Check if the adjacency matrix exists; if not, compute and save it
adj_matrix_path = 'adj_matrix_sparse.npz'


if not os.path.exists(adj_matrix_path):
    print("Computing adjacency matrix...")

    # Normalize data for cosine similarity
    data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)

    # Initialize a sparse adjacency matrix
    num_samples = data.shape[0]
    k = 10  # Number of top neighbors to keep
    adj_matrix = sp.lil_matrix((num_samples, num_samples))

    for i in range(len(data_normalized)):
        similarities = cosine_similarity(data_normalized.iloc[i].values.reshape(1, -1), data_normalized.values)[0]
        top_indices = np.argsort(-similarities)[1:k+1]  # Exclude the self-correlation (node itself)
        adj_matrix[i, top_indices] = 1  # Set the top-k similarities as 1 for adjacency

    # Save the sparse adjacency matrix
    sp.save_npz(adj_matrix_path, adj_matrix.tocsr())
    print("Adjacency matrix saved as 'adj_matrix_sparse.npz'.")
else:
    print("Loading existing adjacency matrix...")
    adj_matrix = sp.load_npz(adj_matrix_path)

adj_matrix_coo = adj_matrix.tocoo()
edge_index = torch.tensor(np.vstack((adj_matrix_coo.row, adj_matrix_coo.col)), dtype=torch.long)

# Load feature data and labels
features = torch.tensor(data.values, dtype=torch.float32)
labels = torch.tensor(labels.values, dtype=torch.long)

# Create PyTorch Geometric Data object
data = Data(x=features, edge_index=edge_index, y=labels)

# Define a simple GCN model for binary classification
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize the model, optimizer, and loss function
input_dim = features.shape[1]  # Number of features
hidden_dim = 64  # Number of hidden units (can be tuned)
output_dim = 2  # Binary classification

model = GCN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()

# Training loop
model.train()
epochs = 50

for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:  # Print loss every 10 epochs
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Evaluation
model.eval()
_, y_pred = model(data).max(dim=1)
y_true = data.y.numpy()
y_pred_binary = y_pred.numpy()

# Compute metrics
print("Evaluating model...")
accuracy = accuracy_score(y_true, y_pred_binary)
precision = precision_score(y_true, y_pred_binary)
recall = recall_score(y_true, y_pred_binary)
f1 = f1_score(y_true, y_pred_binary)

# Display results
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')

# Save results to a CSV file
results_df = pd.DataFrame({
    'Model': ['GCN'],
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1-Score': [f1]
})
results_df.to_csv('Final_results_GCN_all.csv', index=False)

# Save the model
torch.save(model.state_dict(), 'gcn_model_all.pth')
print("GCN model trained, results saved, and model serialized.")