import numpy as np

# Khởi tạo ma trận chéo D
D = np.array([
    [3, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 2, 0],
    [0, 0, 0, 2]
])

# Tính nghịch đảo của ma trận D
D_inv = np.linalg.inv(D)
print("Inverse of D:")
print(D_inv)
# Output:
# [[0.33333333, 0.        , 0.        , 0.        ],
#  [0.        , 1.        , 0.        , 0.        ],
#  [0.        , 0.        , 0.5       , 0.        ],
#  [0.        , 0.        , 0.        , 0.5       ]]

# Tính nghịch đảo của ma trận (D + I), với I là ma trận đơn vị
D_plus_I_inv = np.linalg.inv(D + np.identity(4))
print("Inverse of D + I:")
print(D_plus_I_inv)
# Output:
# [[0.25      , 0.        , 0.        , 0.        ],
#  [0.        , 0.5       , 0.        , 0.        ],
#  [0.        , 0.        , 0.33333333, 0.        ],
#  [0.        , 0.        , 0.        , 0.33333333]]

# Khởi tạo ma trận A
A = np.array([
    [1, 1, 1, 1],
    [1, 1, 0, 0],
    [1, 0, 1, 1],
    [1, 0, 1, 1]
])

# Tính tích của ma trận (D + I)^-1 và A
result_1 = np.linalg.inv(D + np.identity(4)) @ A
print("Result of (D + I)^-1 @ A:")
print(result_1)
# Output:
# [[0.25      , 0.25      , 0.25      , 0.25      ],
#  [0.5       , 0.5       , 0.        , 0.        ],
#  [0.33333333, 0.        , 0.33333333, 0.33333333],
#  [0.33333333, 0.        , 0.33333333, 0.33333333]]

# Tính tích của ma trận A và (D + I)^-1
result_2 = A @ np.linalg.inv(D + np.identity(4))
print("Result of A @ (D + I)^-1:")
print(result_2)
# Output:
# [[0.25      , 0.5       , 0.33333333, 0.33333333],
#  [0.25      , 0.5       , 0.        , 0.        ],
#  [0.25      , 0.        , 0.33333333, 0.33333333],
#  [0.25      , 0.        , 0.33333333, 0.33333333]]

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv


from torch_geometric.transforms import RandomNodeSplit

import torch_geometric.transforms as T

from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load Cora Dataset
dataset = Planetoid(root=".", name="Cora")
data = dataset[0]

# Compute node degrees
degrees = degree(data.edge_index[0]).numpy()
numbers = Counter(degrees)

# Plot node degree distribution
fig, ax = plt.subplots()
ax.set_xlabel('Node degree')
ax.set_ylabel('Number of nodes')
plt.bar(numbers.keys(), numbers.values())
plt.show()

# GCN Model Definition
class GCN(torch.nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h = self.gcn2(h, edge_index)
        return F.log_softmax(h, dim=1)

    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        self.train()
        for epoch in range(epochs+1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = self.accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = self.accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc*100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc*100:.2f}%')

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        return self.accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])

    def accuracy(self, pred_y, y):
        return (pred_y == y).sum().item() / len(y)

# Train and evaluate GCN on Cora dataset
gcn = GCN(dataset.num_features, 16, dataset.num_classes)
print(gcn)
gcn.fit(data, epochs=100)
test_acc = gcn.test(data)
print(f'GCN test accuracy: {test_acc * 100:.2f}%')

# Load Wikipedia Network Dataset (chameleon)
dataset = WikipediaNetwork(root=".", name="chameleon", transform=T.RandomNodeSplit(num_val=200, num_test=500))
data = dataset[0]
print(f'Dataset: {dataset}')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {data.x.shape[0]}')
print(f'Number of unique features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

# Load target data for regression
df = pd.read_csv('wikipedia/chameleon/musae_chameleon_target.csv')
values = np.log10(df['mục tiêu'])
data.y = torch.tensor(values)

# Visualize target distribution
sns.distplot(values, fit=norm)
plt.show()

# GCN Model for Regression
class GCNRegressor(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h * 4)
        self.gcn2 = GCNConv(dim_h * 4, dim_h * 2)
        self.gcn3 = GCNConv(dim_h * 2, dim_h)
        self.linear = torch.nn.Linear(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn2(h, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn3(h, edge_index)
        h = torch.relu(h)
        h = self.linear(h)
        return h

    def fit(self, data, epochs):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.02, weight_decay=5e-4)
        self.train()
        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = F.mse_loss(out.squeeze()[data.train_mask], data.y[data.train_mask].float())
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                val_loss = F.mse_loss(out.squeeze()[data.val_mask], data.y[data.val_mask])
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.5f} | Val Loss: {val_loss:.5f}')

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        return F.mse_loss(out.squeeze()[data.test_mask], data.y[data.test_mask].float())

# Train and evaluate GCN for regression
gcn = GCNRegressor(dataset.num_features, 128, 1)
print(gcn)
gcn.fit(data, epochs=200)
test_loss = gcn.test(data)
print(f'GCN test loss: {test_loss:.5f}')

# Evaluate the model performance
out = gcn(data.x, data.edge_index).squeeze()[data.test_mask].detach().numpy()
y_true = data.y[data.test_mask].numpy()
mse = mean_squared_error(y_true, out)
mae = mean_absolute_error(y_true, out)

print('=' * 43)
print(f'MSE = {mse:.4f} | RMSE = {np.sqrt(mse):.4f} | MAE = {mae:.4f}')
print('=' * 43)

# Plot prediction vs actual values
sns.regplot(x=y_true, y=out)
plt.show()
