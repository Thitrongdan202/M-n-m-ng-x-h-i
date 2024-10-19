import matplotlib.pyplot as plt
from collections import Counter
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree

# Tải bộ dữ liệu Planetoid
dataset = Planetoid(root=".", name="Cora")
data = dataset[0]

# Tính toán bậc của các đỉnh
degrees = degree(data.edge_index[0]).numpy()

# Đếm số lượng đỉnh cho mỗi bậc
degree_count = Counter(degrees)

# Vẽ biểu đồ
fig, ax = plt.subplots()
ax.set_xlabel('Node Degree')
ax.set_ylabel('Number of Nodes')
ax.bar(degree_count.keys(), degree_count.values())
plt.show()

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# Hàm tính độ chính xác
def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()


# Lớp GCN
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

        for epoch in range(epochs + 1):
            optimizer.zero_grad()
            out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])

            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
                print(
                    f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc * 100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc * 100:.2f}%')

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
        return acc


# Khởi tạo và huấn luyện mô hình
gcn = GCN(dataset.num_features, 16, dataset.num_classes)
print(gcn)
gcn.fit(data, epochs=100)

# Kiểm tra độ chính xác trên bộ dữ liệu test
acc = gcn.test(data)
print(f'GCN test accuracy: {acc * 100:.2f}%')

from torch_geometric.datasets import WikipediaNetwork
import torch_geometric.transforms as T
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Tải và chuẩn bị bộ dữ liệu WikipediaNetwork
dataset = WikipediaNetwork(root=".", name="chameleon",
                           transform=T.RandomNodeSplit(num_val=200, num_test=500))
data = dataset[0]

# In thông tin về dataset
print(f'Dataset: {dataset}')
print('-------------------')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {data.x.shape[0]}')
print(f'Number of unique features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

# Đọc file CSV
df = pd.read_csv('wikipedia/chameleon/musae_chameleon_target.csv')

# Lấy giá trị log10 của cột 'target'
values = np.log10(df['target'])

# Gán giá trị cho data.y
data.y = torch.tensor(values.values, dtype=torch.float64)

# In tensor để kiểm tra
print(data.y)

# Vẽ biểu đồ phân phối
df['target'] = values
sns.histplot(df['target'], kde=True, stat="density", line_kws={"linestyle": "--"}, color='b')
sns.lineplot(x=np.linspace(values.min(), values.max(), 100),
             y=norm.pdf(np.linspace(values.min(), values.max(), 100), values.mean(), values.std()),
             color='r')
plt.xlabel('Log10(Target)')
plt.ylabel('Density')
plt.title('Distribution of Log10(Target)')
plt.show()

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Định nghĩa lớp GCN
class GCN(torch.nn.Module):
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
                val_loss = F.mse_loss(out.squeeze()[data.val_mask], data.y[data.val_mask].float())
                print(f"Epoch {epoch:>3} | Train Loss: {loss:.5f} | Val Loss: {val_loss:.5f}")

    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        return F.mse_loss(out.squeeze()[data.test_mask], data.y[data.test_mask].float())

# Khởi tạo mô hình GCN và huấn luyện
gcn = GCN(dataset.num_features, 128, 1)
print(gcn)
gcn.fit(data, epochs=200)

# Kiểm tra mô hình với bộ dữ liệu test
loss = gcn.test(data)
print(f'GCN test loss: {loss:.5f}')

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

# Dự đoán giá trị cho dữ liệu test
out = gcn(data.x, data.edge_index)
y_pred = out.squeeze()[data.test_mask].detach().numpy()
y_true = data.y[data.test_mask].numpy()

# Tính các metric
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mse)

# In kết quả các metric
print('=' * 43)
print(f'MSE = {mse:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}')
print('=' * 43)

# Vẽ biểu đồ hồi quy
plt.figure(figsize=(8, 6))
sns.regplot(x=y_true, y=y_pred, line_kws={"color": "red"})
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Regression Plot of Actual vs Predicted Values')
plt.grid(True)
plt.show()


