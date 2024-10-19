from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
from collections import Counter
import matplotlib.pyplot as plt



import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

def accuracy(pred_y, y):
 return ((pred_y == y).sum() / len(y)).item()

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
     optimizer = torch.optim.Adam(self.parameters(),
                                  lr=0.01,
                                  weight_decay=5e-4)
     self.train()
     for epoch in range(epochs + 1):
         optimizer.zero_grad()
     out = self(data.x, data.edge_index)
     loss = criterion(out[data.train_mask],
                      data.y[data.train_mask])
     acc = accuracy(out[data.train_mask].
                    argmax(dim=1), data.y[data.train_mask])
     loss.backward()
     optimizer.step()
     if (epoch % 20 == 0):
         val_loss = criterion(out[data.val_mask],
                              data.y[data.val_mask])
         val_acc = accuracy(out[data.val_mask].
                            argmax(dim=1), data.y[data.val_mask])
         print(
             f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc * 100:>5.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc * 100:.2f}%')

         @ torch.no_grad()

         def test(self, data):
             self.eval()
             out = self(data.x, data.edge_index)
             acc = accuracy(out.argmax(dim=1)[data.test_mask],
                            data.y[data.test_mask])
             return acc

         gcn = GCN(dataset.num_features, 16, dataset.num_classes)
         print(gcn)
         gcn.fit(data, epochs=100)

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.gcn1 = GCNConv(dataset.num_node_features, 16)  # Input size = num of features in dataset
        self.gcn2 = GCNConv(16, dataset.num_classes)  # Output size = num of classes in dataset

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def test(self, data):
        self.eval()  # Set model to evaluation mode
        out = self.forward(data)
        pred = out.argmax(dim=1)
        correct = (pred == data.y).sum()
        accuracy = correct / data.num_nodes
        return accuracy.item()

# Instantiate the GCN model
gcn = GCN()

# Define optimizer
optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01)

# Training loop
for epoch in range(101):
    gcn.train()  # Set model to training mode
    optimizer.zero_grad()
    out = gcn(data)  # Forward pass
    loss = F.nll_loss(out, data.y)  # Loss calculation
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    # Calculate training accuracy
    pred = out.argmax(dim=1)
    correct = (pred == data.y).sum().item()
    acc = correct / data.num_nodes

    # Validation (using same dataset for simplicity)
    gcn.eval()  # Set model to evaluation mode
    val_out = gcn(data)
    val_loss = F.nll_loss(val_out, data.y)
    val_pred = val_out.argmax(dim=1)
    val_correct = (val_pred == data.y).sum().item()
    val_acc = val_correct / data.num_nodes

    if epoch % 20 == 0 or epoch == 100:
        print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc * 100:>5.2f}% '
              f'| Val Loss: {val_loss:.2f} | Val Acc: {val_acc * 100:.2f}%')

# Test accuracy
acc = gcn.test(data)
print(f'GCN test accuracy: {acc * 100:.2f}%')

