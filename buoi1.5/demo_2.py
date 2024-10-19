from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree
from collections import Counter
import matplotlib.pyplot as plt

dataset = Planetoid(root=".", name="Cora")
data = dataset[0]

degrees = degree(data.edge_index[0]).numpy()

numbers = Counter(degrees)

fig, ax = plt.subplots()
ax.set_xlabel('Node degree')
ax.set_ylabel('Number of nodes')
plt.bar(numbers.keys(), numbers.values())
