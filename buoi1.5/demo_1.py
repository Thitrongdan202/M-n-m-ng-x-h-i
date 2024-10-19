import matplotlib.pyplot as plt
import networkx as nx

# Create a simple graph with 5 nodes and edges
G = nx.Graph()

# Add nodes
G.add_nodes_from([1, 2, 3, 4, 5])

# Add edges (connections between nodes)
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 5), (4, 5)])

# Draw the graph
nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=15)

# Display the graph
plt.show()

# Set node positions
pos = nx.spring_layout(G)

# Draw the graph with different properties
nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=700, font_size=12)
nx.draw_networkx_edges(G, pos, width=2)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(1, 2): '5', (2, 4): '2'}, font_color='red')

# Display the graph
plt.show()

# Add weighted edges
G.add_weighted_edges_from([(1, 2, 0.6), (3, 4, 0.2)])

# Draw graph with edge labels (weights)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=12)
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))

# Display
plt.show()

import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create a line plot
plt.plot(x, y, marker='o')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')

# Show plot
plt.show()

categories = ['A', 'B', 'C', 'D']
values = [3, 7, 1, 5]

# Create a bar chart
plt.bar(categories, values, color='blue')

# Add labels and title
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Simple Bar Chart')

# Show plot
plt.show()

labels = ['A', 'B', 'C', 'D']
sizes = [15, 30, 45, 10]

# Create a pie chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

# Equal aspect ratio ensures the pie chart is circular
plt.axis('equal')

# Show plot
plt.title('Simple Pie Chart')
plt.show()

import numpy as np

# Generate random data
data = np.random.randn(1000)

# Create a histogram
plt.hist(data, bins=30, color='green')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Simple Histogram')

# Show plot
plt.show()

x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11]
y = [99, 86, 87, 88, 100, 86, 103, 87, 94, 78]

# Create a scatter plot
plt.scatter(x, y, color='red')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Scatter Plot')

# Show plot
plt.show()

# Generate random data
data = [np.random.normal(0, std, 100) for std in range(1, 4)]

# Create a box plot
plt.boxplot(data, vert=True, patch_artist=True)

# Add labels and title
plt.xlabel('Data Set')
plt.ylabel('Value')
plt.title('Simple Box Plot')

# Show plot
plt.show()

import seaborn as sns
import numpy as np

# Generate random matrix data
data = np.random.rand(10, 12)

# Create a heatmap
sns.heatmap(data, annot=True, cmap='coolwarm')

# Add title
plt.title('Simple Heatmap')

# Show plot
plt.show()

from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Sample data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = [1, 2, 3, 4, 5]
y = [10, 20, 30, 40, 50]
z = [5, 6, 2, 3, 13]

# Create a 3D scatter plot
ax.scatter(x, y, z)

# Add labels and title
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.title('Simple 3D Plot')

# Show plot
plt.show()

x = [1, 2, 3, 4, 5]
y1 = [1, 4, 6, 8, 9]
y2 = [2, 2, 7, 10, 12]

# Create an area plot
plt.fill_between(x, y1, color='skyblue', alpha=0.5)
plt.fill_between(x, y2, color='orange', alpha=0.5)

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Area Plot')

# Show plot
plt.show()

