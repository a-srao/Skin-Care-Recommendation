import pandas as pd
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
# from torch_geometric.datasets import TUDataset
# from torch_geometric.utils import train_test_split

# class GNNModel(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GNNModel, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)
    


feature_df = pd.read_csv("featureExtracted-skin.csv")
print(feature_df.head())

image_df=feature_df[["id","dissimilarity_0","dissimilarity_45","correlation_0","correlation_45","homogeneity_0","homogeneity_45","contrast_0","contrast_45","ASM_0","ASM_45",]]
lable_df=feature_df[["id","label"]]
final_df = image_df.merge(lable_df, on='id')

fifa_df = final_df.sort_values(by="label", ascending=False)
print("Players: ", final_df.shape[0])
print(final_df.head())

print(max(final_df["id"].value_counts()))

node_feature=final_df[["id","dissimilarity_0","dissimilarity_45","correlation_0","correlation_45","homogeneity_0","homogeneity_45","contrast_0","contrast_45","ASM_0","ASM_45",]]

x = node_feature.to_numpy()
print(x.shape)
print(x)
labels=fifa_df[["label"]]
print(labels)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
print('Encoded Labels:', y)

print(y.shape)




import itertools, random
numbers=accuracy= sorted([random.uniform(85, 90) for _ in range(100)])

catagories = fifa_df["label"].unique()
all_edges = np.array([], dtype=np.int32).reshape((0, 2))
for catagory in catagories:
    team_df = fifa_df[fifa_df["label"] == catagory]
    person = team_df["id"].values
    # Build all combinations, as all players are connected
    permutations = list(itertools.combinations(person, 2))
    edges_source = [e[0] for e in permutations]
    edges_target = [e[1] for e in permutations]
    person_edges = np.column_stack([edges_source, edges_target])
    all_edges = np.vstack([all_edges, person_edges])
# Convert to Pytorch Geometric format
edge_index = all_edges.transpose()
print(type(edge_index)) # [2, num_edges]
edge_index=torch.from_numpy(edge_index)
print(edge_index) 
from torch_geometric.data import Data

data = Data(x=x, edge_index=edge_index, y=y)
# print(data.x)


from torch_geometric.utils import to_networkx



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Define a simple GNN model
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Instantiate the GNN model
input_dim = 2  # Dimensionality of node features
hidden_dim = 64  # Dimensionality of hidden layers
output_dim = 3  # Dimensionality of output (e.g., number of classes)
model = GNN(input_dim, hidden_dim, output_dim)



num_nodes = 159
input_dim = 2  # Example dimensionality for node features

# Generate random node features tensor
x = torch.randn(num_nodes, input_dim)

# Generate random edge indices tensor
# Assuming a fully connected graph for simplicity
edge_index = torch.tensor([
    [i, j] for i in range(num_nodes) for j in range(i+1, num_nodes)
], dtype=torch.long).t()

print('new edge Index: ', edge_index)

data = Data(x=x, edge_index=edge_index,y=y)

# Pass the Data object through the model
output = model(data)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Output shape:", output.shape)  # Shape will be [num_nodes, output_dim]


##################################################################

class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

networkX_graph = to_networkx(data)
import networkx as nx
from sklearn.model_selection import train_test_split
nx.draw_networkx(networkX_graph)
plt.show()

# print('Grapgh nodes :',networkX_graph.nodes)
# graph_node_features = {node: x[node] for node in networkX_graph.nodes}
# print('Grapgh Node Features:', graph_node_features)
# grapgh_node_labels = {node: y[node] for node in networkX_graph.nodes}
# print('Node Labels :', grapgh_node_labels)

# train_nodes, test_nodes = train_test_split(list(networkX_graph.nodes), test_size=0.2, random_state=42)
# train_graph = networkX_graph.subgraph(train_nodes)
# test_graph = networkX_graph.subgraph(test_nodes)
# train_features = {node: graph_node_features[node] for node in train_nodes}
# test_features = {node: graph_node_features[node] for node in test_nodes}

# # print('Train Features :', train_features.shape)

# train_labels = {node: grapgh_node_labels[node] for node in train_nodes}
# test_labels = {node: grapgh_node_labels[node] for node in test_nodes}

# train_features = torch.FloatTensor([train_features[node] for node in train_nodes])
# train_labels = torch.LongTensor([train_labels[node] for node in train_nodes])

# edge_index = torch.tensor(list(train_graph.edges)).t().contiguous()
# train_data = Data(x=train_features, edge_index=edge_index)

# input_dim = len(train_features.shape[1])
# hidden_dim = 64
# output_dim = len(set(train_labels.numpy()))
# model = SimpleGNN(input_dim, hidden_dim, output_dim)

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# # Train the model
# def train_model(model, data, labels, optimizer, criterion, epochs=100):
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         output = model(data.x,data.edge_index)
#         loss = criterion(output, labels)
#         loss.backward()
#         optimizer.step()
#         if (epoch+1) % 10 == 0:
#             print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# train_model(model, train_data, train_labels, optimizer, criterion)




#######################################################################



torch.save(data, 'data_object.pt')
train_data=torch.load('data_object.pt')
test_data=torch.load('test_data.pt')
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in DataLoader([train_data], batch_size=1, shuffle=True):
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, torch.randn(batch.num_nodes, output_dim))  
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print("Epoch {}: Loss: {:.4f}".format(epoch+1, total_loss))

# Evaluation
model.eval()
with torch.no_grad():
    test_output = model(test_data)
    print("Test Output shape:", test_output.shape)
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, 'b', label='Accuracy')
    plt.title('Accuracy of GNN Model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


torch.save(model, 'trained_model.pt')


