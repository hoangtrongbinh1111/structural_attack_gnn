import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import joblib
import argparse
import os
import sys
from tqdm.auto import tqdm
from sklearn import preprocessing

# Compute acc function
# Change compute_accuracy
def compute_accuracy(pred_labels, labels):
    return (pred_labels == labels).float().mean().item()


# Build PyG Data object from NetworkX
def build_pyg_data_from_nx(G_nx, num_node_features):
    node_mapping = {node: i for i, node in enumerate(G_nx.nodes)}
    edge_index = []
    edge_h = []
    edge_label = []
    
    for u, v, attr in G_nx.edges(data=True):
        u_idx = node_mapping[u]
        v_idx = node_mapping[v]
        edge_index.append([u_idx, v_idx])
        edge_h.append(attr['h'])
        edge_label.append(attr['Label'])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_h, dtype=torch.float)
    edge_label = torch.tensor(edge_label, dtype=torch.long)
    
    x = torch.ones(len(G_nx.nodes), num_node_features)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.edge_label = edge_label
    return data

# GAT Edge Classifier model
class GATEdgeClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super(GATEdgeClassifier, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.2)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.2)
        self.edge_classifier = nn.Sequential(
            nn.Linear(out_channels * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        src, dst = edge_index
        edge_emb = torch.cat([x[src], x[dst]], dim=1)
        edge_logits = self.edge_classifier(edge_emb)
        return edge_logits

# ----- Argument parsing -----
parser = argparse.ArgumentParser(description="Train GAT model")
parser.add_argument('--train', required=True, help="CSV file with train data", dest="train_path")
parser.add_argument('--test', required=True, help="CSV file with test data", dest="test_path")
parser.add_argument('--savedir', default="./results/", dest="save_dir", help="Directory for saving model and results")
args = parser.parse_args()

# ----- Paths -----
train_path = args.train_path
test_path = args.test_path
save_dir = args.save_dir
if not save_dir.endswith("/"):
    save_dir += "/"

if 'ToN_IoT' in train_path:
    save_dir += 'ToN_IoT/'
else:
    save_dir += 'CTU/'

save_results_dir = save_dir
mal_name = os.path.basename(train_path).split("_")[0]
save_dir += f"models/gat/{mal_name}/"
save_results_dir += f"scores/gat/{mal_name}/"

os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_results_dir, exist_ok=True)

save_dict_path = save_dir + "dict.pth"
save_model_path = save_dir + "model.pth"
save_results_path = save_results_dir + "test_scores_after_train.csv"

# ----- Read + normalize data -----
print("Reading train and test data...")
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

print("Normalizing numerical features...")
scaler = preprocessing.MinMaxScaler()
if 'ToN_IoT' in train_path:
    cols_to_norm = list(train_data.columns)
    cols_to_norm.remove('Src IP')
    cols_to_norm.remove('Dst IP')
    cols_to_norm.remove('Label')
else:
    cols_to_norm = ['Dur','TotPkts','TotBytes','SrcBytes','BytesPerPkt','PktsPerSec','RatioOutIn', 'DstBytes']

train_data[cols_to_norm] = scaler.fit_transform(train_data[cols_to_norm])
joblib.dump(scaler, save_dir + "scaler.skl")

feature_cols = list(train_data.columns)
feature_cols.remove("Label")
feature_cols.remove("Src IP")
feature_cols.remove("Dst IP")

if mal_name == 'neris':
    mal = train_data[train_data.Label == 1]
    ben = train_data[train_data.Label == 0]
    mal = mal.sample(30000)
    ben = ben.sample(len(mal) * 10)
    train_data = pd.concat([mal, ben]).sample(frac=1)

train_data['h'] = train_data[feature_cols].values.tolist()

# ----- Build graph -----
print("Building graph...")
G_nx = nx.from_pandas_edgelist(train_data, "Src IP", "Dst IP", ['h', 'Label'], create_using=nx.MultiGraph())
G_nx = G_nx.to_directed()

num_node_features = len(train_data['h'].iloc[0])
G_pyg = build_pyg_data_from_nx(G_nx, num_node_features)

print(f"Number of nodes in train: {G_pyg.num_nodes}")
print(f"Number of edges in train: {G_pyg.num_edges}")

# ----- Setup training -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = GATEdgeClassifier(in_channels=G_pyg.x.shape[1], hidden_channels=128, out_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

G_pyg = G_pyg.to(device)
edge_label = G_pyg.edge_label.to(device)

# ----- Training loop -----
print("\n\n-----------TRAINING-------------")
pbar = tqdm(range(1, 201))
for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    edge_logits = model(G_pyg.x, G_pyg.edge_index)
    loss = criterion(edge_logits, edge_label)
    loss.backward()
    optimizer.step()
    pbar.update(1)
    
    # if epoch % 10 == 0:
    #     pred = edge_logits.argmax(1)
    #     acc = compute_accuracy(pred, edge_label)
    #     pbar.write(f'Epoch: {epoch}')
    #     pbar.write(f'Training acc: {acc:.4f}')
        
    if epoch % 1 == 0:
        pred_labels = edge_logits.argmax(1)
        acc = compute_accuracy(pred_labels, edge_label)
        pbar.write(f'Epoch: {epoch}')
        pbar.write(f'Training acc: {acc:.4f}')

# ----- Save model -----
torch.save(model.state_dict(), save_dict_path)
torch.save(model, save_model_path)
