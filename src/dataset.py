import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from features import smiles_to_graph


def load_data(path):
    df = pd.read_excel(path)
    graphs = []
    for _, row in df.iterrows():
        graph = smiles_to_graph(row["SMILES"])
        if graph is None:
            continue
        
        node_features, edge_index = graph
        data = Data( x=node_features,edge_index=edge_index)
        data.y = torch.tensor([float(row["BBB_labels"])],dtype=torch.float32)
        graphs.append(data)
    return graphs


def split_data(graphs,validation_size=0.2,random_state=42):
    labels = [int(graph.y.item()) for graph in graphs]
    train_graphs, val_graphs = train_test_split(graphs, test_size=validation_size, random_state=random_state, stratify=labels)
    return train_graphs, val_graphs

def load_prediction_data(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    
    graphs = []
    valid_rows = []

    for index, row in df.iterrows():
        graph = smiles_to_graph(row["SMILES"])
        if graph is None:
            continue

        node_features, edge_index = graph
        data = Data(x=node_features,edge_index=edge_index)
        graphs.append(data)
        valid_rows.append(index)
        
    skipped = len(df) - len(valid_rows)

    if skipped > 0:
        print("Skipped invalid / single-atom SMILES:",skipped)

    df = df.loc[valid_rows].reset_index(drop=True)
    return df, graphs
