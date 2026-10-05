import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,matthews_corrcoef,confusion_matrix)
from features import smiles_to_graph
from model import GraphConvModel

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load test data
df = pd.read_excel("../data/test_mol_df.xlsx")
print("Test molecules:", len(df))

# Convert test molecules
test_graphs = []

for _, row in df.iterrows():
    graph = smiles_to_graph(row["SMILES"])
    if graph is None:
        continue
    node_features, edge_index = graph
    data = Data( x=node_features,edge_index=edge_index)
    data.y = torch.tensor([float(row["BBB_labels"])],dtype=torch.float32)
    test_graphs.append(data)

print("Valid test graphs:", len(test_graphs))

# DataLoader
test_loader = DataLoader(test_graphs,batch_size=32,shuffle=False)

# Load model
model = GraphConvModel(input_features=33,embedding_size=128)
model.load_state_dict( torch.load("../best_model.pt", map_location=device ))
model = model.to(device)
model.eval()

# Predictions
all_labels = []
all_probabilities = []
all_predictions = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        output = model(batch.x,batch.edge_index,batch.batch)
        probabilities = torch.sigmoid(output)
        predictions = ( probabilities >= 0.5).float()
        all_labels.extend( batch.y.view(-1).cpu().numpy() )
        all_probabilities.extend(probabilities.view(-1).cpu().numpy())
        all_predictions.extend(predictions.view(-1).cpu().numpy())

# Metrics
accuracy = accuracy_score(all_labels,all_predictions)
precision = precision_score(all_labels,all_predictions,zero_division=0)
recall = recall_score(all_labels,all_predictions,zero_division=0)
f1 = f1_score(all_labels,all_predictions,zero_division=0)
roc_auc = roc_auc_score(all_labels, all_probabilities)
mcc = matthews_corrcoef(all_labels,all_predictions)

# Confusion Matrix
cm = confusion_matrix( all_labels, all_predictions)

# Print results
print()
print("========== TEST RESULTS ==========")
print(f"Accuracy : {accuracy:.4f}")
print( f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"MCC      : {mcc:.4f}")

print()
print("Confusion Matrix:")
print(cm)
