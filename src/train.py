import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import matthews_corrcoef
from dataset import load_data, split_data
from model import GraphConvModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load dataset
graphs = load_data("../data/train_mol_set.xlsx")
print("Total graphs:", len(graphs))

# Train / validation split
train_graphs, val_graphs = split_data(graphs,validation_size=0.2,random_state=42)
print("Training graphs:", len(train_graphs))
print("Validation graphs:", len(val_graphs))

# DataLoaders
train_loader = DataLoader(train_graphs,batch_size=32,shuffle=True)
val_loader = DataLoader(val_graphs,batch_size=32,shuffle=False)

# Model
model = GraphConvModel(input_features=33,embedding_size=128)
model = model.to(device)

# Loss and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# Training
epochs = 100

# graphB3 keeps the checkpoint with the best MCC (best_mcc_model_weights.pth), so we select on validation MCC instead of validation loss.
best_val_mcc = float("-inf")
best_val_loss = float("inf")

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch.x,batch.edge_index,batch.batch)
        loss = criterion(output.view(-1),batch.y.view(-1) )
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_labels = []
    val_predictions = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model( batch.x,batch.edge_index, batch.batch)
            loss = criterion( output.view(-1), batch.y.view(-1))
            val_loss += loss.item()
            
            # logit >= 0 is the same as probability >= 0.5
            predictions = (output.view(-1) >= 0.0).int()

            val_labels.extend( batch.y.view(-1).int().cpu().numpy() )
            val_predictions.extend( predictions.cpu().numpy() )
    val_loss /= len(val_loader)
    val_mcc = matthews_corrcoef(val_labels,val_predictions)

    # Save best model
    if val_mcc > best_val_mcc:
        best_val_mcc = val_mcc
        best_val_loss = val_loss
        torch.save(model.state_dict(),"../best_model.pt")
        best = "*"
    else:
        best = ""

    print( f"Epoch {epoch + 1:03d} | "f"Train Loss: {train_loss:.4f} | " f"Val Loss: {val_loss:.4f} | " f"Val MCC: {val_mcc:.4f} {best}")

print()
print("Training complete.")
print("Best validation MCC:", best_val_mcc)
print("Validation loss at best MCC:", best_val_loss)
print("Best model saved as: ../best_model.pt")
