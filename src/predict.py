import os

import pandas as pd
import torch

from torch_geometric.loader import DataLoader

from dataset import load_prediction_data
from model import GraphConvModel


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# Paths
# -----------------------------

INPUT_PATH = "../data/test_mol_df.xlsx"

MODEL_PATH = "../best_model.pt"

OUTPUT_PATH = "../results/predictions.csv"

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)


# -----------------------------
# Load molecules
# -----------------------------

df, graphs = load_prediction_data(INPUT_PATH)

print("Valid molecules:", len(graphs))


loader = DataLoader(
    graphs,
    batch_size=32,
    shuffle=False
)


# -----------------------------
# Load model
# -----------------------------

model = GraphConvModel(
    input_features=33,
    embedding_size=128
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()


# -----------------------------
# Predictions
# -----------------------------

all_probabilities = []
all_predictions = []

with torch.no_grad():

    for batch in loader:

        batch = batch.to(device)

        output = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        probabilities = torch.sigmoid(
            output
        )

        # logit >= 0 is the same as probability >= 0.5
        predictions = (
            output >= 0.0
        ).int()

        all_probabilities.extend(
            probabilities.view(-1).cpu().tolist()
        )

        all_predictions.extend(
            predictions.view(-1).cpu().tolist()
        )


# -----------------------------
# Save results
# -----------------------------

results = pd.DataFrame(
    {
        "SMILES": df["SMILES"],
        "y_pred": all_predictions,
        "y_prob": all_probabilities
    }
)

results["status"] = results["y_pred"].apply(
    lambda value: "BBB+" if value == 1 else "BBB-"
)

if "BBB_labels" in df.columns:

    results["BBB_labels"] = df["BBB_labels"]

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print(results.head(10))

print()
print("Predictions saved to:", OUTPUT_PATH)
