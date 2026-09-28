import os
import random

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from torch_geometric.explain import (
    Explainer,
    GNNExplainer,
    unfaithfulness
)

from dataset import load_prediction_data
from model import GraphConvModel
from visualize import atom_importance, highlight_molecule


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# Settings
# -----------------------------

INPUT_PATH = "../data/test_mol_df.xlsx"

MODEL_PATH = "../best_model.pt"

OUTPUT_DIR = "../results"

FIGURE_DIR = os.path.join(
    OUTPUT_DIR,
    "explain_figures"
)

# GNNExplainer optimises a mask for every molecule
# (4000 epochs in graphB3), so start with a small subset.
MAX_MOLECULES = 20

EXPLAINER_EPOCHS = 4000

EXPLAINER_LR = 0.0005

SEED = 454

os.makedirs(FIGURE_DIR, exist_ok=True)


def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Load molecules
# -----------------------------

df, graphs = load_prediction_data(INPUT_PATH)

df = df.iloc[:MAX_MOLECULES].reset_index(drop=True)

graphs = graphs[:MAX_MOLECULES]

print("Molecules to explain:", len(graphs))


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
# Explainer
# -----------------------------

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(
        epochs=EXPLAINER_EPOCHS,
        lr=EXPLAINER_LR
    ),
    explanation_type="model",
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config=dict(
        mode="binary_classification",
        task_level="graph",
        return_type="raw"
    )
)


# -----------------------------
# Predict + explain
# -----------------------------

y_pred = []
y_prob = []
unfaithfulness_scores = []

for index in tqdm(
    range(len(graphs)),
    desc="Running GNNExplainer"
):

    data = graphs[index].to(device)

    batch_index = torch.zeros(
        data.num_nodes,
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        output = model(
            data.x,
            data.edge_index,
            batch_index
        )

    label = int(output.item() >= 0.0)

    y_pred.append(label)

    y_prob.append(
        torch.sigmoid(output).item()
    )

    # molecules without bonds (e.g. salts) have no edges to explain
    if data.edge_index.size(1) == 0:

        unfaithfulness_scores.append(float("nan"))

        continue

    set_seed(SEED)

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        batch_index=batch_index
    )

    unfaithfulness_scores.append(
        float(
            unfaithfulness(
                explainer,
                explanation
            )
        )
    )

    atom_bins = atom_importance(
        data.edge_index.cpu().numpy(),
        explanation.edge_mask.detach().cpu().numpy()
    )

    highlight_molecule(
        df.loc[index, "SMILES"],
        atom_bins,
        label,
        os.path.join(
            FIGURE_DIR,
            f"molecule_{index}.png"
        )
    )


# -----------------------------
# Save results
# -----------------------------

results = pd.DataFrame(
    {
        "SMILES": df["SMILES"],
        "y_pred": y_pred,
        "y_prob": y_prob
    }
)

results["status"] = results["y_pred"].apply(
    lambda value: "BBB+" if value == 1 else "BBB-"
)

results["Unfaith"] = unfaithfulness_scores

if "BBB_labels" in df.columns:

    results["BBB_labels"] = df["BBB_labels"]

results_path = os.path.join(
    OUTPUT_DIR,
    "explain_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print(results)

print()
print("Results saved to:", results_path)
print("Figures saved to:", FIGURE_DIR)
