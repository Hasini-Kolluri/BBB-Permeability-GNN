import numpy as np

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def gradient_colors(start_color, end_color, n_colors):

    start_color = np.array(start_color)
    end_color = np.array(end_color)

    colors = []

    for i in range(n_colors):

        color = start_color + (
            end_color - start_color
        ) * (i / (n_colors - 1))

        colors.append(tuple(color.tolist()))

    return colors


def atom_importance(edge_index, edge_mask):

    # edge_index: numpy array, shape [2, num_edges]
    # edge_mask : numpy array, shape [num_edges]
    #
    # features.py stores every bond as two consecutive
    # directed edges (a -> b, b -> a), so edges 2k and 2k+1
    # belong to the same bond. graphB3 averages the pair.

    edge_index = np.asarray(edge_index)
    edge_mask = np.asarray(edge_mask, dtype=float)

    bond_scores = edge_mask.reshape(-1, 2).mean(axis=1)

    atom_bins = {}

    for k, score in enumerate(bond_scores):

        # 10 importance windows: 0.0-0.1 ... 0.9-1.0
        bin_index = min(int(score * 10), 9)

        for atom in (
            int(edge_index[0, 2 * k]),
            int(edge_index[1, 2 * k])
        ):

            # an atom takes its most important bond's window
            atom_bins[atom] = max(
                atom_bins.get(atom, 0),
                bin_index
            )

    return atom_bins


def highlight_molecule(
    smiles,
    atom_bins,
    label,
    output_path,
    image_size=(500, 500)
):

    mol = Chem.MolFromSmiles(smiles)

    # BBB+ -> red shades, BBB- -> blue shades (same as graphB3)
    if label == 1:

        low = [1.0, 0.8, 0.8]
        medium = [1.0, 0.6, 0.6]
        high = [1.0, 0.1, 0.1]

    else:

        low = [0.6, 0.8, 1.0]
        medium = [0.4, 0.6, 1.0]
        high = [0.2, 0.4, 0.9]

    colors = (
        gradient_colors(low, medium, 5)
        + gradient_colors(medium, high, 5)
    )

    highlight_colors = {}

    for atom, bin_index in atom_bins.items():

        highlight_colors[atom] = colors[bin_index]

    drawer = rdMolDraw2D.MolDraw2DCairo(
        image_size[0],
        image_size[1]
    )

    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(highlight_colors.keys()),
        highlightAtomColors=highlight_colors
    )

    drawer.FinishDrawing()

    drawer.WriteDrawingText(output_path)
