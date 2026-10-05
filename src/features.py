import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os

def one_hot(value, allowable_set, unknown=False):
    if unknown:
        result = [0.0] * (len(allowable_set) + 1)
        if value in allowable_set:
            index = allowable_set.index(value)
            result[index] = 1.0
        else:
            result[-1] = 1.0
        return result
    else:
        result = [0.0] * len(allowable_set)
        if value in allowable_set:
            index = allowable_set.index(value)
            result[index] = 1.0
        return result


def atom_type_features(mol):
    allowable = ["C","N","O","F","P","S","Cl","Br","I"]
    features = []

    for atom in mol.GetAtoms():
        value = atom.GetSymbol()
        features.append(one_hot(value,allowable,unknown=True))
    return torch.tensor(features,dtype=torch.float32)


def formal_charge_features(mol):
    allowable = [-2, -1, 0, 1, 2]
    features = []

    for atom in mol.GetAtoms():
        value = atom.GetFormalCharge()
        features.append(one_hot(value,allowable,unknown=False))
    return torch.tensor(features,dtype=torch.float32)


def hybridization_features(mol):
    allowable = ["SP","SP2","SP3"]
    features = []
    
    for atom in mol.GetAtoms():
        value = str(atom.GetHybridization())
        features.append( one_hot(value,allowable,unknown=False) )
    return torch.tensor(features,dtype=torch.float32)


_FEATURE_FACTORY = None

def get_feature_factory():
    global _FEATURE_FACTORY
    if _FEATURE_FACTORY is None:
        _FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory( os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))
    return _FEATURE_FACTORY


def hydrogen_bond_features(mol):
    factory = get_feature_factory()
    rdkit_features = factory.GetFeaturesForMol(mol)
    features = [[0.0, 0.0] for _ in mol.GetAtoms()]

    for feature in rdkit_features:
        atom_id = feature.GetAtomIds()[0]
        family = feature.GetFamily()
        if family == "Donor":
            features[atom_id][0] = 1.0
        elif family == "Acceptor":
            features[atom_id][1] = 1.0
    return torch.tensor(features, dtype=torch.float32)


def aromatic_features(mol):
    features = []
    
    for atom in mol.GetAtoms():
        features.append([float(atom.GetIsAromatic())])
    return torch.tensor(features,dtype=torch.float32)


def degree_features(mol):
    allowable = [1,2,3,4,5]
    features = []

    for atom in mol.GetAtoms():
        value = atom.GetTotalDegree()
        features.append(one_hot( value,allowable,unknown=True))
    return torch.tensor(features,dtype=torch.float32)

def hydrogen_count_features(mol):
    allowable = [0,1,2,3]
    features = []

    for atom in mol.GetAtoms():
        value = atom.GetTotalNumHs()
        features.append(one_hot(value,allowable,unknown=False))
    return torch.tensor(features, dtype=torch.float32)


def chirality_features(mol):
    features = []
    
    for atom in mol.GetAtoms():
        feature = [0.0, 0.0]
        if atom.HasProp("_CIPCode"):
            code = atom.GetProp("_CIPCode")
            if code == "R":
                feature[0] = 1.0
            elif code == "S":
                feature[1] = 1.0
        features.append(feature)
        
    return torch.tensor(features,dtype=torch.float32)


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
        
    if mol.GetNumAtoms() <= 1:
        return None

    atom_type = atom_type_features(mol)
    formal_charge = formal_charge_features(mol)
    hybridization = hybridization_features(mol)
    hydrogen_bond = hydrogen_bond_features(mol)
    aromatic = aromatic_features(mol)
    degree = degree_features(mol)
    hydrogen_count = hydrogen_count_features(mol)
    chirality = chirality_features(mol)

    node_features = torch.cat([ atom_type, formal_charge,hybridization, hydrogen_bond,aromatic, degree,hydrogen_count, chirality],dim=1)

    src = []
    dst = []

    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()

        src.append(start)
        dst.append(end)

        src.append(end)
        dst.append(start)

    edge_index = torch.tensor(
        [src, dst],
        dtype=torch.long
    )

    return node_features, edge_index
