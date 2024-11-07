
import os
from os.path import join
import sys

from typing import List, Union, Any
from datetime import datetime
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem
from IPython.display import display

from joblib import Parallel, delayed
from lib import utilities

RDLogger.DisableLog('rdApp.*')



### Modified version of the code from
### https://keras.io/examples/graph/mpnn-molecular-graphs/
class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            # print(k, s)
            s = sorted(list(s)) + ["unk"]
            # print("s=", s)
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim + 1))) ## The + 1 marks a bit that will be populated with -1 is the value is not allowed
            # print("==>", self.features_mapping[k])
            self.dim += len(s)
            # print("==>", self.dim)
        # print("==>", self.features_mapping)
    def encode(self, inputs):
        output = np.zeros((self.dim,))
        # print(output)
        for name_feature, feature_mapping in self.features_mapping.items():
            # print(name_feature, feature_mapping)
            feature = getattr(self, name_feature)(inputs)   ## e.g.: atomic_num(inputs)
            # print("feature: ", feature)
            if feature not in feature_mapping:
                output[feature_mapping['unk']] = 1.0
            else:
                output[feature_mapping[feature]] = 1.0
        return output

class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def atomic_num(self, atom:Chem.rdchem.Atom):
        return atom.GetAtomicNum()

    def n_valence(self, atom:Chem.rdchem.Atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom:Chem.rdchem.Atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom:Chem.rdchem.Atom):
        return atom.GetHybridization().name.lower()
    
    def chiral_tag(self, atom:Chem.rdchem.Atom):
        return int(atom.GetChiralTag())
    
    def is_aromatic(self, atom:Chem.rdchem.Atom):
        return atom.GetIsAromatic()
    
    def is_in_ring(self, atom):
        return atom.GetIsAromatic()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond:Chem.rdchem.Bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond:Chem.rdchem.Bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond:Chem.rdchem.Bond):
        return bond.GetIsConjugated()
    
    def stereo(self, bond:Chem.rdchem.Bond):
        return bond.GetStereo().name.lower()




ATOMIC_NUM_MAX = 100 ## We only consider atoms with a number of 100 or less (e.g.: 'H':1, 'C':6, 'O':8)
ATOM_FEATURIZER = AtomFeaturizer(
    allowable_sets={
        "atomic_num": set(range(1, ATOMIC_NUM_MAX+1)),
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3", "sp3", "sp3d", "sp3d2"},
        "chiral_tag": {0,1,2,3},
        "is_aromatic": {True, False},
        "is_in_ring": {True, False}        
    }
)

BOND_FEATURIZER = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
        "stereo": {"stereonone, stereoz, stereoe, stereocis, stereotrans"}
    }
)



def graph_from_molecule(molecule:AllChem.Mol, atom_featurizer:AtomFeaturizer=ATOM_FEATURIZER, bond_featurizer:BondFeaturizer=BOND_FEATURIZER):
    
    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []

    try:
    # if True:
        for atom in molecule.GetAtoms():
            atom_features.append(atom_featurizer.encode(atom))

            # Add self-loops
            pair_indices.append([atom.GetIdx(), atom.GetIdx()])
            bond_features.append(bond_featurizer.encode(None))

            for neighbor in atom.GetNeighbors():
                bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
                bond_features.append(bond_featurizer.encode(bond))

        atom_features = torch.tensor(np.array(atom_features), dtype=torch.long).view(-1, len(atom_features[0]))
        bond_features  = torch.tensor(np.array(bond_features), dtype=torch.long).view(-1, len(bond_features[0]))
        pair_indices = torch.tensor(pair_indices).t().to(torch.long).view(2, -1)

        # return np.array(atom_features), np.array(bond_features), np.array(pair_indices)
        return Data(x=atom_features, edge_index=pair_indices, edge_attr=bond_features)
    except Exception as exp:
        return None


def graph_from_smiles(smiles:str, atom_featurizer:AtomFeaturizer=ATOM_FEATURIZER, bond_featurizer:BondFeaturizer=BOND_FEATURIZER
                      , add_explicit_h:bool=True):
    try:
    # if True:
        mol = utilities.molecule_from_smiles(smiles, add_explicit_h)
        if not mol is None:
            graph_data = graph_from_molecule(mol, atom_featurizer, bond_featurizer)
            graph_data.smiles = smiles
            # if not y is None:
            #     graph_data.y = y
            return graph_data
        else:
            print(f"Could not generate a valid molecule from SMILES '{smiles}'.")
            return None
    except Exception as exp:
        print(f"Could not generate graph from SMILES '{smiles}'.")
        return None

def graph_from_smiles_list(smiles_list:List[str], atom_featurizer:AtomFeaturizer=ATOM_FEATURIZER, bond_featurizer:BondFeaturizer=BOND_FEATURIZER
                           , add_explicit_h:bool=True):
    graphs = Parallel(n_jobs=8)(
        delayed(graph_from_smiles)(smiles, atom_featurizer, bond_featurizer, add_explicit_h)
        for smiles in tqdm(smiles_list, desc="Converting SMILES to chemical graphs."))
    return graphs
    
def get_dataset(input_df: pd.DataFrame, smiles_column:str, target_column:str, atom_featurizer:AtomFeaturizer
                , bond_featurizer:BondFeaturizer, add_explicit_h:bool=True):
    # print("input_df.shape=", input_df.shape)
    # print(input_df
    graph_dataset = graph_from_smiles_list(smiles_list=input_df[smiles_column], atom_featurizer=atom_featurizer, bond_featurizer=bond_featurizer
                                        , add_explicit_h=add_explicit_h)
    # print("graph_dataset = ", graph_dataset)
    # print(len(graph_dataset))
    targets = input_df[target_column].values
    # print("targets = ", targets)

    for i in range(len(targets)):
        graph_dataset[i].y =  torch.tensor([targets[i]]) # torch.as_tensor([targets[i]]) / torch.FloatTensor([targets[i]]
    
    return graph_dataset


def predict_from_loader(loader, model, device='cpu'):
    pred_target = np.empty((0))
    outputs = np.empty((0,model.out_channels))
    for data in loader:
        data = data.to(device)
        data.x = data.x.float()
        output = model(data)
        _, predicted = torch.max(output, 1) ## returns maximum values(_), and their indices (predicted) along the dimmension 1
        pred_target = np.concatenate((pred_target, predicted.cpu()))
        outputs = np.concatenate((outputs, output.detach().cpu().numpy()))
    # return outputs, pred_target
    return pred_target


def predict_from_data_list(data:Union[Data, List[Data]], model, batch_size=128, device='cpu'):
    loader = None
    if isinstance(data, List):
        loader = DataLoader(dataset=data, batch_size=batch_size)
    elif isinstance(data, Data):
        loader = DataLoader(dataset=[data], batch_size=batch_size)
    else:
        raise TypeError("data must be an instance of the following classes: Data, or List.")
    
    return predict_from_loader(loader, model, device=device)

def predict_from_smiles_list(model, smiles_list:List[str], atom_featurizer:AtomFeaturizer=ATOM_FEATURIZER
                             , bond_featurizer:BondFeaturizer=BOND_FEATURIZER
                             , batch_size=128, device='cpu', add_explicit_h:bool=True):
    graphs=graph_from_smiles_list(smiles_list, add_explicit_h=add_explicit_h)
    return predict_from_data_list(graphs, model=model, batch_size=batch_size, device=device)