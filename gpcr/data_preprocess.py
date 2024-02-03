import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

import torch
from torch.utils import data
import pickle
import sys
import os
from subword_nmt.apply_bpe import BPE
import codecs
import pandas as pd
sys.path.append('..')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
num_atom_feat = 75
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom,
                  explicit_H=False,
                  use_chirality=False):
    results = one_of_k_encoding_unk(
      atom.GetSymbol(),
      [
        'C',
        'N',
        'O',
        'S',
        'F',
        'Si',
        'P',
        'Cl',
        'Br',
        'Mg',
        'Na',
        'Ca',
        'Fe',
        'As',
        'Al',
        'I',
        'B',
        'V',
        'K',
        'Tl',
        'Yb',
        'Sb',
        'Sn',
        'Ag',
        'Pd',
        'Co',
        'Se',
        'Ti',
        'Zn',
        'H',
        'Li',
        'Ge',
        'Cu',
        'Au',
        'Ni',
        'Cd',
        'In',
        'Mn',
        'Zr',
        'Cr',
        'Pt',
        'Hg',
        'Pb',
        'other'
      ]) + one_of_k_encoding(atom.GetDegree(),
                             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
      results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                [0, 1, 2, 3, 4])
    if use_chirality:
      try:
        results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
      except:
        results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    return results

def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)+np.eye(adjacency.shape[0])

class Data_Encoder(data.Dataset):

    def __init__(self,txtpath):
        'Initialization'
        vocab_path = '../ESPF/protein_codes_uniprot.txt'
        bpe_codes_protein = codecs.open(vocab_path)
        self.pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
        sub_csv = pd.read_csv('../ESPF/subword_units_map_uniprot.csv')

        idx2word_p = sub_csv['index'].values
        self.words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

        vocab_path = '../ESPF/drug_codes_chembl.txt'
        bpe_codes_drug = codecs.open(vocab_path)
        self.dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
        sub_csv = pd.read_csv('../ESPF/subword_units_map_chembl.csv')
        idx2word_d = sub_csv['index'].values

        self.words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))
        with open(txtpath, "r") as f:
            data_list=f.read().strip().split('\n')
        smiles, sequences, interactions =[], [], []
        for no, data in enumerate(data_list):
            smile, sequence, interaction = data.strip().split()
            smiles.append(smile)
            sequences.append(sequence)
            interactions.append(interaction)

        self.smiles = smiles
        self.Sequences = sequences
        self.interactions = interactions

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.interactions)

    def __getitem__(self, index):
        'Generates one sample of data'
        d = self.smiles[index]
        p = self.Sequences[index]
        label = np.array(self.interactions[index], dtype=np.float32)
        d_v, input_mask_d = self.drug2emb_encoder(d)
        p_v, input_mask_p = self.protein2emb_encoder(p)

        atom_feature, adj, num_size = self.mol_features(d)

        d_v = torch.FloatTensor(d_v)
        p_v = torch.FloatTensor(p_v)
        label = torch.LongTensor(label)
        input_mask_d = torch.LongTensor(input_mask_d)
        input_mask_p = torch.LongTensor(input_mask_p)

        atom_feature = torch.FloatTensor(atom_feature)
        adj = torch.FloatTensor(adj)
        num_size = torch.tensor(num_size)

        sample = {'atom_feature': atom_feature, 'adj': adj, 'num_size': num_size, 'd_v':d_v,
                  'p_v': p_v, 'input_mask_d': input_mask_d, 'input_mask_p': input_mask_p, 'label':label,}

        return sample

    def drug2emb_encoder(self, x):
        max_d = 50
        t1 = self.dbpe.process_line(x).split()
        try:
            i1 = np.asarray([self.words2idx_d[i] for i in t1])
        except:
            i1 = np.array([0])

        l = len(i1)

        if l < max_d:
            i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
            input_mask = ([1] * l) + ([0] * (max_d - l))

        else:
            i = i1[:max_d]
            input_mask = [1] * max_d

        return i, np.asarray(input_mask)

    def protein2emb_encoder(self, x):
        max_p = 545
        t1 = self.pbpe.process_line(x).split()
        try:
            i1 = np.asarray([self.words2idx_p[i] for i in t1])
        except:
            i1 = np.array([0])

        l = len(i1)

        if l < max_p:
            i = np.pad(i1, (0, max_p - l), 'constant', constant_values=0)
            input_mask = ([1] * l) + ([0] * (max_p - l))
        else:
            i = i1[:max_p]
            input_mask = [1] * max_p

        return i, np.asarray(input_mask)

    def mol_features(self,smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            raise RuntimeError("SMILES cannot been parsed!")
        num_size = Chem.MolFromSmiles(smiles).GetNumAtoms()
        atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
        for atom in mol.GetAtoms():
            atom_feat[atom.GetIdx(), :] = atom_features(atom)
        adj_matrix = adjacent_matrix(mol)
        return atom_feat, adj_matrix, num_size



