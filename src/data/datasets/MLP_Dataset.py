import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class MoleculeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train=True):
        self.train = train
        
        if train:
            self.ids = df["Pubchem_ID"].values
            labels = df["taste_cluster"].values.astype(int)
            self.y = labels.astype(np.int64)

            self.X = df.drop(columns=["Pubchem_ID", "taste_cluster", "SMILES"]).values.astype(np.float32)
        else:
            self.ids = df["Pubchem_ID"].values
            self.X = df.drop(columns=["Pubchem_ID", "SMILES"]).values.astype(np.float32)
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.train:
            y = torch.tensor(self.y[idx], dtype=torch.long)
            return x, y
        else:
            return x, self.ids[idx]

