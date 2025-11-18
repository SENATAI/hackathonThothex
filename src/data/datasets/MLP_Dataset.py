import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision import transforms


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
        

class MoleculeImageDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.uint8)  # uint8
        if self.transform:
            img = self.transform(img)  # transforms.ToTensor() + Normalize
        else:
            img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float) / 255.0  # HWC -> CHW, normalize to [0,1]
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, label
        else:
            return img

class MultimodalDataset(Dataset):
    def __init__(self, images, df: pd.DataFrame, train=True, transform=None):
        """
        Multimodal dataset combining images and tabular features (DFT + RDKit from df).
        - images: np.array of shape (N, H, W, C)
        - df: processed DataFrame with 'Pubchem_ID', 'taste_cluster' (if train), 'SMILES', and 'feat_*' columns
        """
        self.images = images
        self.transform = transform
        self.train = train
        
        # Extract tabular features (assume all 'feat_*' columns are numerical features)
        feat_cols = [col for col in df.columns if col.startswith('feat_')]
        
        if train:
            self.ids = df["Pubchem_ID"].values
            self.labels = df["taste_cluster"].values.astype(int)
            self.tabular = df[feat_cols].values.astype(np.float32)
        else:
            self.ids = df["Pubchem_ID"].values
            self.tabular = df[feat_cols].values.astype(np.float32)
            self.labels = None

        # Ensure alignment
        assert len(self.images) == len(self.tabular), "Images and tabular data must have the same number of samples"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.uint8)
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float) / 255.0  # HWC -> CHW, normalize to [0,1]
        
        tab = torch.tensor(self.tabular[idx], dtype=torch.float)
        
        if self.train:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, tab, label
        else:
            return img, tab, self.ids[idx]