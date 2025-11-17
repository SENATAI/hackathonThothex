import torch
from torch.utils.data import Dataset

class SMILESDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128, mode = "train"):
        self.smiles = df["SMILES"].astype(str).tolist()
        self.ids = df["Pubchem_ID"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode

        if mode == "train":
            self.labels = df["taste_cluster"].tolist()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.smiles[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        if self.mode == "train":
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)
            }
        else:
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "id": self.ids[idx]
            }
    

