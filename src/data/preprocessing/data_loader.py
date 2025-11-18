import pandas as pd
import os
import requests
from tqdm import tqdm
from utils.paths import TRAIN_DATA_PATH, TEST_DATA_PATH, DATA_DIR
from pathlib import Path

class DataLoader:
    def __init__(self):
        self.train_data = pd.read_csv(TRAIN_DATA_PATH)
        self.test_data = pd.read_csv(TEST_DATA_PATH)

        self.train_data = self.train_data[self.train_data["Pubchem_ID"] > 0]
        self.test_data = self.test_data[self.test_data["Pubchem_ID"] > 0]

        # Load and merge DFT data
        #dft_path = (Path(DATA_DIR) / 'pubchem_dft.csv')
        dft_path = r'C:\Users\edtop\ITMO\thotex_hack\thortex-hackaton\data\pubchem_dft.csv'
        dft_data = pd.read_csv(dft_path)
        dft_columns = ['Pubchem_ID', 'G', 'HOMO', 'LUMO', 'Gap', 'Dipole', 'E', 'CavityVolume', 'CavitySurface']

        self.train_data = self.train_data.merge(dft_data[dft_columns], on='Pubchem_ID', how='left', suffixes=('', '_dft'))
        self.test_data = self.test_data.merge(dft_data[dft_columns], on='Pubchem_ID', how='left', suffixes=('', '_dft'))

        # If DFT has SMILES or taste_cluster, you can optionally use them to fill NaNs in main data
        # For example: self.train_data['SMILES'] = self.train_data['SMILES'].fillna(self.train_data.get('SMILES_dft', pd.Series()))

    def get_data(self, download: bool = False):
        if download:
            tqdm.pandas(desc="Fetching SMILES")
            self.train_data['SMILES'] = self.train_data['Pubchem_ID'].progress_apply(self._get_smiles_from_pubchem)
            self.test_data['SMILES'] = self.test_data['Pubchem_ID'].progress_apply(self._get_smiles_from_pubchem)

        # Fill NaN in DFT columns with 0 (or use .fillna(self.train_data['G'].mean()) for mean imputation)
        dft_cols = ['G', 'HOMO', 'LUMO', 'Gap', 'Dipole', 'E', 'CavityVolume', 'CavitySurface']
        self.train_data[dft_cols] = self.train_data[dft_cols].fillna(0)
        self.test_data[dft_cols] = self.test_data[dft_cols].fillna(0)

        return self.train_data, self.test_data

    def _get_smiles_from_pubchem(self, cid):
        if pd.isna(cid):
            return None
        try:
            cid = int(cid)
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/TXT"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.text.strip()
            return None
        except Exception as e:
            print(f"Error fetching CID {cid}: {e}")
            return None