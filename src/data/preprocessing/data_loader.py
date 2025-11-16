import pandas as pd
import os
import requests
from tqdm import tqdm
from utils.paths import TRAIN_DATA_PATH, TEST_DATA_PATH


class DataLoader:
    def __init__(self):

        self.train_data = pd.read_csv(TRAIN_DATA_PATH)
        self.test_data = pd.read_csv(TEST_DATA_PATH)

        self.train_data = self.train_data[self.train_data["Pubchem_ID"] > 0]
        self.test_data = self.test_data[self.test_data["Pubchem_ID"] > 0]


    def get_data(self, download: bool = False):

        if download:
            tqdm.pandas(desc="Fetching SMILES")

            self.train_data['SMILES'] = self.train_data['Pubchem_ID'].progress_apply(
                self._get_smiles_from_pubchem
            )

            self.test_data['SMILES'] = self.test_data['Pubchem_ID'].progress_apply(
                self._get_smiles_from_pubchem
            )

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
