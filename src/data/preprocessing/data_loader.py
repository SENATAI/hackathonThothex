import pandas as pd
import requests
from tqdm import tqdm

from utils.paths import TRAIN_DATA_PATH, TEST_DATA_PATH

class DataLoader:
    def __init__(self, split: str):
        if split == 'train':
            self.data = pd.read_csv(TRAIN_DATA_PATH)
        else:
            self.data = pd.read_csv(TEST_DATA_PATH)

        # убираем некорректные CID
        self.data = self.data[self.data["Pubchem_ID"] > 0]

        # подключаем tqdm для apply
        tqdm.pandas(desc="Fetching SMILES")

        # получение SMILES
        self.data['SMILES'] = self.data['Pubchem_ID'].progress_apply(
            self._get_smiles_from_pubchem
        )

    def get_data(self):
        return self.data

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
