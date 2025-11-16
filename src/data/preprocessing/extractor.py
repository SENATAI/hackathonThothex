import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
from data.preprocessing.data_loader import DataLoader


class Extractor:
    def __init__(
        self,
        morgan_bits: int = 2048,
        morgan_radius: int = 2,
        download: bool = False
    ):
        loader = DataLoader()
        self.raw_train_data, self.raw_test_data = loader.get_data(download=download)

        self.morgan_bits = morgan_bits
        self.morgan_radius = morgan_radius
        self.morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.morgan_radius,
            fpSize=self.morgan_bits
        )

        self.descriptor_fns = [fn for _, fn in Descriptors._descList]
        self.n_descriptors = len(self.descriptor_fns)

        self._morgan_train = None
        self._desc_train = None
        self._morgan_test = None
        self._desc_test = None

        self._extract_raw_features()

    def morgan_fp(self, mol):
        if mol is None:
            return np.zeros(self.morgan_bits, dtype=np.int8)
        fp = self.morgan_generator.GetFingerprint(mol)
        return np.array(fp, dtype=np.int8)

    def rdkit_desc(self, mol):
        if mol is None:
            return np.zeros(self.n_descriptors, dtype=np.float32)
        return np.array([fn(mol) for fn in self.descriptor_fns], dtype=np.float32)

    def extract_raw(self, smiles_list):
        morgan_list = []
        rdkit_list = []
        for smi in tqdm(smiles_list, desc="Extracting raw features"):
            mol = Chem.MolFromSmiles(smi)
            morgan_list.append(self.morgan_fp(mol))
            rdkit_list.append(self.rdkit_desc(mol))
        return np.stack(morgan_list), np.stack(rdkit_list)

    def _extract_raw_features(self):
        """Извлекает сырые признаки для train и test один раз при инициализации."""
        train_smiles = self.raw_train_data["SMILES"].tolist()
        test_smiles = self.raw_test_data["SMILES"].tolist()

        self._morgan_train, self._desc_train = self.extract_raw(train_smiles)
        self._morgan_test, self._desc_test = self.extract_raw(test_smiles)