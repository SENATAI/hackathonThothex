import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings

from data.preprocessing.data_loader import DataLoader

warnings.filterwarnings("ignore")


class Extractor:
    def __init__(
        self,
        morgan_bits=2048,
        morgan_radius=2,
        pca_variance=0.95,
        download: bool = False
    ):

        loader = DataLoader()
        self.raw_train_data, self.raw_test_data = loader.get_data(download=download)

        self.processed_train_data = self.raw_train_data.copy()
        self.processed_test_data = self.raw_test_data.copy()

        self.morgan_bits = morgan_bits
        self.morgan_radius = morgan_radius
        self.pca_variance = pca_variance

        self.scaler = None
        self.pca = None

        self.descriptor_fns = [d[1] for d in Descriptors._descList]

        self._process_train()
        self._process_test()


    def morgan_fp(self, mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.morgan_radius, nBits=self.morgan_bits
        )
        return np.array(fp)


    def rdkit_desc(self, mol):
        if mol is None:
            return np.zeros(len(self.descriptor_fns))
        return np.array([fn(mol) for fn in self.descriptor_fns])


    def extract_raw(self, smiles_list):
        morgan_list = []
        rdkit_list = []

        for smi in tqdm(smiles_list, desc="Extracting raw features"):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                morgan = np.zeros(self.morgan_bits)
                rdkit = np.zeros(len(self.descriptor_fns))
            else:
                morgan = self.morgan_fp(mol)
                rdkit = self.rdkit_desc(mol)

            morgan_list.append(morgan)
            rdkit_list.append(rdkit)

        return np.array(morgan_list), np.array(rdkit_list)


    def _process_train(self):
        smiles = self.raw_train_data["SMILES"].tolist()

        morgan_raw, rdkit_raw = self.extract_raw(smiles)

        self.scaler = StandardScaler()
        rdkit_scaled = self.scaler.fit_transform(rdkit_raw)

        self.pca = PCA(n_components=self.pca_variance, svd_solver="full")
        rdkit_pca = self.pca.fit_transform(rdkit_scaled)

        X = np.hstack([morgan_raw, rdkit_pca])

        for i in range(X.shape[1]):
            self.processed_train_data[f"feat_{i}"] = X[:, i]


    def _process_test(self):
        if self.scaler is None or self.pca is None:
            raise RuntimeError("Train must be processed before test.")

        smiles = self.raw_test_data["SMILES"].tolist()

        morgan_raw, rdkit_raw = self.extract_raw(smiles)

        rdkit_scaled = self.scaler.transform(rdkit_raw)
        rdkit_pca = self.pca.transform(rdkit_scaled)

        X = np.hstack([morgan_raw, rdkit_pca])

        for i in range(X.shape[1]):
            self.processed_test_data[f"feat_{i}"] = X[:, i]


    def get_train(self):
        return self.processed_train_data

    def get_test(self):
        return self.processed_test_data

    def get_scaler(self):
        return self.scaler

    def get_pca(self):
        return self.pca
