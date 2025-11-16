import numpy as np
from deepchem.feat import DMPNNFeaturizer
from tqdm import tqdm
from data.preprocessing.data_loader import DataLoader
from typing import List, Tuple
import pandas as pd


class GNNExtractor:
    def __init__(
        self,
        download: bool = False
    ):
        loader = DataLoader()
        self.raw_train_data, self.raw_test_data = loader.get_data(download=download)

        self.featurizer = DMPNNFeaturizer()

        self._processed_train_graphs: List = []
        self._processed_train_targets: np.ndarray = np.array([])
        self._processed_test_graphs: List = []
        self._processed_test_targets: np.ndarray = np.array([])

        self._process_data()

    def _process_split(self, df: pd.DataFrame) -> Tuple[List, np.ndarray]:
        """Преобразует DataFrame в (графы, таргеты), пропуская некорректные SMILES."""
        graphs = []
        targets = []
        smiles_list = df["SMILES"].tolist()
        target_array = df.drop(columns=["SMILES"]).values.astype(int)

        for i, smi in enumerate(tqdm(smiles_list, desc="Processing molecules")):
            mol = self.featurizer.featurize([smi])
            if len(mol) == 0 or mol[0] is None:
                print(f"Skipping invalid SMILES: {smi}")
                continue
            graphs.append(mol[0])
            targets.append(target_array[i])

        targets = np.array(targets, dtype=int)
        return graphs, targets

    def _process_data(self):
        """Обрабатывает train и test, синхронизируя графы и таргеты."""
        self._processed_train_graphs, self._processed_train_targets = self._process_split(self.raw_train_data)
        self._processed_test_graphs, self._processed_test_targets = self._process_split(self.raw_test_data)