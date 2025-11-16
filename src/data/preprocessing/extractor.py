import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Descriptors3D
from rdkit.Chem import AllChem
import numpy as np

from data.preprocessing.data_loader import DataLoader

class Extractor:
    def __init__(self):
        data_loader = DataLoader('train')
        self.data = data_loader.get_data().copy()
        if 'SMILES' not in self.data.columns:
            raise ValueError("Колонка 'SMILES' отсутствует. Сначала загрузите данные с SMILES.")

        # Генерируем признаки
        self.features_df = self._generate_features()

    def _smiles_to_mol(self, smiles):
        """Преобразует SMILES в RDKit молекулу."""
        if not isinstance(smiles, str) or not smiles.strip():
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            # Генерируем 3D координаты для 3D-дескрипторов
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)
            return mol
        except Exception:
            return None

    def _get_rdkit_descriptors(self, mol):
        """Извлекает основные RDKit дескрипторы."""
        if mol is None:
            return {name: np.nan for name, _ in Descriptors.descList}
        return {name: func(mol) for name, func in Descriptors.descList}

    def _get_3d_descriptors(self, mol):
        """Извлекает 3D-дескрипторы (требуют 3D-оптимизацию)."""
        if mol is None:
            return {name: np.nan for name, _ in Descriptors3D.descList}
        return {name: func(mol) for name, func in Descriptors3D.descList}

    def _generate_features(self):
        """Основная функция генерации признаков."""
        features_list = []
        for _, row in self.data.iterrows():
            mol = self._smiles_to_mol(row['SMILES'])
            desc_1d2d = self._get_rdkit_descriptors(mol)
            desc_3d = self._get_3d_descriptors(mol)

            # Объединяем все дескрипторы
            features = {**desc_1d2d, **desc_3d}
            features_list.append(features)

        # Создаём DataFrame с признаками
        features_df = pd.DataFrame(features_list, index=self.data.index)

        # Удаляем константные/неполезные признаки (по желанию)
        features_df = features_df.select_dtypes(include=[np.number])

        return features_df

    def get_features(self):
        """Возвращает датафрейм с признаками."""
        return self.features_df

    def get_full_data_with_features(self):
        """Возвращает исходные данные + признаки."""
        return pd.concat([self.data, self.features_df], axis=1)