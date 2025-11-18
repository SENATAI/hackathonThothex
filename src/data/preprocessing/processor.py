import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from data.preprocessing.extractor import Extractor


class Processor(Extractor):
    def __init__(
        self,
        morgan_bits: int = 2048,
        morgan_radius: int = 2,
        pca_variance: float = 0.95,
        feature_type: str = "both",  # morgan | rdkit | both
        use_bert: bool = True,
        download: bool = False
    ):
        self.use_bert = use_bert

        super().__init__(
            morgan_bits=morgan_bits,
            morgan_radius=morgan_radius,
            download=download
        )

        self.pca_variance = pca_variance
        self.feature_type = feature_type.lower()

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.pca_variance, svd_solver="full")

        self.processed_train_data = self._process_train()
        self.processed_test_data = self._process_test()

    # -----------------------------------------------------------
    def _get_features(self, part_list):
        """Simply concatenate blocks already preprocessed."""
        return np.hstack(part_list)

    # -----------------------------------------------------------
    def _process_train(self):
        # -------------------------
        # RDKit descriptors: scale + PCA
        # -------------------------
        desc_scaled = self.scaler.fit_transform(self._desc_train)
        desc_pca = self.pca.fit_transform(desc_scaled)

        # -------------------------
        # Build training feature matrix
        # -------------------------
        blocks = []

        if self.feature_type in ["morgan", "both"]:
            blocks.append(self._morgan_train)

        if self.feature_type in ["rdkit", "both"]:
            blocks.append(desc_pca)

        if self.use_bert:
            blocks.append(self._bert_train)

        X = self._get_features(blocks)

        # -------------------------
        # Combine with raw dataframe
        # -------------------------
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        features_df = pd.DataFrame(X, columns=feature_names, index=self.raw_train_data.index)

        df = pd.concat([self.raw_train_data, features_df], axis=1)
        return df

    # -----------------------------------------------------------
    def _process_test(self):
        # -------------------------
        # Use pre-fitted scaler + PCA
        # -------------------------
        desc_scaled = self.scaler.transform(self._desc_test)
        desc_pca = self.pca.transform(desc_scaled)

        blocks = []

        if self.feature_type in ["morgan", "both"]:
            blocks.append(self._morgan_test)

        if self.feature_type in ["rdkit", "both"]:
            blocks.append(desc_pca)

        if self.use_bert:
            blocks.append(self._bert_test)

        X = self._get_features(blocks)

        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        features_df = pd.DataFrame(X, columns=feature_names, index=self.raw_test_data.index)

        df = pd.concat([self.raw_test_data, features_df], axis=1)
        return df
