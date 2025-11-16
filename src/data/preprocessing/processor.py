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
        download: bool = False
    ):
        super().__init__(morgan_bits=morgan_bits, morgan_radius=morgan_radius, download=download)

        self.pca_variance = pca_variance
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.pca_variance, svd_solver="full")

        self.processed_train_data = self._process_train()
        self.processed_test_data = self._process_test()

    def _process_train(self):
        desc_scaled = self.scaler.fit_transform(self._desc_train)
        desc_pca = self.pca.fit_transform(desc_scaled)

        X = np.hstack([self._morgan_train, desc_pca])

        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        features_df = pd.DataFrame(X, columns=feature_names, index=self.raw_train_data.index)

        df = pd.concat([self.raw_train_data, features_df], axis=1)
        return df


    def _process_test(self):
        desc_scaled = self.scaler.transform(self._desc_test)
        desc_pca = self.pca.transform(desc_scaled)

        X = np.hstack([self._morgan_test, desc_pca])

        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        features_df = pd.DataFrame(X, columns=feature_names, index=self.raw_test_data.index)

        df = pd.concat([self.raw_test_data, features_df], axis=1)
        return df