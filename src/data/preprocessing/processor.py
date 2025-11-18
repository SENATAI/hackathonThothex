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
        feature_type: str = "both",  # Supports 'morgan', 'rdkit', 'dft', 'both', 'both+dft', 'rdkit+dft', 'morgan+dft'
        use_pca_rdkit: bool = False,  # Enable/disable PCA for RDKit
        use_pca_dft: bool = False,   # Enable/disable PCA for DFT (default off since low-dim)
        remove_zero_var: bool = True,  # Remove zero-variance columns after scaling
        download: bool = False
    ):
        super().__init__(morgan_bits=morgan_bits, morgan_radius=morgan_radius, download=download)

        self.pca_variance = pca_variance
        self.feature_type = feature_type.lower()

        valid_types = ["morgan", "rdkit", "dft", "both", "both+dft", "rdkit+dft", "morgan+dft"]
        if self.feature_type not in valid_types:
            raise ValueError(f"feature_type must be one of: {valid_types}")

        self.use_pca_rdkit = use_pca_rdkit
        self.use_pca_dft = use_pca_dft
        self.remove_zero_var = remove_zero_var

        self.scaler_desc = StandardScaler()
        self.scaler_dft = StandardScaler()
        self.pca_rdkit = PCA(n_components=self.pca_variance, svd_solver="full") if self.use_pca_rdkit else None
        self.pca_dft = PCA(n_components=self.pca_variance, svd_solver="full") if self.use_pca_dft else None

        self.processed_train_data = self._process_train()
        self.processed_test_data = self._process_test()

    def _remove_zero_variance(self, X_scaled):
        """Remove columns with zero variance (std == 0)."""
        if X_scaled.shape[1] == 0:
            return X_scaled
        std = np.std(X_scaled, axis=0)
        keep_cols = std > 0
        return X_scaled[:, keep_cols]

    def _get_features(self, morgan, desc_scaled, dft_scaled, pca_fit=False):
        # Apply PCA if enabled
        if self.use_pca_rdkit and self.pca_rdkit:
            desc_processed = self.pca_rdkit.fit_transform(desc_scaled) if pca_fit else self.pca_rdkit.transform(desc_scaled)
        else:
            desc_processed = desc_scaled

        if self.use_pca_dft and self.pca_dft:
            dft_processed = self.pca_dft.fit_transform(dft_scaled) if pca_fit else self.pca_dft.transform(dft_scaled)
        else:
            dft_processed = dft_scaled

        # Remove zero-var if enabled (after PCA/scaling)
        if self.remove_zero_var:
            desc_processed = self._remove_zero_variance(desc_processed)
            dft_processed = self._remove_zero_variance(dft_processed)
            # Morgan is binary, but we can remove all-zero cols too
            morgan = self._remove_zero_variance(morgan)

        # Combine based on feature_type
        if self.feature_type == "morgan":
            X = morgan
        elif self.feature_type == "rdkit":
            X = desc_processed
        elif self.feature_type == "dft":
            X = dft_processed
        elif self.feature_type == "both":
            X = np.hstack([morgan, desc_processed])
        elif self.feature_type == "both+dft":
            X = np.hstack([morgan, desc_processed, dft_processed])
        elif self.feature_type == "rdkit+dft":
            X = np.hstack([desc_processed, dft_processed])
        elif self.feature_type == "morgan+dft":
            X = np.hstack([morgan, dft_processed])
        return X

    def _process_train(self):
        desc_scaled = self.scaler_desc.fit_transform(self._desc_train)
        dft_scaled = self.scaler_dft.fit_transform(self._dft_train)
        X = self._get_features(self._morgan_train, desc_scaled, dft_scaled, pca_fit=True)

        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        features_df = pd.DataFrame(X, columns=feature_names, index=self.raw_train_data.index)

        df = pd.concat([self.raw_train_data, features_df], axis=1)
        return df

    def _process_test(self):
        desc_scaled = self.scaler_desc.transform(self._desc_test)
        dft_scaled = self.scaler_dft.transform(self._dft_test)
        X = self._get_features(self._morgan_test, desc_scaled, dft_scaled, pca_fit=False)

        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        features_df = pd.DataFrame(X, columns=feature_names, index=self.raw_test_data.index)

        df = pd.concat([self.raw_test_data, features_df], axis=1)
        return df