from data.preprocessing.processor import Processor

class Pipeline(Processor):
    def __init__(
        self,
        morgan_bits: int = 2048,
        morgan_radius: int = 2,
        pca_variance: float = 0.95,
        feature_type: str = "both",
        use_pca_rdkit: bool = False,
        use_pca_dft: bool = False,
        remove_zero_var: bool = True,
        download: bool = False
    ):
        super().__init__(
            morgan_bits=morgan_bits,
            morgan_radius=morgan_radius,
            pca_variance=pca_variance,
            feature_type=feature_type,
            use_pca_rdkit=use_pca_rdkit,
            use_pca_dft=use_pca_dft,
            remove_zero_var=remove_zero_var,
            download=download
        )

    def get_raw_data(self):
        """Returns original DataFrames with SMILES and labels."""
        return self.raw_train_data, self.raw_test_data

    def get_raw_features(self):
        """Returns raw Morgan, RDKit descriptors, and DFT."""
        return (
            self._morgan_train, self._desc_train, self._dft_train,
            self._morgan_test, self._desc_test, self._dft_test
        )

    def get_processed_data(self):
        """Returns processed DataFrames with numerical features (based on feature_type)."""
        return self.processed_train_data, self.processed_test_data

    def get_images(self):
        """Returns molecule images: (train_images, test_images) in (N, 224, 224, 3)."""
        return self._img_train, self._img_test