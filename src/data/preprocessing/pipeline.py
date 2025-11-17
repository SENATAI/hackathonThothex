from data.preprocessing.processor import Processor


class Pipeline(Processor):
    def __init__(
        self,
        morgan_bits: int = 2048,
        morgan_radius: int = 2,
        pca_variance: float = 0.95,
        feature_type: str = "both",
        download: bool = False
    ):
        super().__init__(
            morgan_bits=morgan_bits,
            morgan_radius=morgan_radius,
            pca_variance=pca_variance,
            feature_type=feature_type,
            download=download
        )

    def get_raw_data(self):
        """Возвращает исходные DataFrame'ы с SMILES и метками."""
        return self.raw_train_data, self.raw_test_data

    def get_raw_features(self):
        """Возвращает сырые Morgan и RDKit дескрипторы."""
        return self._morgan_train, self._desc_train, self._morgan_test, self._desc_test

    def get_processed_data(self):
        """Возвращает обработанные DataFrame'ы с числовыми признаками (Morgan/RDKit/оба)."""
        return self.processed_train_data, self.processed_test_data

    def get_images(self):
        """Возвращает изображения молекул: (train_images, test_images) в формате (N, 224, 224, 1)."""
        return self._img_train, self._img_test