from data.preprocessing.extractor import Extractor

class Pipeline(Extractor):
    def __init__(self, 
                morgan_bits: int = 2048,
                morgan_radius: int = 2,
                pca_variance: float = 0.95, 
                download: bool = False
                ):
        super().__init__(morgan_bits, morgan_radius, pca_variance, download)

    def get_raw_data(self):
        return self.raw_train_data, self.raw_test_data
    
    def get_processed_data(self):
        return self.processed_train_data, self.processed_test_data