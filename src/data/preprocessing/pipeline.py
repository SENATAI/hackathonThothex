from data.preprocessing.GNNExtractor import GNNExtractor


class Pipeline(GNNExtractor):
    def __init__(
        self,
        download: bool = False
    ):
        super().__init__(download=download)

    def get_raw_data(self):
        return self.raw_train_data, self.raw_test_data

    def get_processed_data(self):
        return (
            (self._processed_train_graphs, self._processed_train_targets),
            (self._processed_test_graphs, self._processed_test_targets)
        )