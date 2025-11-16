# from data.preprocessing.preparator import Cleaner
from data.preprocessing.data_loader import DataLoader

class Pipeline:
    def __init__(self, split: str, columns_to_clean: list = ['']):
        self.prepared_data = DataLoader(split).get_data()
        # self.cleaner = Cleaner(split, columns_to_clean)

    def get_raw_data(self):
        return ...
    
    def get_prepared_data(self):
        return self.prepared_data