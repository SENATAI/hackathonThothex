import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
from data.preprocessing.data_loader import DataLoader
from PIL import ImageOps

class Extractor:
    def __init__(
        self,
        morgan_bits: int = 2048,
        morgan_radius: int = 2,
        img_size: tuple = (224, 224),
        download: bool = False
    ):
        loader = DataLoader()
        self.raw_train_data, self.raw_test_data = loader.get_data(download=download)

        self.morgan_bits = morgan_bits
        self.morgan_radius = morgan_radius
        self.img_size = img_size
        self.morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.morgan_radius,
            fpSize=self.morgan_bits
        )

        self.descriptor_fns = [fn for _, fn in Descriptors._descList]
        self.n_descriptors = len(self.descriptor_fns)

        # DFT columns
        self.dft_cols = ['G', 'HOMO', 'LUMO', 'Gap', 'Dipole', 'E', 'CavityVolume', 'CavitySurface']
        self.n_dft = len(self.dft_cols)

        # Хранение признаков
        self._morgan_train = None
        self._desc_train = None
        self._dft_train = None
        self._img_train = None

        self._morgan_test = None
        self._desc_test = None
        self._dft_test = None
        self._img_test = None

        self._extract_raw_features()
    
    def _extract_raw(self, smiles_list):
        """
        Возвращает (morgan_array, desc_array, img_array)
        - morgan_array: shape (N, morgan_bits) dtype=np.int8
        - desc_array: shape (N, n_descriptors) dtype=np.float32
        - img_array: shape (N, H, W, 3) dtype=np.uint8
        """
        morgan_list = []
        desc_list = []
        img_list = []

        for smi in tqdm(smiles_list, desc="Extracting raw features", leave=False):
            if smi is None or (isinstance(smi, float) and np.isnan(smi)):
                # отсутствующая запись -> нули
                morgan_list.append(np.zeros(self.morgan_bits, dtype=np.int8))
                desc_list.append(np.zeros(self.n_descriptors, dtype=np.float32))
                img_list.append(np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8))
                continue

            mol = None
            try:
                mol = Chem.MolFromSmiles(smi)
            except Exception:
                mol = None

            if mol is None:
                morgan_list.append(np.zeros(self.morgan_bits, dtype=np.int8))
                desc_list.append(np.zeros(self.n_descriptors, dtype=np.float32))
                img_list.append(np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8))
                continue

            # Morgan fingerprint -> numpy array
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.morgan_radius, nBits=self.morgan_bits)
                arr = np.zeros((self.morgan_bits,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(fp, arr)  # заполняет arr значениями 0/1
                morgan_list.append(arr)
            except Exception:
                morgan_list.append(np.zeros(self.morgan_bits, dtype=np.int8))

            # RDKit дескрипторы
            desc_vals = []
            for fn in self.descriptor_fns:
                try:
                    v = fn(mol)
                    # некоторые дескрипторы возвращают tuple — берем первый элемент
                    if isinstance(v, (tuple, list)):
                        v = v[0] if len(v) > 0 else 0.0
                    # None -> 0.0
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        v = 0.0
                    desc_vals.append(float(v))
                except Exception:
                    desc_vals.append(0.0)
            desc_list.append(np.array(desc_vals, dtype=np.float32))

            # Изображение молекулы
            try:
                pil_img = Draw.MolToImage(mol, size=self.img_size)  # PIL Image
                pil_img = pil_img.convert("RGB")
                # Убедимся в размере (fit на случай, если Draw вернул не тот размер)
                pil_img = pil_img.resize((self.img_size[0], self.img_size[1]))
                img_arr = np.array(pil_img, dtype=np.uint8)
                img_list.append(img_arr)
            except Exception:
                img_list.append(np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8))

        # Конвертация в numpy массивы
        morgan_arr = np.vstack(morgan_list).astype(np.int8) if len(morgan_list) > 0 else np.zeros((0, self.morgan_bits), dtype=np.int8)
        desc_arr = np.vstack(desc_list).astype(np.float32) if len(desc_list) > 0 else np.zeros((0, self.n_descriptors), dtype=np.float32)
        img_arr = np.stack(img_list, axis=0).astype(np.uint8) if len(img_list) > 0 else np.zeros((0, self.img_size[1], self.img_size[0], 3), dtype=np.uint8)

        return morgan_arr, desc_arr, img_arr
    
    def _extract_dft(self, df):
        """Extracts DFT features as numpy array."""
        return df[self.dft_cols].values.astype(np.float32)

    def _extract_raw_features(self):
        """Performs feature extraction once during initialization."""
        train_smiles = self.raw_train_data["SMILES"].tolist()
        test_smiles = self.raw_test_data["SMILES"].tolist()

        self._morgan_train, self._desc_train, self._img_train = self._extract_raw(train_smiles)
        self._morgan_test, self._desc_test, self._img_test = self._extract_raw(test_smiles)

        self._dft_train = self._extract_dft(self.raw_train_data)
        self._dft_test = self._extract_dft(self.raw_test_data)