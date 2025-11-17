import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
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

        # Хранение признаков
        self._morgan_train = None
        self._desc_train = None
        self._img_train = None

        self._morgan_test = None
        self._desc_test = None
        self._img_test = None

        self._extract_raw_features()

    def morgan_fp(self, mol):
        """Возвращает Morgan fingerprint фиксированной длины."""
        if mol is None:
            return np.zeros(self.morgan_bits, dtype=np.int8)
        fp = self.morgan_generator.GetFingerprint(mol)
        return np.array(fp.ToList(), dtype=np.int8)

    def rdkit_desc(self, mol):
        """Возвращает вектор RDKit-дескрипторов."""
        if mol is None:
            return np.zeros(self.n_descriptors, dtype=np.float32)
        return np.array([fn(mol) for fn in self.descriptor_fns], dtype=np.float32)

    def _mol_to_image(self, mol):
        """
        Преобразует молекулу в RGB изображение формы (244, 244, 3), нормализованное к [0, 1].
        Линии молекулы яркие, фон темный.
        """
        target_size = (244, 244)

        if mol is None:
            return np.zeros((*target_size, 3), dtype=np.float32)

        try:
            Chem.rdDepictor.Compute2DCoords(mol)

            img = Draw.MolToImage(mol, size=target_size, fitImage=True)

            img_gray = img.convert("L")
            img_gray = ImageOps.invert(img_gray)

            img_array = np.array(img_gray, dtype=np.float32) / 255.0

            img_rgb = np.stack([img_array]*3, axis=-1)

            return img_rgb

        except Exception:
            return np.zeros((*target_size, 3), dtype=np.float32)

    def extract_raw(self, smiles_list):
        """Извлекает Morgan-фингерпринты, дескрипторы и изображения по списку SMILES."""
        morgan_list = []
        rdkit_list = []
        img_list = []

        for smi in tqdm(smiles_list, desc="Extracting raw features"):
            mol = None
            if isinstance(smi, str) and smi.strip():
                try:
                    mol = Chem.MolFromSmiles(smi.strip())
                except Exception:
                    mol = None

            morgan_list.append(self.morgan_fp(mol))
            rdkit_list.append(self.rdkit_desc(mol))
            img_list.append(self._mol_to_image(mol))

        return (
            np.stack(morgan_list),      # (N, morgan_bits)
            np.stack(rdkit_list),       # (N, n_descriptors)
            np.stack(img_list)          # (N, H, W, 1)
        )

    def _extract_raw_features(self):
        """Выполняет извлечение признаков один раз при инициализации."""
        train_smiles = self.raw_train_data["SMILES"].tolist()
        test_smiles = self.raw_test_data["SMILES"].tolist()

        self._morgan_train, self._desc_train, self._img_train = self.extract_raw(train_smiles)
        self._morgan_test, self._desc_test, self._img_test = self.extract_raw(test_smiles)