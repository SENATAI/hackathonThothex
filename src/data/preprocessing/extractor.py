import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

from data.preprocessing.data_loader import DataLoader


class Extractor:
    def __init__(
        self,
        morgan_bits: int = 2048,
        morgan_radius: int = 2,
        bert_model_name: str = "DeepChem/ChemBERTa-77M-MTR",
        download: bool = False,
        device: str = None
    ):
        loader = DataLoader()
        self.raw_train_data, self.raw_test_data = loader.get_data(download=download)

        # -------------------------
        # RDKit Morgan Fingerprints
        # -------------------------
        self.morgan_bits = morgan_bits
        self.morgan_radius = morgan_radius
        self.morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.morgan_radius,
            fpSize=self.morgan_bits
        )

        # -------------------------
        # RDKit descriptors
        # -------------------------
        self.descriptor_fns = [fn for _, fn in Descriptors._descList]
        self.n_descriptors = len(self.descriptor_fns)

        # -------------------------
        # ChemBERTa
        # -------------------------
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading ChemBERTa on {self.device}...")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name).to(self.device)
        self.bert_model.eval()

        # -------------------------
        # Storage
        # -------------------------
        self._morgan_train = None
        self._desc_train = None
        self._bert_train = None

        self._morgan_test = None
        self._desc_test = None
        self._bert_test = None

        # Extract everything
        self._extract_raw_features()

    # ---------------------------------------------------------------
    # RDKit FEATURES
    # ---------------------------------------------------------------
    def morgan_fp(self, mol):
        if mol is None:
            return np.zeros(self.morgan_bits, dtype=np.int8)
        fp = self.morgan_generator.GetFingerprint(mol)
        return np.array(fp.ToList(), dtype=np.int8)

    def rdkit_desc(self, mol):
        if mol is None:
            return np.zeros(self.n_descriptors, dtype=np.float32)
        return np.array([fn(mol) for fn in self.descriptor_fns], dtype=np.float32)

    # ---------------------------------------------------------------
    # BERT EMBEDDINGS
    # ---------------------------------------------------------------
    @torch.no_grad()
    def chemberta_embed(self, smiles_list, batch=32):
        all_embeds = []
        for i in tqdm(range(0, len(smiles_list), batch), desc="ChemBERTa embeddings"):
            smi_batch = smiles_list[i:i+batch]

            tokens = self.bert_tokenizer(
                smi_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.device)

            outputs = self.bert_model(**tokens)
            # Mean pooling over the token dimension
            last_hidden = outputs.last_hidden_state  # [B, L, H]
            emb = last_hidden.mean(dim=1).cpu().numpy()

            all_embeds.append(emb)

        return np.vstack(all_embeds)

    # ---------------------------------------------------------------
    # COMMON EXTRACTION
    # ---------------------------------------------------------------
    def extract_raw(self, smiles_list):
        morgan_list = []
        rdkit_list = []
        mols = []

        for smi in tqdm(smiles_list, desc="Parsing SMILES"):
            mol = None
            if isinstance(smi, str) and smi.strip():
                try:
                    mol = Chem.MolFromSmiles(smi.strip())
                except Exception:
                    mol = None
            mols.append(mol)

        for mol in tqdm(mols, desc="RDKit features"):
            morgan_list.append(self.morgan_fp(mol))
            rdkit_list.append(self.rdkit_desc(mol))

        morgan_arr = np.stack(morgan_list)
        rdkit_arr = np.stack(rdkit_list)

        return morgan_arr, rdkit_arr, mols

    # ---------------------------------------------------------------
    def _extract_raw_features(self):
        train_smiles = self.raw_train_data["SMILES"].tolist()
        test_smiles = self.raw_test_data["SMILES"].tolist()

        # RDKit + Morgan
        self._morgan_train, self._desc_train, _ = self.extract_raw(train_smiles)
        self._morgan_test, self._desc_test, _ = self.extract_raw(test_smiles)

        # ChemBERTa embeddings
        print("Extracting ChemBERTa train embeddings...")
        self._bert_train = self.chemberta_embed(train_smiles)

        print("Extracting ChemBERTa test embeddings...")
        self._bert_test = self.chemberta_embed(test_smiles)
