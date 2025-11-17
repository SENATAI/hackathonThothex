import os
from pathlib import Path

def find_project_root():
    """
    Ищет корень проекта по наличию .git.
    """
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / '.git').is_dir():
            return parent
    return current_path.parent

PROJECT_ROOT = find_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

TRAIN_DATA_PATH = os.path.join(RAW_DATA_DIR, 'train.csv')
TEST_DATA_PATH = os.path.join(RAW_DATA_DIR, 'test.csv')