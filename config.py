from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = PROJECT_ROOT / "dataset/raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset/processed"
MODEL_DIR = PROJECT_ROOT / "models"

WINDOW_SIZE = 100
RANDOM_STATE = 42
BEST_THRESHOLD = 0.0262