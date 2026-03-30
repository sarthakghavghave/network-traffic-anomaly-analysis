from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = PROJECT_ROOT / "dataset/raw"
PROCESSED_DIR = PROJECT_ROOT / "dataset/processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIG_DIR = PROJECT_ROOT / "figures"

WINDOW_SIZE = 100
RANDOM_STATE = 42
BEST_THRESHOLD = 0.0262

TOP_FEATURES = [
    'state_INT_mean',
    'proto_nunique',
    'sttl_mean',
    'sbytes_mean',
    'dbytes_mean',
    'service_dns_mean'
    'rate_mean',
]