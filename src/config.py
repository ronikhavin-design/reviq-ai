from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
REPORTS_DIR = DATA_DIR / "reports"
MODELS_DIR = ROOT / "models"
CONFIGS_DIR = ROOT / "configs"

MODELS_DIR.mkdir(exist_ok=True)


def load_config(path: Path = CONFIGS_DIR / "model_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


CONFIG = load_config()
