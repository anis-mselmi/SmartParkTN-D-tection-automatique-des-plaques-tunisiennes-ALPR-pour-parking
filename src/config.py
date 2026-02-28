from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
DB_PATH = DATA_DIR / "smartpark.db"
TUNISIA_DATA_DIR = DATA_DIR / "tunisia"
TN_VEHICLE_REGISTRY_PATH = TUNISIA_DATA_DIR / "vehicle_registry_tn.csv"
TN_POLICY_PATH = TUNISIA_DATA_DIR / "parking_policy_tn.json"
TN_ALPR_ANNOTATIONS_PATH = TUNISIA_DATA_DIR / "alpr_tn_annotations.csv"

DEFAULT_HOURLY_RATE = 2.5
DEFAULT_MAX_FREE_MINUTES = 10

TUNISIAN_PLATE_REGEX = r"^(?:\d{1,3})\s?(?:تونس|TU|TN)?\s?(?:\d{1,4})$"
