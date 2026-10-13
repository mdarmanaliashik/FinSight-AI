"""
FinSight AI - Configuration & Environment Module
Dynamic project path resolution, reproducibility seeds, and artifact paths.
"""
import os
import sys
from pathlib import Path

# Deterministic random seed
RANDOM_SEED = 42

def get_project_root() -> Path:
    """
    Dynamically resolves project root directory without hardcoded paths.
    Checks environment variable FINSIGHT_ROOT, then traverses upwards looking for markers.
    """
    env_root = os.environ.get("FINSIGHT_ROOT")
    if env_root and os.path.exists(env_root):
        return Path(env_root).resolve()

    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "src").exists() and (parent / "notebooks").exists():
            return parent
        if (parent / "olist").exists() or (parent / "online+retail").exists():
            return parent
            
    # Fallback to current working directory
    return Path(os.getcwd()).resolve()

PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
JD_RETAIL_DIR = PROJECT_ROOT / "jd_retail"
if not JD_RETAIL_DIR.exists():
    fallback_jd = Path(r"d:\Others\jd_retail")
    if fallback_jd.exists():
        JD_RETAIL_DIR = fallback_jd
    else:
        JD_RETAIL_DIR = PROJECT_ROOT / "data" / "raw" / "jd_retail"

OLIST_DIR = PROJECT_ROOT / "olist"
if not OLIST_DIR.exists():
    OLIST_DIR = Path(r"d:\Others\olist")

# Ensure processed directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
