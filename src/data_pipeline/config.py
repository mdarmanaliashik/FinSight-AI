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

def _resolve_data_dir(candidates):
    for c in candidates:
        if c and c.exists():
            return c
    return candidates[0]

# Multi-Environment Candidates for JD Retail
JD_RETAIL_DIR = _resolve_data_dir([
    PROJECT_ROOT / "jd_retail",
    PROJECT_ROOT / "data" / "raw" / "jd_retail",
    PROJECT_ROOT / "data" / "jd_retail",
    Path("/kaggle/input/jd-retail"),
    Path("/kaggle/input/jd-order-data"),
    Path("/content/jd_retail"),
    Path(r"d:\Others\jd_retail")
])

# Multi-Environment Candidates for Olist E-Commerce
OLIST_DIR = _resolve_data_dir([
    PROJECT_ROOT / "olist",
    PROJECT_ROOT / "data" / "raw" / "olist",
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "olist",
    Path("/kaggle/input/brazilian-ecommerce"),
    Path("/kaggle/input/olist"),
    Path("/content/olist"),
    Path("/content/data/raw"),
    Path(r"d:\Others\olist")
])

# Ensure processed directory exists
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

