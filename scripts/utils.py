"""
Shared utilities for the Li Keqiang Index data pipeline.

Provides:
- Path constants (PROJECT_ROOT, DATA_RAW, DATA_PROCESSED, DATA_REF)
- State name mapper (load_state_mapper, canonicalize_state)
- Download helper (download_file)
- Source URL constants
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_REF = PROJECT_ROOT / "data" / "reference"
PUBLIC_DATA = PROJECT_ROOT / "public" / "data"

# ---------------------------------------------------------------------------
# Source URLs
# ---------------------------------------------------------------------------

POSOCO_URL = "https://robbieandrew.github.io/india/data/POSOCO_data.csv"

GST_URL_TEMPLATE = (
    "https://tutorial.gst.gov.in/offlineutilities/gst_statistics/"
    "statewise_GST_collection_{fy}.xlsx"
)
GST_FISCAL_YEARS = [
    "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]

RBI_URLS = {
    "rbi_scb_credit_by_state.xlsx": (
        "https://rbidocs.rbi.org.in/rdocs/Publications/DOCs/"
        "156T_1112202520771561966C49F1B9C00F56ACF97557.XLSX"
    ),
    "rbi_scb_deposits_by_state.xlsx": (
        "https://rbidocs.rbi.org.in/rdocs/Publications/DOCs/"
        "155T_11122025BC88547570414295AB088FBCF5C90806.XLSX"
    ),
}

# Fiscal year month mapping: month name -> (month_number, is_next_calendar_year)
# For FY "2023-24": Apr=2023-04, ..., Dec=2023-12, Jan=2024-01, ..., Mar=2024-03
FY_MONTH_MAP = {
    "apr": (4, False), "may": (5, False), "jun": (6, False),
    "jul": (7, False), "aug": (8, False), "sep": (9, False),
    "oct": (10, False), "nov": (11, False), "dec": (12, False),
    "jan": (1, True), "feb": (2, True), "mar": (3, True),
    # Full names
    "april": (4, False), "june": (6, False), "july": (7, False),
    "august": (8, False), "september": (9, False), "october": (10, False),
    "november": (11, False), "december": (12, False), "january": (1, True),
    "february": (2, True), "march": (3, True),
}


def fy_month_to_yyyy_mm(month_label: str, fiscal_year: str) -> Optional[str]:
    """Convert a month label + fiscal year to YYYY-MM format.

    Args:
        month_label: e.g., "Apr", "January", "Nov-22"
        fiscal_year: e.g., "2023-24"

    Returns:
        e.g., "2023-04" or None if parsing fails.
    """
    # Extract start year from FY
    try:
        start_year = int(fiscal_year.split("-")[0])
    except (ValueError, IndexError):
        return None

    # Clean the month label — strip trailing year suffix like "Apr-22"
    clean = month_label.strip().split("-")[0].strip().lower()

    if clean not in FY_MONTH_MAP:
        return None

    month_num, is_next_year = FY_MONTH_MAP[clean]
    year = start_year + 1 if is_next_year else start_year
    return f"{year:04d}-{month_num:02d}"


def month_to_fiscal_year(yyyy_mm: str) -> str:
    """Convert YYYY-MM to fiscal year string.

    FY runs Apr-Mar. April 2023 → "2023-24", January 2024 → "2023-24".
    """
    parts = yyyy_mm.split("-")
    year, month = int(parts[0]), int(parts[1])
    if month >= 4:
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"


# ---------------------------------------------------------------------------
# State Name Mapper
# ---------------------------------------------------------------------------

def load_state_mapper() -> Dict[str, str]:
    """Load canonical state names and build alias -> canonical mapping.

    Returns:
        Dict mapping lowercase alias -> canonical name.
    """
    ref_path = DATA_REF / "state_metadata.csv"
    if not ref_path.exists():
        print(f"  WARNING: {ref_path} not found. State name matching disabled.")
        return {}

    df = pd.read_csv(ref_path)
    mapper: Dict[str, str] = {}
    for _, row in df.iterrows():
        canonical = str(row["canonical_name"]).strip()
        if not canonical or canonical == "nan":
            continue
        mapper[canonical.lower()] = canonical
        if pd.notna(row.get("aliases")):
            for alias in str(row["aliases"]).split(";"):
                alias = alias.strip()
                if alias:
                    mapper[alias.lower()] = canonical
    return mapper


def canonicalize_state(name: str, mapper: Dict[str, str]) -> Optional[str]:
    """Map a single state name to its canonical form.

    Args:
        name: Raw state name (any case).
        mapper: Dict from load_state_mapper().

    Returns:
        Canonical name or None if no match.
    """
    if not name or not mapper:
        return None
    key = str(name).strip().lower()
    return mapper.get(key)


def load_state_metadata() -> pd.DataFrame:
    """Load the full state metadata CSV."""
    return pd.read_csv(DATA_REF / "state_metadata.csv")


# ---------------------------------------------------------------------------
# Non-State Filters
# ---------------------------------------------------------------------------

# Rows to skip when parsing state data from various sources
NON_STATE_LABELS = {
    "total", "grand total", "all india", "all-india",
    "sub total", "cbic", "oidar", "other territory",
    "northern region", "north-eastern region", "eastern region",
    "central region", "western region", "southern region",
}


def is_state_row(value: str) -> bool:
    """Check if a row label looks like a state (not a total/region/note)."""
    clean = str(value).strip().lower()
    if clean in NON_STATE_LABELS:
        return False
    if clean.startswith("note"):
        return False
    if "the above numbers" in clean:
        return False
    if clean == "" or clean == "nan":
        return False
    return True


# ---------------------------------------------------------------------------
# Download Helper
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path, description: str) -> bool:
    """Download a file if it doesn't already exist. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        size_kb = dest.stat().st_size / 1024
        print(f"  [SKIP] {description} already exists ({size_kb:.0f} KB)")
        return True

    print(f"  [DOWNLOAD] {description}")
    print(f"             {url}")
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size_kb = dest.stat().st_size / 1024
        print(f"  [OK] Saved {dest.name} ({size_kb:.0f} KB)")
        return True
    except requests.exceptions.HTTPError as e:
        print(f"  [FAIL] HTTP {e.response.status_code}: {description}")
        return False
    except Exception as e:
        print(f"  [FAIL] {description}: {e}")
        return False


# ---------------------------------------------------------------------------
# Logging Helpers
# ---------------------------------------------------------------------------

def ensure_dirs():
    """Create all required data directories."""
    for subdir in ["gst", "electricity", "rbi", "epfo"]:
        (DATA_RAW / subdir).mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    PUBLIC_DATA.mkdir(parents=True, exist_ok=True)
    (PUBLIC_DATA / "states").mkdir(parents=True, exist_ok=True)


def print_summary(label: str, df: pd.DataFrame, key_cols: list = None):
    """Print a brief summary of a DataFrame."""
    print(f"\n  {label}")
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    if key_cols:
        for col in key_cols:
            if col in df.columns:
                nunique = df[col].nunique()
                print(f"  {col}: {nunique} unique values")
