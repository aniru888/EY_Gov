"""
Phase 0: Data Source Validation Script
Li Keqiang Index for Indian States

Downloads available data sources, loads each into pandas,
validates structure, and produces a validation report.

Usage:
    python scripts/00_validate_sources.py
    python scripts/00_validate_sources.py --download
    python scripts/00_validate_sources.py --download --report-file validation_report.md
"""

import argparse
import datetime
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_REF = PROJECT_ROOT / "data" / "reference"

POSOCO_URL = "https://robbieandrew.github.io/india/data/POSOCO_data.csv"

GST_URL_TEMPLATE = (
    "https://tutorial.gst.gov.in/offlineutilities/gst_statistics/"
    "statewise_GST_collection_{fy}.xlsx"
)
GST_FISCAL_YEARS = [
    "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]

# RBI Handbook of Statistics on Indian States 2024-25 — direct Excel downloads
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

# ---------------------------------------------------------------------------
# State Name Mapper
# ---------------------------------------------------------------------------

def load_state_mapper() -> Dict[str, str]:
    """Load canonical state names and build alias -> canonical mapping."""
    ref_path = DATA_REF / "state_metadata.csv"
    if not ref_path.exists():
        print(f"  WARNING: {ref_path} not found. State name matching disabled.")
        return {}

    df = pd.read_csv(ref_path)
    mapper: Dict[str, str] = {}
    for _, row in df.iterrows():
        canonical = str(row["canonical_name"]).strip()
        mapper[canonical.lower()] = canonical
        if pd.notna(row.get("aliases")):
            for alias in str(row["aliases"]).split(";"):
                alias = alias.strip()
                if alias:
                    mapper[alias.lower()] = canonical
    return mapper


def match_states(found_states: List[str], mapper: Dict[str, str]) -> Dict[str, Any]:
    """Match found state names against canonical list."""
    matched = []
    unmatched = []
    for s in found_states:
        key = str(s).strip().lower()
        if key in mapper:
            matched.append((s, mapper[key]))
        else:
            unmatched.append(s)

    canonical_all = set(mapper.values())
    matched_canonical = set(m[1] for m in matched)
    missing = sorted(canonical_all - matched_canonical)

    return {
        "matched_count": len(matched_canonical),
        "unmatched_names": unmatched,
        "missing_states": missing,
    }

# ---------------------------------------------------------------------------
# Download Helpers
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


def download_automated_sources():
    """Download sources that can be fetched programmatically."""
    print("\n" + "=" * 60)
    print(" DOWNLOADING AUTOMATED SOURCES")
    print("=" * 60)

    # Electricity (POSOCO)
    print("\n--- Electricity (POSOCO) ---")
    download_file(
        POSOCO_URL,
        DATA_RAW / "electricity" / "POSOCO_data.csv",
        "POSOCO daily electricity demand CSV",
    )

    # GST Excels from gst.gov.in
    print("\n--- GST State Collections ---")
    success = 0
    for fy in GST_FISCAL_YEARS:
        url = GST_URL_TEMPLATE.format(fy=fy)
        dest = DATA_RAW / "gst" / f"statewise_GST_collection_{fy}.xlsx"
        if download_file(url, dest, f"GST FY {fy}"):
            success += 1
    print(f"\n  GST download summary: {success}/{len(GST_FISCAL_YEARS)} files")

    # RBI Handbook tables
    print("\n--- RBI Bank Credit/Deposits ---")
    for filename, url in RBI_URLS.items():
        download_file(url, DATA_RAW / "rbi" / filename, f"RBI {filename}")

# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------

def validate_gst(mapper: Dict[str, str]) -> Dict[str, Any]:
    """Validate GST state collection Excel files."""
    result: Dict[str, Any] = {
        "source": "GST State Collections",
        "status": "NO-GO",
        "files": [],
        "issues": [],
        "details": {},
    }

    gst_dir = DATA_RAW / "gst"
    xlsx_files = sorted(gst_dir.glob("*.xlsx")) if gst_dir.exists() else []
    if not xlsx_files:
        result["issues"].append("No .xlsx files found in data/raw/gst/")
        result["action"] = (
            "Download from https://www.gst.gov.in/download/gststatistics\n"
            "Or run: python scripts/00_validate_sources.py --download"
        )
        return result

    result["files"] = [f.name for f in xlsx_files]
    all_states: set = set()
    total_rows = 0
    year_range: list = []
    files_parsed = 0

    for fpath in xlsx_files:
        try:
            xls = pd.ExcelFile(fpath, engine="openpyxl")

            # 2020-21+ files have 3 sheets; state data is in 'Collections-Statewise'
            # Earlier files have a single sheet with state data
            if "Collections-Statewise" in xls.sheet_names:
                sheet = "Collections-Statewise"
            else:
                sheet = 0

            df_raw = pd.read_excel(fpath, sheet_name=sheet, header=None, engine="openpyxl")

            # Find header row containing "State Cd" or "State" keyword
            header_row = None
            for i in range(min(10, len(df_raw))):
                row_vals = [str(v).strip().lower() for v in df_raw.iloc[i] if pd.notna(v)]
                if any(v in ("state cd", "state") for v in row_vals):
                    header_row = i
                    break
                if any("state cd" in v or v == "state" for v in row_vals):
                    header_row = i
                    break

            if header_row is None:
                result["issues"].append(f"{fpath.name}: Could not find header row")
                continue

            df = pd.read_excel(fpath, sheet_name=sheet, header=header_row, engine="openpyxl")

            # Find the state NAME column (prefer "State" over "State Cd")
            state_col = None
            for col in df.columns:
                col_str = str(col).strip().lower()
                if col_str == "state":
                    state_col = col
                    break
            # Fallback: any column containing "state" but not "state cd"
            if state_col is None:
                for col in df.columns:
                    col_str = str(col).strip().lower()
                    if "state" in col_str and "cd" not in col_str:
                        state_col = col
                        break

            if state_col is None:
                result["issues"].append(f"{fpath.name}: No 'State' column found in headers: {list(df.columns[:5])}")
                continue

            states = [s for s in df[state_col].dropna().unique()
                      if str(s).strip().lower() not in ("total", "grand total", "all india", "")]
            all_states.update(str(s).strip() for s in states)
            total_rows += len(df)
            files_parsed += 1

            # Extract fiscal year from filename
            fy = fpath.stem.split("_")[-1]
            year_range.append(fy)

            # Count data columns (everything except State Cd, State, totals)
            skip_cols = {str(state_col).lower(), "state cd", "total", "grand total"}
            month_cols = [c for c in df.columns
                          if str(c).strip().lower() not in skip_cols]
            result["details"][fpath.name] = {
                "rows": len(df),
                "states": len(states),
                "data_columns": len(month_cols),
                "sheet": sheet if isinstance(sheet, str) else "Sheet1",
            }

        except Exception as e:
            result["issues"].append(f"{fpath.name}: {e}")

    # State matching
    if all_states:
        state_match = match_states(list(all_states), mapper)
        result["state_match"] = state_match
        result["states_found"] = state_match["matched_count"]
    else:
        result["states_found"] = 0

    result["total_rows"] = total_rows
    result["files_parsed"] = files_parsed
    result["time_range"] = f"FY {min(year_range)} to FY {max(year_range)}" if year_range else "Unknown"

    # GO/NO-GO
    if files_parsed >= 5 and result.get("states_found", 0) >= 25:
        result["status"] = "GO"
    elif files_parsed >= 3:
        result["status"] = "CONDITIONAL GO"

    return result


def validate_electricity(mapper: Dict[str, str]) -> Dict[str, Any]:
    """Validate POSOCO daily electricity demand CSV."""
    result: Dict[str, Any] = {
        "source": "Electricity Consumption (POSOCO)",
        "status": "NO-GO",
        "files": [],
        "issues": [],
    }

    csv_path = DATA_RAW / "electricity" / "POSOCO_data.csv"
    if not csv_path.exists():
        result["issues"].append("File not found: data/raw/electricity/POSOCO_data.csv")
        result["action"] = (
            "Download from https://robbieandrew.github.io/india/data/POSOCO_data.csv\n"
            "Or run: python scripts/00_validate_sources.py --download"
        )
        return result

    result["files"] = [csv_path.name]
    size_mb = csv_path.stat().st_size / (1024 * 1024)

    try:
        df = pd.read_csv(csv_path, low_memory=False)
        result["total_rows"] = len(df)
        result["total_columns"] = len(df.columns)
        result["file_size_mb"] = round(size_mb, 1)

        # Parse date column (first column, expected format YYYYMMDD)
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], format="%Y%m%d", errors="coerce")
        valid_dates = df[date_col].dropna()

        if len(valid_dates) > 0:
            result["time_range"] = f"{valid_dates.min().date()} to {valid_dates.max().date()}"
            result["date_parse_failures"] = int(df[date_col].isna().sum())
        else:
            result["issues"].append("Could not parse any dates from first column")

        # Identify state-level columns
        # POSOCO format: "State: EnergyMet" (colon-space separator)
        all_cols = list(df.columns[1:])  # skip date column
        energy_cols = [c for c in all_cols if "EnergyMet" in str(c)]
        # Region-level columns (NR, WR, SR, ER, NER, India)
        region_prefixes = ("NR:", "WR:", "SR:", "ER:", "NER:", "India:")
        # Also exclude non-state entities (DVC, Essar, AMNSIL, BALCO, Railways, RIL)
        non_state = ("DVC:", "Essar", "AMNSIL:", "BALCO:", "Railways", "RIL ")
        region_cols = [c for c in energy_cols
                       if any(c.startswith(p) for p in region_prefixes)
                       or any(c.startswith(p) for p in non_state)]
        state_cols = [c for c in energy_cols if c not in region_cols]

        result["state_columns_count"] = len(state_cols)
        result["region_columns_count"] = len(region_cols)
        result["sample_columns"] = state_cols[:10]

        # Extract state names from column names ("Punjab: EnergyMet" -> "Punjab")
        state_names = []
        for c in state_cols:
            name = c.replace(": EnergyMet", "").strip()
            state_names.append(name)

        if mapper and state_names:
            state_match = match_states(state_names, mapper)
            result["state_match"] = state_match
            result["states_found"] = state_match["matched_count"]
        else:
            result["states_found"] = len(state_cols)

        # Data density
        if state_cols:
            missing_pct = float(df[state_cols].isna().mean().mean() * 100)
            result["overall_missing_pct"] = round(missing_pct, 1)

        # GO/NO-GO
        if len(df) > 3000 and len(state_cols) >= 15:
            result["status"] = "GO"
        elif len(df) > 1000:
            result["status"] = "CONDITIONAL GO"

    except Exception as e:
        result["issues"].append(f"Failed to load: {e}")

    return result


def validate_rbi(mapper: Dict[str, str]) -> Dict[str, Any]:
    """Validate RBI bank credit data."""
    result: Dict[str, Any] = {
        "source": "RBI Bank Credit",
        "status": "NO-GO",
        "files": [],
        "issues": [],
    }

    rbi_dir = DATA_RAW / "rbi"
    if not rbi_dir.exists():
        rbi_dir.mkdir(parents=True, exist_ok=True)

    files = (list(rbi_dir.glob("*.xlsx")) + list(rbi_dir.glob("*.xls")) +
             list(rbi_dir.glob("*.csv")) + list(rbi_dir.glob("*.parquet")))

    if not files:
        result["issues"].append("No data files found in data/raw/rbi/")
        result["action"] = (
            "Manual download required:\n"
            "  Option A: https://dbie.rbi.org.in\n"
            "            -> Publications -> Handbook of Statistics on Indian States\n"
            "            -> Banking -> Credit by Scheduled Commercial Banks\n"
            "            Download Excel, save to data/raw/rbi/\n"
            "  Option B: https://dataful.in/collections/959/\n"
            "            Register and download RBI credit dataset\n"
            "            Save to data/raw/rbi/"
        )
        return result

    result["files"] = [f.name for f in files]

    for fpath in files:
        try:
            if fpath.suffix in (".xlsx", ".xls"):
                engine = "openpyxl" if fpath.suffix == ".xlsx" else "xlrd"
                try:
                    df_raw = pd.read_excel(fpath, header=None, engine=engine)
                except Exception:
                    df_raw = pd.read_excel(fpath, header=None)

                result["shape"] = list(df_raw.shape)

                # Look for header row containing "State" or known state names
                header_row = None
                for i in range(min(20, len(df_raw))):
                    row_vals = [str(v).strip().lower() for v in df_raw.iloc[i] if pd.notna(v)]
                    if any("state" in v or "andhra" in v or "maharashtra" in v for v in row_vals):
                        header_row = i
                        break

                if header_row is not None:
                    result["probable_header_row"] = header_row
                    df = pd.read_excel(fpath, header=header_row, engine=engine if engine else None)
                    result["columns_found"] = list(df.columns[:10])
                    result["row_count"] = len(df)

                    # Try to find state column
                    state_col = None
                    for col in df.columns:
                        if "state" in str(col).lower():
                            state_col = col
                            break

                    if state_col:
                        states = [s for s in df[state_col].dropna().unique()
                                  if str(s).strip().lower() not in ("total", "grand total", "all india", "")]
                        state_match = match_states([str(s).strip() for s in states], mapper)
                        result["state_match"] = state_match
                        result["states_found"] = state_match["matched_count"]
                else:
                    result["issues"].append(
                        f"{fpath.name}: Could not auto-detect header row. "
                        f"First 5 rows:\n{df_raw.head(5).to_string()}"
                    )

                result["status"] = "CONDITIONAL GO"

            elif fpath.suffix == ".csv":
                df = pd.read_csv(fpath)
                result["shape"] = list(df.shape)
                result["columns_found"] = list(df.columns[:10])
                result["status"] = "CONDITIONAL GO"

            elif fpath.suffix == ".parquet":
                df = pd.read_parquet(fpath)
                result["shape"] = list(df.shape)
                result["columns_found"] = list(df.columns[:10])
                result["status"] = "CONDITIONAL GO"

        except Exception as e:
            result["issues"].append(f"{fpath.name}: {e}")

    return result


def validate_epfo(mapper: Dict[str, str]) -> Dict[str, Any]:
    """Validate EPFO payroll data (replacement for Udyam MSME).

    EPFO monthly payroll release Excels have a 'Statewise' sheet with:
    - 6 age-bucket sections, each listing ~32 states
    - Annual columns for past FYs (2017-18 onward)
    - Monthly columns for the current FY
    We sum across all age buckets to get total net new payroll per state.
    """
    result: Dict[str, Any] = {
        "source": "EPFO Net Payroll (Formal Employment)",
        "status": "NO-GO",
        "files": [],
        "issues": [],
    }

    epfo_dir = DATA_RAW / "epfo"
    # Also check legacy udyam dir for the sample file
    udyam_dir = DATA_RAW / "udyam"
    epfo_dir.mkdir(parents=True, exist_ok=True)

    files = list(epfo_dir.glob("epfo*.xlsx"))
    # Fallback: check udyam dir for sample
    if not files:
        files = [f for f in udyam_dir.glob("epfo*.xlsx") if f.stat().st_size > 1000]

    if not files:
        result["issues"].append("No EPFO Excel files found in data/raw/epfo/ or data/raw/udyam/")
        result["action"] = (
            "Manual download required (EPFO blocks automated requests):\n"
            "  1. Visit https://www.epfindia.gov.in/site_en/Estimate_of_Payroll.php\n"
            "  2. Download the latest 'Payroll Data' Excel\n"
            "  3. Save to data/raw/epfo/\n"
            "  URL pattern: https://www.epfindia.gov.in/site_docs/exmpted_est/"
            "Payroll_Data_EPFO_{Month}_{Year}.xlsx"
        )
        return result

    result["files"] = [f.name for f in files]

    for fpath in files:
        try:
            xls = pd.ExcelFile(fpath, engine="openpyxl")

            if "Statewise" not in xls.sheet_names:
                result["issues"].append(
                    f"{fpath.name}: No 'Statewise' sheet. "
                    f"Sheets found: {xls.sheet_names}"
                )
                continue

            df = pd.read_excel(fpath, sheet_name="Statewise", header=None, engine="openpyxl")
            result["shape"] = list(df.shape)

            # Parse column headers from row 3
            headers = df.iloc[3].tolist()
            result["columns_found"] = [str(h)[:20] for h in headers[:10]]

            # Find all age-bucket section boundaries
            # Sections start with "Age Bucket:" in column 0
            section_starts = []
            for r in range(df.shape[0]):
                val = str(df.iloc[r, 0]).strip()
                if val.startswith("Age Bucket:"):
                    section_starts.append(r)

            result["age_bucket_sections"] = len(section_starts)

            if not section_starts:
                result["issues"].append(f"{fpath.name}: No 'Age Bucket:' sections found")
                continue

            # Each section: header row, then data header row (+1), then state rows (+2 to SUB TOTAL)
            # Get state list from first section (data starts 2 rows after section header)
            first_data_row = section_starts[0] + 2  # skip "Age Bucket:" and column headers
            states = []
            for r in range(first_data_row, df.shape[0]):
                val = str(df.iloc[r, 0]).strip()
                if val.startswith("SUB TOTAL") or val.startswith("Grand Total"):
                    break
                if val and val != "nan":
                    states.append(val)

            n_states = len(states)
            result["states_in_data"] = n_states

            # Sum across all age buckets for total net payroll
            n_data_cols = df.shape[1] - 1  # exclude state name column
            totals = np.zeros((n_states, n_data_cols))

            for sec_start in section_starts:
                data_start = sec_start + 2
                for i in range(n_states):
                    row_idx = data_start + i
                    if row_idx >= df.shape[0]:
                        break
                    for j in range(n_data_cols):
                        try:
                            totals[i, j] += float(df.iloc[row_idx, j + 1])
                        except (ValueError, TypeError):
                            pass

            # Identify annual vs monthly columns from headers
            annual_cols = []
            monthly_cols = []
            for j, h in enumerate(headers[1:], start=0):
                h_str = str(h).strip()
                if "-" in h_str and len(h_str) <= 10 and not h_str.startswith("20"):
                    # FY format like "2018-19"
                    annual_cols.append((j, h_str))
                elif "from" in h_str.lower():
                    # "2017-18 From Sep-17"
                    annual_cols.append((j, h_str))
                elif hasattr(h, 'strftime'):
                    # datetime object = monthly column
                    monthly_cols.append((j, h.strftime("%Y-%m")))
                else:
                    annual_cols.append((j, h_str))

            result["annual_periods"] = len(annual_cols)
            result["monthly_periods"] = len(monthly_cols)
            result["time_range"] = (
                f"{annual_cols[0][1]} to {annual_cols[-1][1]}"
                if annual_cols else "Unknown"
            )
            if monthly_cols:
                result["current_fy_months"] = [m[1] for m in monthly_cols]

            # Match states
            clean_states = [s.strip().title() for s in states]
            state_match = match_states(clean_states, mapper)
            result["state_match"] = state_match
            result["states_found"] = state_match["matched_count"]

            # Summary stats for latest annual period
            if annual_cols:
                last_annual_idx = annual_cols[-1][0]
                last_annual_totals = totals[:, last_annual_idx]
                result["latest_annual_total"] = int(np.sum(last_annual_totals))
                result["latest_annual_period"] = annual_cols[-1][1]

            result["has_time_series"] = True

            # GO/NO-GO
            if n_states >= 25 and len(annual_cols) >= 5:
                result["status"] = "GO"
            elif n_states >= 20 and len(annual_cols) >= 3:
                result["status"] = "CONDITIONAL GO"

        except Exception as e:
            result["issues"].append(f"{fpath.name}: {e}")

    return result

# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

def generate_report(results: List[Dict[str, Any]], output_file: Optional[Path] = None):
    """Print and optionally save the validation report."""
    lines: list = []
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("=" * 60)
    lines.append(" DATA VALIDATION REPORT")
    lines.append(" Li Keqiang Index for Indian States - Phase 0")
    lines.append(f" Generated: {timestamp}")
    lines.append("=" * 60)

    go_count = 0
    conditional_count = 0
    nogo_count = 0

    for r in results:
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"SOURCE: {r['source']}")
        lines.append(f"  Status:      {r['status']}")
        lines.append(f"  Files:       {', '.join(r.get('files', ['None']))}")

        if "total_rows" in r:
            lines.append(f"  Rows:        {r['total_rows']:,}")
        if "shape" in r:
            lines.append(f"  Shape:       {r['shape']}")
        if "total_columns" in r:
            lines.append(f"  Columns:     {r['total_columns']}")
        if "states_found" in r:
            lines.append(f"  States:      {r['states_found']} matched to canonical list")
        if "time_range" in r:
            lines.append(f"  Time Range:  {r['time_range']}")
        if "file_size_mb" in r:
            lines.append(f"  File Size:   {r['file_size_mb']} MB")
        if "overall_missing_pct" in r:
            lines.append(f"  Missing:     {r['overall_missing_pct']}%")
        if "files_parsed" in r:
            lines.append(f"  Parsed:      {r['files_parsed']}/{len(r.get('files', []))} files")
        if "has_time_series" in r:
            lines.append(f"  Time Series: {'Yes' if r['has_time_series'] else 'No (cumulative only)'}")

        sm = r.get("state_match")
        if sm:
            if sm["unmatched_names"]:
                lines.append(f"  Unmatched:   {sm['unmatched_names'][:10]}")
            if sm["missing_states"]:
                missing_short = sm["missing_states"][:8]
                lines.append(f"  Not Found:   {missing_short}" +
                             (f" (+{len(sm['missing_states']) - 8} more)" if len(sm['missing_states']) > 8 else ""))

        if r.get("sample_columns"):
            lines.append(f"  Sample Cols: {r['sample_columns'][:5]}")

        if r.get("issues"):
            lines.append("  Issues:")
            for issue in r["issues"]:
                for line in issue.split("\n"):
                    lines.append(f"    - {line}")

        if r.get("action"):
            lines.append("  Action Required:")
            for line in r["action"].split("\n"):
                lines.append(f"    {line}")

        status = r["status"]
        if status == "GO":
            go_count += 1
        elif status == "CONDITIONAL GO":
            conditional_count += 1
        else:
            nogo_count += 1

    lines.append("")
    lines.append("=" * 60)
    lines.append(f" SUMMARY: {go_count} GO | {conditional_count} CONDITIONAL | {nogo_count} NO-GO")
    lines.append("")

    if go_count + conditional_count == 4:
        lines.append(" GATE: PASSED (review any CONDITIONAL items)")
        lines.append(" Ready to proceed to Phase 1: Data Pipeline")
    elif go_count + conditional_count >= 2:
        lines.append(" GATE: PARTIAL - Some sources need attention")
        lines.append(" Download missing sources and re-run this script")
    else:
        lines.append(" GATE: NOT MET - Resolve issues before Phase 1")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    # Handle Windows console encoding issues (e.g., ₹ symbol from RBI data)
    try:
        print(report_text)
    except UnicodeEncodeError:
        print(report_text.encode("ascii", errors="replace").decode("ascii"))

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report_text, encoding="utf-8")
        print(f"\nReport saved to: {output_file}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 0: Data Source Validation for Li Keqiang Index"
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download automated sources (POSOCO, GST) before validating"
    )
    parser.add_argument(
        "--report-file", type=str, default=None,
        help="Save report to file (e.g., validation_report.md)"
    )
    args = parser.parse_args()

    print("\n Li Keqiang Index - Phase 0: Data Validation Sprint")
    print(f" Project root: {PROJECT_ROOT}")
    print(f" Data directory: {DATA_RAW}\n")

    # Ensure directories exist
    for subdir in ["gst", "electricity", "rbi", "epfo", "udyam"]:
        (DATA_RAW / subdir).mkdir(parents=True, exist_ok=True)

    # Load state name mapper
    mapper = load_state_mapper()
    if mapper:
        canonical_count = len(set(mapper.values()))
        print(f" Loaded {canonical_count} canonical states with {len(mapper)} aliases")
    else:
        print(" WARNING: No state mapper loaded")

    # Download if requested
    if args.download:
        download_automated_sources()

    # Validate all sources
    print("\n" + "=" * 60)
    print(" VALIDATING DATA SOURCES")
    print("=" * 60)

    results = []

    print("\n[1/4] Validating GST...")
    results.append(validate_gst(mapper))

    print("[2/4] Validating Electricity...")
    results.append(validate_electricity(mapper))

    print("[3/4] Validating RBI Bank Credit...")
    results.append(validate_rbi(mapper))

    print("[4/4] Validating EPFO Net Payroll...")
    results.append(validate_epfo(mapper))

    # Generate report
    print()
    output_file = Path(args.report_file) if args.report_file else None
    generate_report(results, output_file)


if __name__ == "__main__":
    main()
