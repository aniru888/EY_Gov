"""
08_fetch_gsdp.py
================
Download and parse RBI Handbook Tables 21 & 22 for GSDP data.

Input:  Downloaded Excel files from RBI Handbook of Statistics on Indian States
Output: data/processed/gsdp_clean.parquet

RBI Excel Structure (same format as Table 156 in 03_fetch_rbi.py):
- Header row at index 4 (0-indexed)
- Year columns represent fiscal year ending (e.g., 2024 = FY 2023-24)
- Values are in Rs lakh (Table 21: current prices, Table 22: constant 2011-12 prices)
- Contains region aggregates (to be filtered out) and state-level data

Output Schema:
- state: canonical state name
- fiscal_year: e.g., "2023-24"
- gsdp_current_lakh: GSDP at current prices (Rs lakh)
- gsdp_constant_lakh: GSDP at constant 2011-12 prices (Rs lakh)
- gsdp_current_crore: GSDP at current prices (Rs crore, = lakh/100)
- gsdp_constant_crore: GSDP at constant prices (Rs crore)
- gsdp_growth_pct: YoY growth rate from constant-price series
- gsdp_rank: rank within each FY by gsdp_current_crore (1=highest)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io

from utils import (
    PROJECT_ROOT,
    DATA_RAW,
    DATA_PROCESSED,
    load_state_mapper,
    canonicalize_state,
    is_state_row,
    download_file,
)

# Constants
RBI_RAW_DIR = DATA_RAW / "rbi"
OUTPUT_PARQUET = DATA_PROCESSED / "gsdp_clean.parquet"

GSDP_URLS = {
    "rbi_gsdp_current_prices.xlsx": (
        "https://rbidocs.rbi.org.in/rdocs/Publications/DOCs/"
        "21T_11122025D994949B48C44B68B4465FBB9ADDFF3D.XLSX"
    ),
    "rbi_gsdp_constant_prices.xlsx": (
        "https://rbidocs.rbi.org.in/rdocs/Publications/DOCs/"
        "22T_11122025E6BC0CB35180406EAB6E0D49DE51C8E8.XLSX"
    ),
}

# Region headers and metadata rows to filter out (same as 03_fetch_rbi.py)
SKIP_PATTERNS = [
    "NORTHERN REGION",
    "NORTH-EASTERN REGION",
    "EASTERN REGION",
    "CENTRAL REGION",
    "WESTERN REGION",
    "SOUTHERN REGION",
    "ALL-INDIA",
    "ALL INDIA",
    "(Rs. Lakh)",
    "(RS. LAKH)",
    "TABLE 21",
    "TABLE 22",
    "GROSS STATE DOMESTIC",
    "AT CURRENT",
    "AT CONSTANT",
    "(BASE YEAR",
    "BASE YEAR",
]


def rbi_year_to_fiscal_year(rbi_year: int) -> str:
    """Convert RBI year to fiscal year format.
    RBI year "2024" means end-March 2024 = FY 2023-24.
    """
    fy_start = rbi_year - 1
    fy_end = rbi_year % 100
    return f"{fy_start}-{fy_end:02d}"


def clean_state_name(state_name: str) -> str:
    """Clean state name by removing footnote markers and extra whitespace."""
    if pd.isna(state_name):
        return ""
    cleaned = str(state_name).strip()
    cleaned = cleaned.rstrip("*+#@$%^&()0123456789")
    return cleaned.strip()


def should_skip_row(state_name: str) -> bool:
    """Determine if a row should be skipped."""
    if not state_name or pd.isna(state_name):
        return True
    state_upper = str(state_name).upper().strip()
    if len(state_upper) < 2:
        return True
    for pattern in SKIP_PATTERNS:
        if pattern in state_upper:
            return True
    return False


def is_fy_column(col) -> bool:
    """Check if a column name looks like a fiscal year (e.g., '2011-12', '2023-24')."""
    s = str(col).strip()
    if len(s) == 7 and s[4] == "-":
        try:
            int(s[:4])
            int(s[5:])
            return True
        except ValueError:
            pass
    return False


def parse_gsdp_sheet(excel_file: Path, sheet_name: str = None, header_row: int = 4) -> pd.DataFrame:
    """Parse a GSDP sheet into long format.

    Handles both sheets (i) and (ii), concatenates results.
    Returns DataFrame with columns: state, fiscal_year, gsdp_value
    """
    print(f"  Parsing file: {excel_file.name}")

    xl = pd.ExcelFile(excel_file)
    sheets = xl.sheet_names
    print(f"    Available sheets: {sheets}")

    all_frames = []

    for use_sheet in sheets:
        print(f"\n    --- Sheet: {use_sheet} ---")

        # Read raw first to inspect
        df_raw = pd.read_excel(excel_file, sheet_name=use_sheet, header=None)
        print(f"    Raw shape: {df_raw.shape}")

        # Find the header row with FY columns (e.g., "2011-12", "2012-13")
        found_header = None
        for r in range(min(10, len(df_raw))):
            row_vals = [str(v).strip() for v in df_raw.iloc[r] if pd.notna(v)]
            fy_count = sum(1 for v in row_vals if is_fy_column(v))
            if fy_count >= 3:
                found_header = r
                print(f"    Found FY header at row {r} ({fy_count} FY columns)")
                break

        if found_header is None:
            print(f"    WARNING: No FY header row found in sheet {use_sheet}, skipping")
            continue

        # Re-read with correct header
        df = pd.read_excel(excel_file, sheet_name=use_sheet, header=found_header)

        # Drop truly empty columns (all NaN)
        empty_cols = [c for c in df.columns if df[c].isna().all()]
        if empty_cols:
            df = df.drop(columns=empty_cols)

        # Find the state column: it should contain actual state name strings
        state_col = None
        for col in df.columns:
            col_str = str(col)
            if any(k in col_str for k in ["Region", "State", "Union", "state", "STATE"]):
                state_col = col
                break

        if state_col is None:
            # Find first column that contains string values (state names)
            for col in df.columns:
                if is_fy_column(col):
                    continue
                sample = df[col].dropna().head(5)
                if len(sample) > 0 and all(isinstance(v, str) for v in sample.values):
                    state_col = col
                    break

        if state_col is None:
            # Last resort: first non-FY column
            for col in df.columns:
                if not is_fy_column(col):
                    state_col = col
                    break
        if state_col is None:
            state_col = df.columns[0]

        df = df.rename(columns={state_col: "state"})
        df["state"] = df["state"].apply(clean_state_name)
        df = df[~df["state"].apply(should_skip_row)].copy()

        # Collect FY columns
        fy_cols = [col for col in df.columns if col != "state" and is_fy_column(col)]
        # Also check for "Base: 2011-12" style or integer year cols
        for col in df.columns:
            if col == "state" or col in fy_cols:
                continue
            try:
                yr = int(float(str(col)))
                if 1990 < yr < 2030:
                    fy_cols.append(col)
            except (ValueError, TypeError):
                continue

        print(f"    FY/year columns: {len(fy_cols)}")
        if not fy_cols:
            print(f"    WARNING: No FY columns in sheet {use_sheet}, skipping")
            continue

        # Melt to long format
        df_long = df[["state"] + fy_cols].melt(
            id_vars=["state"],
            value_vars=fy_cols,
            var_name="fy_raw",
            value_name="gsdp_value"
        )

        # Convert column names to standard FY format
        def normalize_fy(raw):
            s = str(raw).strip()
            # Already FY format: "2011-12"
            if is_fy_column(s):
                return s
            # Integer year: convert using rbi_year_to_fiscal_year
            try:
                yr = int(float(s))
                if 1990 < yr < 2030:
                    return rbi_year_to_fiscal_year(yr)
            except (ValueError, TypeError):
                pass
            return None

        df_long["fiscal_year"] = df_long["fy_raw"].apply(normalize_fy)
        df_long = df_long.dropna(subset=["fiscal_year"])
        df_long = df_long.drop(columns=["fy_raw"])
        df_long["gsdp_value"] = pd.to_numeric(df_long["gsdp_value"], errors="coerce")
        df_long = df_long.dropna(subset=["gsdp_value"])

        print(f"    Parsed: {len(df_long)} rows, {df_long['state'].nunique()} states")
        all_frames.append(df_long)

    if not all_frames:
        raise ValueError(f"No valid data found in any sheet of {excel_file.name}")

    combined = pd.concat(all_frames, ignore_index=True)
    # De-duplicate: prefer later sheet (more recent data)
    combined = combined.drop_duplicates(subset=["state", "fiscal_year"], keep="last")
    print(f"  Combined: {len(combined)} rows, {combined['state'].nunique()} states")
    return combined


def main():
    """Main execution function."""
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 80)
    print("RBI GSDP Data Processing")
    print("=" * 80)

    # Ensure directory exists
    RBI_RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Download GSDP Excel files
    print("\nDownloading GSDP Excel files...")
    for filename, url in GSDP_URLS.items():
        dest = RBI_RAW_DIR / filename
        success = download_file(url, dest, f"RBI GSDP: {filename}")
        if not success:
            print(f"ERROR: Failed to download {filename}")
            print("GSDP data is required for insights. Pipeline cannot continue.")
            sys.exit(1)

    # Load state mapper
    print("\nLoading state mapper...")
    state_mapper = load_state_mapper()
    print(f"  Loaded {len(state_mapper)} state mappings")

    # Parse Table 21: GSDP at current prices
    print("\nParsing Table 21: GSDP at Current Prices...")
    current_file = RBI_RAW_DIR / "rbi_gsdp_current_prices.xlsx"
    df_current = parse_gsdp_sheet(current_file)
    df_current = df_current.rename(columns={"gsdp_value": "gsdp_current_lakh"})

    # Parse Table 22: GSDP at constant prices
    print("\nParsing Table 22: GSDP at Constant Prices...")
    constant_file = RBI_RAW_DIR / "rbi_gsdp_constant_prices.xlsx"
    df_constant = parse_gsdp_sheet(constant_file)
    df_constant = df_constant.rename(columns={"gsdp_value": "gsdp_constant_lakh"})

    # Canonicalize state names in both
    print("\nCanonicalizing state names...")
    unmatched_states = set()

    for df_label, df in [("current", df_current), ("constant", df_constant)]:
        df["state_canonical"] = df["state"].apply(
            lambda x: canonicalize_state(x, state_mapper)
        )
        unmatched = df[df["state_canonical"].isna()]["state"].unique()
        if len(unmatched) > 0:
            print(f"  WARNING [{df_label}]: {len(unmatched)} states could not be canonicalized:")
            for s in sorted(unmatched):
                print(f"    - '{s}'")
                unmatched_states.add(s)

        # Filter to valid states
        valid_mask = (
            df["state"].apply(is_state_row) &
            df["state_canonical"].notna()
        )
        if df_label == "current":
            df_current = df[valid_mask].copy()
            df_current["state"] = df_current["state_canonical"]
            df_current = df_current.drop(columns=["state_canonical"])
        else:
            df_constant = df[valid_mask].copy()
            df_constant["state"] = df_constant["state_canonical"]
            df_constant = df_constant.drop(columns=["state_canonical"])

    print(f"  Current prices: {df_current['state'].nunique()} states")
    print(f"  Constant prices: {df_constant['state'].nunique()} states")

    # Merge current and constant
    print("\nMerging current and constant price data...")
    df_merged = pd.merge(
        df_current,
        df_constant,
        on=["state", "fiscal_year"],
        how="outer"
    )
    print(f"  Merged shape: {df_merged.shape}")

    # Convert lakh to crore
    df_merged["gsdp_current_crore"] = df_merged["gsdp_current_lakh"] / 100
    df_merged["gsdp_constant_crore"] = df_merged["gsdp_constant_lakh"] / 100

    # Compute YoY growth from constant-price series
    print("Computing YoY growth rates...")
    df_merged = df_merged.sort_values(["state", "fiscal_year"]).reset_index(drop=True)
    df_merged["gsdp_growth_pct"] = (
        df_merged.groupby("state")["gsdp_constant_crore"]
        .pct_change(fill_method=None) * 100
    )

    # Compute GSDP rank per FY (by current prices)
    print("Computing GSDP rankings...")
    df_merged["gsdp_rank"] = np.nan
    for fy in df_merged["fiscal_year"].unique():
        fy_mask = (df_merged["fiscal_year"] == fy) & df_merged["gsdp_current_crore"].notna()
        if fy_mask.sum() >= 5:
            df_merged.loc[fy_mask, "gsdp_rank"] = (
                df_merged.loc[fy_mask, "gsdp_current_crore"]
                .rank(ascending=False, method="min")
            )

    # Drop lakh columns (keep crore)
    df_final = df_merged[[
        "state",
        "fiscal_year",
        "gsdp_current_crore",
        "gsdp_constant_crore",
        "gsdp_growth_pct",
        "gsdp_rank",
    ]].copy()

    # Validate
    print("\nValidation checks...")
    assert len(df_final) > 0, "ERROR: No GSDP data produced!"
    assert "Maharashtra" in df_final["state"].values, "ERROR: Maharashtra not found in GSDP data!"

    # Check FY range overlap with index (2017-18 to 2024-25)
    index_fys = {"2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"}
    gsdp_fys = set(df_final["fiscal_year"].unique())
    overlap = index_fys & gsdp_fys
    print(f"  GSDP FY range: {sorted(gsdp_fys)[0]} to {sorted(gsdp_fys)[-1]}")
    print(f"  Overlap with index FYs: {len(overlap)} ({sorted(overlap)})")
    if len(overlap) == 0:
        print("  WARNING: No overlapping fiscal years with index data!")

    # Save
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved to: {OUTPUT_PARQUET}")
    print(f"File size: {OUTPUT_PARQUET.stat().st_size / 1024:.1f} KB")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total rows: {len(df_final)}")
    print(f"Unique states: {df_final['state'].nunique()}")
    print(f"FY range: {sorted(df_final['fiscal_year'].unique())[0]} to {sorted(df_final['fiscal_year'].unique())[-1]}")

    # Spot check: Maharashtra
    print("\nSpot check - Maharashtra:")
    maha = df_final[df_final["state"] == "Maharashtra"].sort_values("fiscal_year")
    if len(maha) > 0:
        print(maha.tail(5).to_string(index=False))
    else:
        print("  WARNING: No data found for Maharashtra!")

    # Top 5 by latest FY
    fys_sorted = sorted(df_final["fiscal_year"].unique())
    latest_fy = fys_sorted[-1]
    latest = df_final[
        (df_final["fiscal_year"] == latest_fy) & df_final["gsdp_rank"].notna()
    ].sort_values("gsdp_rank")
    print(f"\nTop 5 states by GSDP ({latest_fy}):")
    for _, row in latest.head(5).iterrows():
        print(f"  {int(row['gsdp_rank']):2d}. {row['state']:25s} "
              f"Rs {row['gsdp_current_crore']:,.0f} Cr")

    if unmatched_states:
        print(f"\nWARNING: {len(unmatched_states)} states could not be matched.")
        print("Consider adding aliases to data/reference/state_metadata.csv:")
        for s in sorted(unmatched_states):
            print(f"  - '{s}'")

    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
