"""
03_fetch_rbi.py
===============
Parse RBI Scheduled Commercial Bank Credit by State Excel (Table 156)
into a clean parquet file.

Input:  data/raw/rbi/rbi_scb_credit_by_state.xlsx
Output: data/processed/rbi_clean.parquet

RBI Excel Structure:
- Two sheets: T_156(i) covering 2004-2014, T_156(ii) covering 2015-2025
- Each sheet has header row at index 4 (0-indexed)
- Year columns represent end-of-March values (stock measure)
- Values are in ₹ crore
- Contains region aggregates (to be filtered out) and state-level data

Output Schema:
- state: canonical state name
- fiscal_year: e.g., "2023-24" (converted from RBI year "2024")
- period_type: always "annual"
- bank_credit_crore: outstanding credit as of end-March
- credit_yoy_delta: year-over-year change (flow proxy)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io

# Import utilities
from utils import (
    PROJECT_ROOT,
    DATA_RAW,
    DATA_PROCESSED,
    load_state_mapper,
    canonicalize_state,
    is_state_row
)

# Constants
RBI_EXCEL_PATH = DATA_RAW / "rbi" / "rbi_scb_credit_by_state.xlsx"
OUTPUT_PARQUET = DATA_PROCESSED / "rbi_clean.parquet"

# Region headers and metadata rows to filter out
SKIP_PATTERNS = [
    "NORTHERN REGION",
    "NORTH-EASTERN REGION",
    "EASTERN REGION",
    "CENTRAL REGION",
    "WESTERN REGION",
    "SOUTHERN REGION",
    "ALL-INDIA",
    "(As at end-March)",
    "(₹ crore)",
    "TABLE 156",
]


def rbi_year_to_fiscal_year(rbi_year: int) -> str:
    """
    Convert RBI year to fiscal year format.

    RBI year "2024" means end-March 2024 = FY 2023-24
    RBI year "2004" means end-March 2004 = FY 2003-04

    Args:
        rbi_year: Year as shown in RBI table (e.g., 2024)

    Returns:
        Fiscal year string (e.g., "2023-24")
    """
    fy_start = rbi_year - 1
    fy_end = rbi_year % 100  # Get last 2 digits
    return f"{fy_start}-{fy_end:02d}"


def clean_state_name(state_name: str) -> str:
    """
    Clean state name by removing footnote markers and extra whitespace.

    Args:
        state_name: Raw state name from Excel

    Returns:
        Cleaned state name
    """
    if pd.isna(state_name):
        return ""

    # Remove footnote markers (asterisks, daggers, etc.)
    cleaned = str(state_name).strip()
    cleaned = cleaned.rstrip("*†‡§¶#")
    return cleaned.strip()


def should_skip_row(state_name: str) -> bool:
    """
    Determine if a row should be skipped based on state name.

    Args:
        state_name: State name to check

    Returns:
        True if row should be skipped, False otherwise
    """
    if not state_name or pd.isna(state_name):
        return True

    state_upper = str(state_name).upper()

    # Check against skip patterns
    for pattern in SKIP_PATTERNS:
        if pattern in state_upper:
            return True

    return False


def parse_rbi_sheet(excel_file: Path, sheet_name: str, header_row: int = 4) -> pd.DataFrame:
    """
    Parse a single RBI sheet into long format.

    Args:
        excel_file: Path to Excel file
        sheet_name: Name of sheet to parse
        header_row: Row index for header (0-indexed)

    Returns:
        DataFrame with columns: state, fiscal_year, bank_credit_crore
    """
    print(f"  Parsing sheet: {sheet_name}")

    # Read the sheet with header at specified row
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=header_row)

    print(f"    Raw shape: {df.shape}")
    print(f"    Columns: {list(df.columns[:5])}")

    # Drop empty leading columns (Excel often has blank col 0)
    while len(df.columns) > 0 and str(df.columns[0]).startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])

    # Find and rename the state column
    state_col = None
    for col in df.columns:
        if isinstance(col, str) and ("Region" in col or "State" in col or "Union" in col):
            state_col = col
            break

    if state_col is None:
        # Fallback: use first column
        state_col = df.columns[0]

    df = df.rename(columns={state_col: "state"})

    # Clean state names
    df["state"] = df["state"].apply(clean_state_name)

    # Filter out region headers and metadata rows
    df = df[~df["state"].apply(should_skip_row)].copy()

    # Get year columns (all columns except 'state')
    year_cols = [col for col in df.columns if col != "state"]

    # Convert year columns to string to ensure consistency
    year_cols_str = [str(col) for col in year_cols]
    df.columns = ["state"] + year_cols_str

    # Melt from wide to long format
    df_long = df.melt(
        id_vars=["state"],
        value_vars=year_cols_str,
        var_name="rbi_year",
        value_name="bank_credit_crore"
    )

    # Convert RBI year to fiscal year
    df_long["rbi_year"] = df_long["rbi_year"].astype(int)
    df_long["fiscal_year"] = df_long["rbi_year"].apply(rbi_year_to_fiscal_year)

    # Drop RBI year column
    df_long = df_long.drop(columns=["rbi_year"])

    # Convert bank_credit_crore to float
    df_long["bank_credit_crore"] = pd.to_numeric(df_long["bank_credit_crore"], errors="coerce")

    # Drop rows with NaN credit values
    df_long = df_long.dropna(subset=["bank_credit_crore"])

    print(f"    Parsed shape: {df_long.shape}")
    print(f"    States found: {df_long['state'].nunique()}")

    return df_long


def main():
    """Main execution function."""
    # Fix Windows console encoding for ₹ symbol in RBI data
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("="*80)
    print("RBI Bank Credit Data Processing")
    print("="*80)

    # Check if input file exists
    if not RBI_EXCEL_PATH.exists():
        print(f"ERROR: Input file not found: {RBI_EXCEL_PATH}")
        sys.exit(1)

    print(f"\nInput file: {RBI_EXCEL_PATH}")
    print(f"Output file: {OUTPUT_PARQUET}")

    # Load state mapper
    print("\nLoading state mapper...")
    state_mapper = load_state_mapper()
    print(f"  Loaded {len(state_mapper)} state mappings")

    # Parse both sheets
    print("\nParsing RBI sheets...")

    df_sheet1 = parse_rbi_sheet(RBI_EXCEL_PATH, "T_156(i)")
    df_sheet2 = parse_rbi_sheet(RBI_EXCEL_PATH, "T_156(ii)")

    # Concatenate both sheets
    print("\nCombining sheets...")
    df_combined = pd.concat([df_sheet1, df_sheet2], ignore_index=True)
    print(f"  Combined shape: {df_combined.shape}")

    # Handle duplicates: if a state appears in both sheets for same year, prefer sheet 2
    print("\nHandling duplicates...")
    initial_rows = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=["state", "fiscal_year"], keep="last")
    duplicates_removed = initial_rows - len(df_combined)
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Final shape: {df_combined.shape}")

    # Canonicalize state names
    print("\nCanonicalizing state names...")
    df_combined["state_canonical"] = df_combined["state"].apply(
        lambda x: canonicalize_state(x, state_mapper)
    )

    # Filter: keep rows that look like states and can be canonicalized
    df_valid = df_combined[
        df_combined["state"].apply(is_state_row) &
        df_combined["state_canonical"].notna()
    ].copy()

    # Replace state with canonical name
    df_valid["state"] = df_valid["state_canonical"]
    df_valid = df_valid.drop(columns=["state_canonical"])

    print(f"  Valid state rows: {len(df_valid)}")
    print(f"  Unique canonical states: {df_valid['state'].nunique()}")

    # Add period_type column
    df_valid["period_type"] = "annual"

    # Sort by state and fiscal_year for YoY calculation
    df_valid = df_valid.sort_values(["state", "fiscal_year"]).reset_index(drop=True)

    # Compute year-over-year delta
    print("\nComputing year-over-year delta...")
    df_valid["credit_yoy_delta"] = df_valid.groupby("state")["bank_credit_crore"].diff()

    # Reorder columns
    df_final = df_valid[[
        "state",
        "fiscal_year",
        "period_type",
        "bank_credit_crore",
        "credit_yoy_delta"
    ]]

    # Create output directory if it doesn't exist
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    print(f"\nSaving to parquet: {OUTPUT_PARQUET}")
    df_final.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"  Saved {len(df_final)} rows")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nTotal rows: {len(df_final)}")
    print(f"Unique states: {df_final['state'].nunique()}")
    print(f"Year range: {df_final['fiscal_year'].min()} to {df_final['fiscal_year'].max()}")

    print("\nStates covered:")
    for state in sorted(df_final["state"].unique()):
        count = len(df_final[df_final["state"] == state])
        print(f"  {state}: {count} years")

    # Spot check: Maharashtra
    print("\nSpot check - Maharashtra:")
    maha_data = df_final[df_final["state"] == "Maharashtra"].sort_values("fiscal_year")

    if len(maha_data) > 0:
        # Show last 5 years
        print("\nLast 5 years:")
        print(maha_data.tail(5).to_string(index=False))

        # Show statistics
        print("\nStatistics:")
        print(f"  Mean credit: Rs {maha_data['bank_credit_crore'].mean():.2f} crore")
        print(f"  Max credit: Rs {maha_data['bank_credit_crore'].max():.2f} crore ({maha_data.loc[maha_data['bank_credit_crore'].idxmax(), 'fiscal_year']})")
        print(f"  Min credit: Rs {maha_data['bank_credit_crore'].min():.2f} crore ({maha_data.loc[maha_data['bank_credit_crore'].idxmin(), 'fiscal_year']})")

        # Show YoY deltas
        yoy_stats = maha_data["credit_yoy_delta"].dropna()
        if len(yoy_stats) > 0:
            print(f"  Mean YoY delta: Rs {yoy_stats.mean():.2f} crore")
            print(f"  Max YoY delta: Rs {yoy_stats.max():.2f} crore")
    else:
        print("  WARNING: No data found for Maharashtra!")

    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)


if __name__ == "__main__":
    main()
