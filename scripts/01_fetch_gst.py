"""
01_fetch_gst.py

Parses 9 GST Excel files from data/raw/gst/ into a clean parquet file.

Input:
  - data/raw/gst/statewise_GST_collection_{fy}.xlsx (9 files, FY 2017-18 through 2025-26)

Output:
  - data/processed/gst_clean.parquet

Schema:
  - state: str          - canonical state name
  - fiscal_year: str    - e.g., "2023-24"
  - month: str          - e.g., "2023-04" (YYYY-MM format)
  - gst_total: float    - total GST collection in crore (CGST+SGST+IGST+Cess)
"""

from pathlib import Path
from typing import Optional
import datetime
import pandas as pd
import numpy as np

from utils import (
    PROJECT_ROOT,
    DATA_RAW,
    DATA_PROCESSED,
    GST_FISCAL_YEARS,
    load_state_mapper,
    canonicalize_state,
    is_state_row,
)


def detect_sheet_name(file_path: Path, fiscal_year: str) -> str:
    """Detect the correct sheet name for GST data.

    Args:
        file_path: Path to Excel file
        fiscal_year: FY string like "2020-21"

    Returns:
        Sheet name to use
    """
    # 2020-21 onwards use "Collections-Statewise", earlier files use first sheet
    fy_start = int(fiscal_year.split("-")[0])
    if fy_start >= 2020:
        return "Collections-Statewise"
    else:
        return 0  # Use first sheet for 2017-18, 2018-19, 2019-20


def parse_gst_file(file_path: Path, fiscal_year: str, state_mapper: dict) -> pd.DataFrame:
    """Parse a single GST Excel file into a DataFrame.

    Args:
        file_path: Path to the Excel file
        fiscal_year: FY string like "2023-24"
        state_mapper: State name canonicalization mapper

    Returns:
        DataFrame with columns: state, fiscal_year, month, gst_total
    """
    print(f"  Processing {fiscal_year}...")

    try:
        sheet_name = detect_sheet_name(file_path, fiscal_year)

        # Read the entire sheet to inspect structure
        df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

        # Row 5 (index 4) contains main headers: State CD, State, FY label, then datetime month columns
        # Row 6 (index 5) contains sub-headers: CGST, SGST, IGST, CESS, TOTAL (repeated)
        header_row = df_raw.iloc[4]
        subheader_row = df_raw.iloc[5]

        # Data starts from row 7 (index 6)
        data_rows = df_raw.iloc[6:].reset_index(drop=True)

        # Extract month columns from header_row
        # Column 0: State Cd, Column 1: State, Column 2: FY label
        # Columns 3-7: FY total group (skipped for monthly data)
        # Columns 8+: Monthly groups (5 columns per month)

        months = []
        month_indices = []

        # Start from column 7 (index 7) for monthly data
        # Each month has 5 sub-columns, we want the TOTAL (index 4 of each group)
        for col_idx in range(7, len(header_row), 5):
            cell_value = header_row.iloc[col_idx]

            # Check if this is a datetime object
            if pd.notna(cell_value):
                if isinstance(cell_value, (pd.Timestamp, datetime.datetime)):
                    month_str = cell_value.strftime("%Y-%m")
                    months.append(month_str)
                    # The TOTAL column is 4 positions ahead in the group
                    # Group structure: CGST(0), SGST(1), IGST(2), CESS(3), TOTAL(4)
                    total_col_idx = col_idx + 4
                    month_indices.append(total_col_idx)

        if not months:
            print(f"    WARNING: No months found for {fiscal_year}")
            return pd.DataFrame(columns=["state", "fiscal_year", "month", "gst_total"])

        # Extract data for each state and month
        records = []

        for _, row in data_rows.iterrows():
            # Column 1 contains state name
            state_raw = row.iloc[1]

            if pd.isna(state_raw):
                continue

            state_raw = str(state_raw).strip()

            # Filter non-state rows
            if not is_state_row(state_raw):
                continue

            # Canonicalize state name
            state_canonical = canonicalize_state(state_raw, state_mapper)
            if not state_canonical:
                # Skip if we can't canonicalize (likely a non-state row)
                continue

            # Extract monthly totals
            for month, col_idx in zip(months, month_indices):
                gst_value = row.iloc[col_idx]

                # Skip if value is missing or non-numeric
                if pd.isna(gst_value):
                    continue

                try:
                    gst_float = float(gst_value)
                    if gst_float < 0:
                        continue  # Skip negative values (likely errors)
                except (ValueError, TypeError):
                    continue

                records.append({
                    "state": state_canonical,
                    "fiscal_year": fiscal_year,
                    "month": month,
                    "gst_total": gst_float,
                })

        df = pd.DataFrame(records)
        print(f"    Extracted {len(df):,} rows, {len(df['state'].unique())} states, "
              f"{len(df['month'].unique())} months")

        return df

    except Exception as e:
        print(f"    ERROR processing {fiscal_year}: {e}")
        return pd.DataFrame(columns=["state", "fiscal_year", "month", "gst_total"])


def main():
    """Main execution: parse all GST files and save to parquet."""
    print("\n" + "=" * 70)
    print("GST Data Parser")
    print("=" * 70)

    # Load state mapper
    print("\nLoading state mapper...")
    state_mapper = load_state_mapper()
    print(f"  Loaded {len(state_mapper)} state aliases")

    # Process each fiscal year
    print(f"\nProcessing {len(GST_FISCAL_YEARS)} GST files...")

    all_dfs = []
    gst_dir = DATA_RAW / "gst"

    for fy in GST_FISCAL_YEARS:
        file_path = gst_dir / f"statewise_GST_collection_{fy}.xlsx"

        if not file_path.exists():
            print(f"  [SKIP] {fy} - file not found")
            continue

        df = parse_gst_file(file_path, fy, state_mapper)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("\n  ERROR: No data extracted from any file")
        return

    # Concatenate all dataframes
    print("\nCombining data from all fiscal years...")
    df_final = pd.concat(all_dfs, ignore_index=True)

    # Sort by state, month
    df_final = df_final.sort_values(["state", "month"]).reset_index(drop=True)

    # Save to parquet
    output_path = DATA_PROCESSED / "gst_clean.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(output_path, index=False)

    print(f"  Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total rows: {len(df_final):,}")
    print(f"States: {df_final['state'].nunique()}")
    print(f"Fiscal years: {sorted(df_final['fiscal_year'].unique())}")
    print(f"Date range: {df_final['month'].min()} to {df_final['month'].max()}")
    print(f"Total GST collected: {df_final['gst_total'].sum():,.0f} crore")

    # State breakdown
    print(f"\nTop 10 states by total GST collection:")
    state_totals = df_final.groupby("state")["gst_total"].sum().sort_values(ascending=False)
    for i, (state, total) in enumerate(state_totals.head(10).items(), 1):
        print(f"  {i:2d}. {state:25s} {total:12,.0f} crore")

    # Spot check: Maharashtra FY 2023-24 total
    print(f"\nSpot check - Maharashtra FY 2023-24:")
    mh_2324 = df_final[
        (df_final["state"] == "Maharashtra") &
        (df_final["fiscal_year"] == "2023-24")
    ]
    if not mh_2324.empty:
        print(f"  Months: {len(mh_2324)}")
        print(f"  Total collection: {mh_2324['gst_total'].sum():,.0f} crore")
        print(f"  Average per month: {mh_2324['gst_total'].mean():,.0f} crore")
        print(f"  Date range: {mh_2324['month'].min()} to {mh_2324['month'].max()}")
    else:
        print("  No data found for Maharashtra FY 2023-24")

    # Data quality checks
    print(f"\nData quality:")
    print(f"  Missing values: {df_final.isnull().sum().sum()}")
    print(f"  Negative values: {(df_final['gst_total'] < 0).sum()}")
    print(f"  Zero values: {(df_final['gst_total'] == 0).sum()}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
