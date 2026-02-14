"""
04_fetch_epfo.py

Parses the EPFO payroll Excel file into a clean parquet format.

The EPFO Excel has a complex structure with 6 age bucket sections, each containing
32 state rows. This script:
1. Identifies all age bucket sections
2. Extracts state-wise data from each section
3. Sums payroll across all age buckets per (state, column)
4. Classifies columns as annual FY or monthly periods
5. Outputs a clean parquet file

Output: data/processed/epfo_clean.parquet
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import re

# Import utilities
sys.path.append(str(Path(__file__).parent))
from utils import (
    PROJECT_ROOT,
    DATA_RAW,
    DATA_PROCESSED,
    load_state_mapper,
    canonicalize_state
)


def find_age_bucket_sections(df):
    """
    Find all age bucket section start rows.

    Returns:
        list of tuples: [(section_name, header_row_index, data_start_row, data_end_row), ...]
    """
    sections = []

    for idx, val in enumerate(df.iloc[:, 0]):
        if pd.notna(val) and isinstance(val, str) and val.strip().startswith("Age Bucket:"):
            section_name = val.strip()
            header_row = idx + 1
            data_start = idx + 2

            # Find the end of this section (look for "SUB TOTAL")
            data_end = None
            for end_idx in range(data_start, len(df)):
                end_val = df.iloc[end_idx, 0]
                if pd.notna(end_val) and isinstance(end_val, str) and end_val.strip().upper() == "SUB TOTAL":
                    data_end = end_idx
                    break

            if data_end is not None:
                sections.append((section_name, header_row, data_start, data_end))

    return sections


def extract_section_data(df, header_row, data_start, data_end):
    """
    Extract state-wise data from a single age bucket section.

    Returns:
        dict: {state_name: {col_index: value}}
    """
    section_data = {}

    for row_idx in range(data_start, data_end):
        state_name = df.iloc[row_idx, 0]

        # Skip if not a valid state name
        if pd.isna(state_name) or not isinstance(state_name, str):
            continue

        state_name = state_name.strip()
        if not state_name or state_name.upper() == "SUB TOTAL":
            continue

        # Extract values for all data columns (columns 1 onwards)
        state_values = {}
        for col_idx in range(1, df.shape[1]):
            value = df.iloc[row_idx, col_idx]
            # Convert to numeric, treating NaN/empty as 0
            if pd.isna(value):
                numeric_value = 0
            else:
                try:
                    numeric_value = float(value)
                except (ValueError, TypeError):
                    numeric_value = 0

            state_values[col_idx] = numeric_value

        section_data[state_name] = state_values

    return section_data


def sum_across_sections(sections_data):
    """
    Sum payroll values across all age bucket sections for each (state, column).

    Args:
        sections_data: list of dicts from extract_section_data

    Returns:
        dict: {state_name: {col_index: total_value}}
    """
    total_data = {}

    # Collect all unique state names across sections
    all_states = set()
    for section in sections_data:
        all_states.update(section.keys())

    # Sum across sections for each state
    for state in all_states:
        total_data[state] = {}

        # Determine all columns from first section
        if sections_data:
            first_section = sections_data[0]
            if state in first_section:
                cols = first_section[state].keys()
            else:
                # Use any state from first section to get columns
                sample_state = next(iter(first_section.keys()))
                cols = first_section[sample_state].keys()

            for col_idx in cols:
                total = 0
                for section in sections_data:
                    if state in section and col_idx in section[state]:
                        total += section[state][col_idx]

                total_data[state][col_idx] = total

    return total_data


def classify_columns(df, header_row):
    """
    Classify columns as annual FY or monthly periods.

    Returns:
        list of tuples: [(col_index, period_type, period_value, is_partial_year), ...]
        - period_type: "annual" or "monthly"
        - period_value: FY string (e.g., "2023-24") or YYYY-MM for monthly
        - is_partial_year: True for "2017-18 From Sep-17"
    """
    columns_info = []

    for col_idx in range(1, df.shape[1]):
        header_val = df.iloc[header_row, col_idx]

        if pd.isna(header_val):
            continue

        # Check if datetime object (monthly column)
        if isinstance(header_val, (datetime, pd.Timestamp)):
            month_str = pd.to_datetime(header_val).strftime("%Y-%m")
            columns_info.append((col_idx, "monthly", month_str, False))

        # Otherwise treat as FY string
        elif isinstance(header_val, str):
            fy_str = header_val.strip()

            # Handle "2017-18 From Sep-17" format
            is_partial = False
            if "From" in fy_str or "from" in fy_str:
                is_partial = True
                # Extract just the FY part
                match = re.match(r'(\d{4}-\d{2})', fy_str)
                if match:
                    fy_str = match.group(1)

            columns_info.append((col_idx, "annual", fy_str, is_partial))

    return columns_info


def build_output_dataframe(total_data, columns_info):
    """
    Build the final output dataframe with canonical state names.

    Returns:
        pd.DataFrame with schema:
        - state: str
        - fiscal_year: str
        - period_type: str
        - month: str|None
        - epfo_payroll: int
        - is_partial_year: bool
    """
    rows = []

    # Load state mapper for canonicalization
    state_mapper = load_state_mapper()

    for state_raw, col_data in total_data.items():
        # Canonicalize state name
        # First convert to title case
        state_title = state_raw.strip().title()
        canonical_state = canonicalize_state(state_title, state_mapper)

        if canonical_state is None:
            print(f"Warning: Could not canonicalize state '{state_raw}', skipping")
            continue

        # Create row for each column
        for col_idx, period_type, period_value, is_partial in columns_info:
            if col_idx in col_data:
                payroll_value = int(round(col_data[col_idx]))

                if period_type == "annual":
                    rows.append({
                        "state": canonical_state,
                        "fiscal_year": period_value,
                        "period_type": "annual",
                        "month": None,
                        "epfo_payroll": payroll_value,
                        "is_partial_year": is_partial
                    })
                else:  # monthly
                    # Extract FY from YYYY-MM
                    # FY runs Apr-Mar, so Apr-Dec is current FY, Jan-Mar is previous FY
                    year, month = map(int, period_value.split("-"))
                    if month >= 4:
                        fy = f"{year}-{str(year+1)[2:]}"
                    else:
                        fy = f"{year-1}-{str(year)[2:]}"

                    rows.append({
                        "state": canonical_state,
                        "fiscal_year": fy,
                        "period_type": "monthly",
                        "month": period_value,
                        "epfo_payroll": payroll_value,
                        "is_partial_year": False
                    })

    df = pd.DataFrame(rows)
    return df


def print_summary(df):
    """Print summary statistics about the parsed data."""
    print("\n" + "="*80)
    print("EPFO PAYROLL DATA SUMMARY")
    print("="*80)

    # States
    states = sorted(df["state"].unique())
    print(f"\nStates ({len(states)}):")
    print(", ".join(states))

    # Fiscal year range
    fiscal_years = sorted(df["fiscal_year"].unique())
    print(f"\nFiscal Years ({len(fiscal_years)}):")
    print(", ".join(fiscal_years))

    # Total India payroll per year (annual only)
    print("\nTotal India Annual Payroll (All States):")
    annual_data = df[df["period_type"] == "annual"].copy()
    india_total = annual_data.groupby("fiscal_year")["epfo_payroll"].sum().sort_index()
    for fy, total in india_total.items():
        partial_flag = " (partial year)" if annual_data[annual_data["fiscal_year"] == fy]["is_partial_year"].any() else ""
        print(f"  {fy}: {total:,}{partial_flag}")

    # Tamil Nadu spot check
    print("\nTamil Nadu Annual Payroll (Spot Check):")
    tn_data = df[(df["state"] == "Tamil Nadu") & (df["period_type"] == "annual")].copy()
    tn_data = tn_data.sort_values("fiscal_year")
    for _, row in tn_data.iterrows():
        partial_flag = " (partial year)" if row["is_partial_year"] else ""
        print(f"  {row['fiscal_year']}: {row['epfo_payroll']:,}{partial_flag}")

    # Monthly data sample
    monthly_data = df[df["period_type"] == "monthly"]
    if not monthly_data.empty:
        print(f"\nMonthly Data Points: {len(monthly_data)}")
        print(f"Monthly Period Range: {monthly_data['month'].min()} to {monthly_data['month'].max()}")

    print("\n" + "="*80)


def main():
    """Main execution function."""
    print("Starting EPFO payroll data processing...")

    # Locate the EPFO Excel file
    epfo_file = DATA_RAW / "epfo" / "epfo_sample_sep2025.xlsx"

    # Check fallback location
    if not epfo_file.exists():
        epfo_file = DATA_RAW / "udyam" / "epfo_sample_sep2025.xlsx"

    if not epfo_file.exists():
        raise FileNotFoundError(
            f"EPFO file not found at:\n"
            f"  - {DATA_RAW / 'epfo' / 'epfo_sample_sep2025.xlsx'}\n"
            f"  - {DATA_RAW / 'udyam' / 'epfo_sample_sep2025.xlsx'}"
        )

    print(f"Reading: {epfo_file}")

    # Read the Excel file without headers
    df = pd.read_excel(epfo_file, sheet_name="Statewise", header=None)
    print(f"Loaded sheet with {df.shape[0]} rows x {df.shape[1]} columns")

    # Find all age bucket sections
    sections = find_age_bucket_sections(df)
    print(f"\nFound {len(sections)} age bucket sections:")
    for section_name, header_row, data_start, data_end in sections:
        print(f"  - {section_name}: rows {data_start}-{data_end} ({data_end - data_start} states)")

    if not sections:
        raise ValueError("No age bucket sections found in the Excel file")

    # Extract data from each section
    sections_data = []
    for section_name, header_row, data_start, data_end in sections:
        section_data = extract_section_data(df, header_row, data_start, data_end)
        sections_data.append(section_data)
        print(f"Extracted {len(section_data)} states from {section_name}")

    # Sum across all age bucket sections
    print("\nSumming payroll across all age buckets...")
    total_data = sum_across_sections(sections_data)
    print(f"Computed totals for {len(total_data)} states")

    # Classify columns (use header row from first section)
    first_header_row = sections[0][1]
    columns_info = classify_columns(df, first_header_row)
    annual_cols = [c for c in columns_info if c[1] == "annual"]
    monthly_cols = [c for c in columns_info if c[1] == "monthly"]
    print(f"\nColumn classification:")
    print(f"  - Annual FY columns: {len(annual_cols)}")
    print(f"  - Monthly columns: {len(monthly_cols)}")

    # Build output dataframe
    print("\nBuilding output dataframe...")
    output_df = build_output_dataframe(total_data, columns_info)

    # Save to parquet
    output_file = DATA_PROCESSED / "epfo_clean.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_file, index=False)
    print(f"\nSaved {len(output_df)} rows to: {output_file}")

    # Print summary
    print_summary(output_df)

    print("\nEPFO data processing complete!")


if __name__ == "__main__":
    main()
