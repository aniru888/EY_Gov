"""
02_fetch_electricity.py

Processes POSOCO daily electricity CSV into clean monthly parquet.

Input:  data/raw/electricity/POSOCO_data.csv (daily state-level energy met data)
Output: data/processed/electricity_clean.parquet (monthly aggregated by state)

Output Schema:
- state: str (canonical state name)
- fiscal_year: str (e.g., "2023-24")
- month: str (YYYY-MM format)
- electricity_mu: float (total monthly energy demand in MU)
- days_with_data: int (count of non-null days in the month)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime

# Import utilities
from utils import (
    PROJECT_ROOT,
    DATA_RAW,
    DATA_PROCESSED,
    load_state_mapper,
    canonicalize_state,
    month_to_fiscal_year
)


# Constants
INPUT_FILE = DATA_RAW / "electricity" / "POSOCO_data.csv"
OUTPUT_FILE = DATA_PROCESSED / "electricity_clean.parquet"

# Transition date for DD/DNH -> DNHDDPDCL
DD_DNH_TRANSITION_DATE = pd.Timestamp("2020-01-26")

# Prefixes to exclude (regional aggregates and non-state entities)
EXCLUDE_PREFIXES = [
    "NR:", "WR:", "SR:", "ER:", "NER:",  # Regional aggregates
    "India:",                             # National aggregate
    "DVC:", "Essar", "AMNSIL:", "BALCO:", "Railways", "RIL "  # Non-state entities
]


def is_state_column(col_name: str) -> bool:
    """
    Check if column represents a state (not a region or non-state entity).

    Args:
        col_name: Column name from POSOCO CSV

    Returns:
        True if column represents a state, False otherwise
    """
    if not isinstance(col_name, str):
        return False

    # Check if column matches the pattern ": EnergyMet"
    if ": EnergyMet" not in col_name:
        return False

    # Exclude regional aggregates and non-state entities
    for prefix in EXCLUDE_PREFIXES:
        if col_name.startswith(prefix):
            return False

    return True


def extract_state_name(col_name: str) -> str:
    """
    Extract state name from column header.

    Args:
        col_name: Column name like "Punjab: EnergyMet"

    Returns:
        State name like "Punjab"
    """
    return col_name.replace(": EnergyMet", "").strip()


def handle_dd_dnh_transition(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Handle the DD/DNH -> DNHDDPDCL transition.

    Before 2020-01-26: Sum DD + DNH into a single entry
    After 2020-01-26: Use DNHDDPDCL directly

    Args:
        df: DataFrame with columns including 'date' and state names
        date_col: Name of the date column

    Returns:
        DataFrame with DD/DNH transition handled
    """
    # Check which columns exist
    has_dd = "DD" in df.columns
    has_dnh = "DNH" in df.columns
    has_dnhddpdcl = "DNHDDPDCL" in df.columns

    if not (has_dd or has_dnh or has_dnhddpdcl):
        # No DD/DNH/DNHDDPDCL columns found, return as-is
        return df

    df = df.copy()

    # Create a unified column for the union territory
    df["Dadra & Nagar Haveli and Daman & Diu"] = np.nan

    # Before transition: sum DD + DNH
    before_mask = df[date_col] < DD_DNH_TRANSITION_DATE
    if before_mask.any():
        if has_dd and has_dnh:
            df.loc[before_mask, "Dadra & Nagar Haveli and Daman & Diu"] = (
                df.loc[before_mask, "DD"].fillna(0) +
                df.loc[before_mask, "DNH"].fillna(0)
            )
            # Set to NaN if both were NaN
            both_null = df.loc[before_mask, "DD"].isna() & df.loc[before_mask, "DNH"].isna()
            df.loc[before_mask & both_null, "Dadra & Nagar Haveli and Daman & Diu"] = np.nan
        elif has_dd:
            df.loc[before_mask, "Dadra & Nagar Haveli and Daman & Diu"] = df.loc[before_mask, "DD"]
        elif has_dnh:
            df.loc[before_mask, "Dadra & Nagar Haveli and Daman & Diu"] = df.loc[before_mask, "DNH"]

    # After transition: use DNHDDPDCL
    after_mask = df[date_col] >= DD_DNH_TRANSITION_DATE
    if after_mask.any() and has_dnhddpdcl:
        df.loc[after_mask, "Dadra & Nagar Haveli and Daman & Diu"] = df.loc[after_mask, "DNHDDPDCL"]

    # Drop the original columns
    cols_to_drop = []
    if has_dd:
        cols_to_drop.append("DD")
    if has_dnh:
        cols_to_drop.append("DNH")
    if has_dnhddpdcl:
        cols_to_drop.append("DNHDDPDCL")

    df = df.drop(columns=cols_to_drop)

    return df


def process_electricity_data():
    """
    Main processing function to convert POSOCO CSV to clean monthly parquet.
    """
    print("="*80)
    print("POSOCO Electricity Data Processing")
    print("="*80)

    # Check input file exists
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    print(f"\n1. Loading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Loaded {len(df):,} rows × {len(df.columns):,} columns")

    # Parse date column (first column, YYYYMMDD format)
    print("\n2. Parsing dates...")
    date_col_name = df.columns[0]
    df["date"] = pd.to_datetime(df[date_col_name], format="%Y%m%d")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

    # Identify state columns
    print("\n3. Identifying state columns...")
    state_columns = [col for col in df.columns if is_state_column(col)]
    print(f"   Found {len(state_columns)} state columns")

    # Create mapping from column name to simplified state name
    col_to_state = {col: extract_state_name(col) for col in state_columns}

    # Select date and state columns
    df_states = df[["date"] + state_columns].copy()

    # Rename columns to simplified state names
    df_states = df_states.rename(columns=col_to_state)

    print(f"   State columns: {sorted(col_to_state.values())}")

    # Handle DD/DNH -> DNHDDPDCL transition
    print("\n4. Handling DD/DNH -> DNHDDPDCL transition...")
    df_states = handle_dd_dnh_transition(df_states, date_col="date")

    # Melt from wide to long format
    print("\n5. Reshaping from wide to long format...")
    df_long = df_states.melt(
        id_vars=["date"],
        var_name="raw_state",
        value_name="electricity_mu"
    )
    print(f"   Reshaped to {len(df_long):,} rows")

    # Drop rows with missing electricity values
    before_drop = len(df_long)
    df_long = df_long.dropna(subset=["electricity_mu"])
    print(f"   Dropped {before_drop - len(df_long):,} rows with missing values")

    # Add month column (YYYY-MM format)
    print("\n6. Creating month column...")
    df_long["month"] = df_long["date"].dt.to_period("M").astype(str)

    # Aggregate from daily to monthly
    print("\n7. Aggregating daily data to monthly...")
    df_monthly = df_long.groupby(["raw_state", "month"]).agg(
        electricity_mu=("electricity_mu", "sum"),
        days_with_data=("electricity_mu", "count")
    ).reset_index()
    print(f"   Aggregated to {len(df_monthly):,} monthly records")

    # Canonicalize state names
    print("\n8. Canonicalizing state names...")
    state_mapper = load_state_mapper()
    df_monthly["state"] = df_monthly["raw_state"].apply(
        lambda x: canonicalize_state(x, state_mapper)
    )

    # Check for unmapped states
    unmapped = df_monthly[df_monthly["state"].isna()]["raw_state"].unique()
    if len(unmapped) > 0:
        print(f"   WARNING: {len(unmapped)} unmapped states found:")
        for state in unmapped:
            print(f"      - {state}")

    # Drop unmapped states
    df_monthly = df_monthly.dropna(subset=["state"])
    df_monthly = df_monthly.drop(columns=["raw_state"])

    # Add fiscal year column
    print("\n9. Adding fiscal year column...")
    df_monthly["fiscal_year"] = df_monthly["month"].apply(month_to_fiscal_year)

    # Reorder columns
    df_monthly = df_monthly[[
        "state",
        "fiscal_year",
        "month",
        "electricity_mu",
        "days_with_data"
    ]]

    # Sort by state and month
    df_monthly = df_monthly.sort_values(["state", "month"]).reset_index(drop=True)

    # Save to parquet
    print(f"\n10. Saving to: {OUTPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_monthly.to_parquet(OUTPUT_FILE, index=False)
    print(f"    Saved {len(df_monthly):,} rows")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nUnique states: {df_monthly['state'].nunique()}")
    print(f"States: {sorted(df_monthly['state'].unique())}")

    print(f"\nDate range:")
    print(f"  First month: {df_monthly['month'].min()}")
    print(f"  Last month:  {df_monthly['month'].max()}")

    print(f"\nFiscal years covered: {sorted(df_monthly['fiscal_year'].unique())}")

    print(f"\nTotal monthly records: {len(df_monthly):,}")

    # Spot-check: Maharashtra monthly values
    print("\n" + "-"*80)
    print("SPOT CHECK: Maharashtra Monthly Values (First 10)")
    print("-"*80)
    maha = df_monthly[df_monthly["state"] == "Maharashtra"].head(10)
    if len(maha) > 0:
        print(maha.to_string(index=False))
    else:
        print("No Maharashtra data found")

    # Data quality check
    print("\n" + "-"*80)
    print("DATA QUALITY")
    print("-"*80)
    print(f"Average days with data per month: {df_monthly['days_with_data'].mean():.1f}")
    print(f"Min days with data: {df_monthly['days_with_data'].min()}")
    print(f"Max days with data: {df_monthly['days_with_data'].max()}")

    months_with_low_data = df_monthly[df_monthly["days_with_data"] < 28]
    if len(months_with_low_data) > 0:
        print(f"\nMonths with <28 days of data: {len(months_with_low_data)}")

    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    process_electricity_data()
