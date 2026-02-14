"""
05_clean_and_merge.py

Merges all 4 clean parquet files into a single dataset.

Inputs:
  - data/processed/gst_clean.parquet       (monthly: state, fiscal_year, month, gst_total)
  - data/processed/electricity_clean.parquet (monthly: state, fiscal_year, month, electricity_mu, days_with_data)
  - data/processed/rbi_clean.parquet       (annual: state, fiscal_year, bank_credit_crore, credit_yoy_delta)
  - data/processed/epfo_clean.parquet      (annual+monthly: state, fiscal_year, epfo_payroll)

Output:
  - data/processed/merged.parquet

Schema:
  - state: str
  - slug: str
  - region: str
  - fiscal_year: str
  - period_type: str ("annual" or "monthly")
  - month: str|None (YYYY-MM for monthly rows, None for annual)
  - gst_total: float|NaN
  - electricity_mu: float|NaN
  - bank_credit_yoy: float|NaN
  - epfo_payroll: float|NaN
  - n_components: int (count of non-NaN component columns per row)
  - is_partial: bool (partial year data, e.g. FY 2017-18 for GST)
"""

import pandas as pd
import numpy as np

from utils import (
    PROJECT_ROOT,
    DATA_PROCESSED,
    DATA_REF,
    load_state_metadata,
)


# Constants
COMPONENT_COLS = ["gst_total", "electricity_mu", "bank_credit_yoy", "epfo_payroll"]
# Only include FYs from 2017-18 onwards (GST + EPFO start year)
MIN_FY = "2017-18"
# Partial year flags
PARTIAL_FYS = {
    "2017-18": ["gst"],  # GST started Jul 2017 (9 months, not 12)
}


def load_parquets():
    """Load all 4 clean parquet files."""
    print("Loading clean parquets...")

    gst = pd.read_parquet(DATA_PROCESSED / "gst_clean.parquet")
    print(f"  GST:         {gst.shape[0]:,} rows, {gst['state'].nunique()} states")

    elec = pd.read_parquet(DATA_PROCESSED / "electricity_clean.parquet")
    print(f"  Electricity: {elec.shape[0]:,} rows, {elec['state'].nunique()} states")

    rbi = pd.read_parquet(DATA_PROCESSED / "rbi_clean.parquet")
    print(f"  RBI:         {rbi.shape[0]:,} rows, {rbi['state'].nunique()} states")

    epfo = pd.read_parquet(DATA_PROCESSED / "epfo_clean.parquet")
    print(f"  EPFO:        {epfo.shape[0]:,} rows, {epfo['state'].nunique()} states")

    return gst, elec, rbi, epfo


def aggregate_monthly_to_annual(gst: pd.DataFrame, elec: pd.DataFrame):
    """Aggregate GST and electricity from monthly to annual (sum by state x FY)."""
    print("\nAggregating monthly data to annual...")

    gst_annual = gst.groupby(["state", "fiscal_year"]).agg(
        gst_total=("gst_total", "sum")
    ).reset_index()
    print(f"  GST annual:  {gst_annual.shape[0]:,} rows")

    elec_annual = elec.groupby(["state", "fiscal_year"]).agg(
        electricity_mu=("electricity_mu", "sum")
    ).reset_index()
    print(f"  Elec annual: {elec_annual.shape[0]:,} rows")

    return gst_annual, elec_annual


def build_annual_merged(gst_annual, elec_annual, rbi, epfo):
    """Outer join all 4 components on (state, fiscal_year) for annual data."""
    print("\nBuilding annual merged dataset...")

    # Prepare RBI: just need state, fiscal_year, credit_yoy_delta
    rbi_annual = rbi[["state", "fiscal_year", "credit_yoy_delta"]].copy()
    rbi_annual = rbi_annual.rename(columns={"credit_yoy_delta": "bank_credit_yoy"})

    # Prepare EPFO: only annual rows
    epfo_annual = epfo[epfo["period_type"] == "annual"][
        ["state", "fiscal_year", "epfo_payroll"]
    ].copy()

    # Outer join all 4 on (state, fiscal_year)
    merged = gst_annual.merge(
        elec_annual, on=["state", "fiscal_year"], how="outer"
    ).merge(
        rbi_annual, on=["state", "fiscal_year"], how="outer"
    ).merge(
        epfo_annual, on=["state", "fiscal_year"], how="outer"
    )

    merged["period_type"] = "annual"
    merged["month"] = None

    print(f"  Annual merged: {merged.shape[0]:,} rows, {merged['state'].nunique()} states")
    return merged


def build_monthly_merged(gst, elec):
    """Join GST + electricity monthly data (RBI/EPFO don't have monthly)."""
    print("\nBuilding monthly merged dataset...")

    gst_m = gst[["state", "fiscal_year", "month", "gst_total"]].copy()
    elec_m = elec[["state", "fiscal_year", "month", "electricity_mu"]].copy()

    merged = gst_m.merge(
        elec_m, on=["state", "fiscal_year", "month"], how="outer"
    )

    merged["bank_credit_yoy"] = np.nan
    merged["epfo_payroll"] = np.nan
    merged["period_type"] = "monthly"

    print(f"  Monthly merged: {merged.shape[0]:,} rows")
    return merged


def add_metadata(df):
    """Add slug, region from state_metadata.csv."""
    print("\nAdding state metadata (slug, region)...")
    meta = load_state_metadata()
    meta = meta[["canonical_name", "slug", "region"]].rename(
        columns={"canonical_name": "state"}
    )
    df = df.merge(meta, on="state", how="left")

    # Check for states missing metadata
    missing = df[df["slug"].isna()]["state"].unique()
    if len(missing) > 0:
        print(f"  WARNING: {len(missing)} states missing metadata: {list(missing)}")

    return df


def merge_population(df):
    """Left-join population data and compute per-capita columns."""
    print("\nMerging population data...")
    pop_path = DATA_PROCESSED / "population_clean.parquet"
    if not pop_path.exists():
        print("  WARNING: population_clean.parquet not found. Per-capita columns skipped.")
        df["population"] = np.nan
        for col in ["gst_per_capita", "electricity_per_capita", "credit_per_capita", "epfo_per_capita"]:
            df[col] = np.nan
        return df

    pop = pd.read_parquet(pop_path)
    print(f"  Population data: {len(pop)} rows, {pop['state'].nunique()} states")

    df = df.merge(pop, on=["state", "fiscal_year"], how="left")

    # Per-capita columns (only for annual rows with population)
    has_pop = df["population"].notna() & (df["population"] > 0)

    # GST per capita: Rs crore per lakh people
    df["gst_per_capita"] = np.nan
    mask = has_pop & df["gst_total"].notna()
    df.loc[mask, "gst_per_capita"] = df.loc[mask, "gst_total"] / (df.loc[mask, "population"] / 1e5)

    # Electricity per capita: MU per million people
    df["electricity_per_capita"] = np.nan
    mask = has_pop & df["electricity_mu"].notna()
    df.loc[mask, "electricity_per_capita"] = df.loc[mask, "electricity_mu"] / (df.loc[mask, "population"] / 1e6)

    # Credit per capita: Rs crore per lakh people
    df["credit_per_capita"] = np.nan
    mask = has_pop & df["bank_credit_yoy"].notna()
    df.loc[mask, "credit_per_capita"] = df.loc[mask, "bank_credit_yoy"] / (df.loc[mask, "population"] / 1e5)

    # EPFO per capita: persons per million people
    df["epfo_per_capita"] = np.nan
    mask = has_pop & df["epfo_payroll"].notna()
    df.loc[mask, "epfo_per_capita"] = df.loc[mask, "epfo_payroll"] / (df.loc[mask, "population"] / 1e6)

    matched = df[has_pop]["state"].nunique()
    print(f"  States with population data: {matched}")

    return df


def compute_coverage(df):
    """Count non-NaN components per row and flag partial years."""
    print("\nComputing coverage and partial year flags...")

    df["n_components"] = df[COMPONENT_COLS].notna().sum(axis=1)

    # Flag partial years
    df["is_partial"] = False
    for fy, components in PARTIAL_FYS.items():
        mask = df["fiscal_year"] == fy
        if mask.any():
            df.loc[mask, "is_partial"] = True

    return df


def main():
    print("=" * 70)
    print("Data Merge: Combining 4 Components")
    print("=" * 70)

    # Load
    gst, elec, rbi, epfo = load_parquets()

    # Aggregate monthly -> annual
    gst_annual, elec_annual = aggregate_monthly_to_annual(gst, elec)

    # Build annual merged
    annual = build_annual_merged(gst_annual, elec_annual, rbi, epfo)

    # Build monthly merged
    monthly = build_monthly_merged(gst, elec)

    # Concatenate annual + monthly
    print("\nConcatenating annual + monthly...")
    merged = pd.concat([annual, monthly], ignore_index=True)
    print(f"  Total rows: {merged.shape[0]:,}")

    # Filter to FY 2017-18 onwards
    print(f"\nFiltering to FY >= {MIN_FY}...")
    merged = merged[merged["fiscal_year"] >= MIN_FY].reset_index(drop=True)
    print(f"  Rows after filter: {merged.shape[0]:,}")

    # Add metadata
    merged = add_metadata(merged)

    # Merge population and compute per-capita
    merged = merge_population(merged)

    # Compute coverage
    merged = compute_coverage(merged)

    # Reorder columns
    col_order = [
        "state", "slug", "region", "fiscal_year", "period_type", "month",
        "gst_total", "electricity_mu", "bank_credit_yoy", "epfo_payroll",
        "population", "gst_per_capita", "electricity_per_capita",
        "credit_per_capita", "epfo_per_capita",
        "n_components", "is_partial"
    ]
    merged = merged[[c for c in col_order if c in merged.columns]]

    # Sort
    merged = merged.sort_values(
        ["state", "fiscal_year", "period_type", "month"]
    ).reset_index(drop=True)

    # Save
    output_path = DATA_PROCESSED / "merged.parquet"
    merged.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print(f"Total rows: {merged.shape[0]:,}")
    print(f"  Annual rows:  {(merged['period_type'] == 'annual').sum():,}")
    print(f"  Monthly rows: {(merged['period_type'] == 'monthly').sum():,}")
    print(f"States: {merged['state'].nunique()}")
    print(f"Fiscal years: {sorted(merged['fiscal_year'].unique())}")

    # Coverage matrix (annual only)
    print("\nAnnual coverage matrix (non-NaN components):")
    annual_only = merged[merged["period_type"] == "annual"]
    for n in [4, 3, 2, 1, 0]:
        count = (annual_only["n_components"] == n).sum()
        if count > 0:
            print(f"  {n} components: {count:,} rows")

    # Spot-check: Maharashtra
    print("\nSpot-check: Maharashtra annual data")
    mh = annual_only[annual_only["state"] == "Maharashtra"].sort_values("fiscal_year")
    if not mh.empty:
        cols = ["fiscal_year"] + COMPONENT_COLS + ["n_components"]
        print(mh[cols].tail(5).to_string(index=False))

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
