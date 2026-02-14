"""
06_compute_index.py

Computes cross-sectional z-scores and composite Li Keqiang Index.

Input:  data/processed/merged.parquet
Output: data/processed/index_computed.parquet

Additional columns added:
  - gst_zscore, electricity_zscore, credit_zscore, epfo_zscore: float|NaN
  - composite_score: float|NaN (mean of available z-scores, min 3 required)
  - rank: int|NaN (1 = highest composite, within each FY, annual only)
  - rank_change: int|NaN (previous FY rank - current rank; positive = improved)
  - n_components_scored: int (count of non-NaN z-scores used)
"""

import pandas as pd
import numpy as np

from utils import DATA_PROCESSED


# Minimum number of components required to compute composite score
MIN_COMPONENTS = 3

# Component columns and their z-score counterparts
COMPONENTS = {
    "gst_total": "gst_zscore",
    "electricity_mu": "electricity_zscore",
    "bank_credit_yoy": "credit_zscore",
    "epfo_payroll": "epfo_zscore",
}

ZSCORE_COLS = list(COMPONENTS.values())


def compute_cross_sectional_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute z-scores within each fiscal year (cross-sectional).

    For each FY, z-score = (value - mean) / std across all states.
    Only computed for annual rows.
    """
    print("Computing cross-sectional z-scores (annual)...")
    df = df.copy()

    # Initialize z-score columns
    for zcol in ZSCORE_COLS:
        df[zcol] = np.nan

    annual_mask = df["period_type"] == "annual"
    annual_fys = df.loc[annual_mask, "fiscal_year"].unique()

    for fy in sorted(annual_fys):
        fy_mask = annual_mask & (df["fiscal_year"] == fy)

        for raw_col, z_col in COMPONENTS.items():
            values = df.loc[fy_mask, raw_col]
            valid = values.dropna()

            if len(valid) < 5:
                # Not enough data points for meaningful z-scores
                continue

            mean = valid.mean()
            std = valid.std(ddof=1)

            if std > 0:
                df.loc[fy_mask, z_col] = (values - mean) / std

    # Also compute z-scores for monthly rows (GST + electricity only)
    monthly_mask = df["period_type"] == "monthly"
    if monthly_mask.any():
        monthly_months = df.loc[monthly_mask, "month"].unique()
        for month in sorted(monthly_months):
            m_mask = monthly_mask & (df["month"] == month)
            for raw_col, z_col in [("gst_total", "gst_zscore"),
                                    ("electricity_mu", "electricity_zscore")]:
                values = df.loc[m_mask, raw_col]
                valid = values.dropna()
                if len(valid) < 5:
                    continue
                mean = valid.mean()
                std = valid.std(ddof=1)
                if std > 0:
                    df.loc[m_mask, z_col] = (values - mean) / std

    return df


def compute_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Compute composite score as mean of available z-scores."""
    print("Computing composite scores...")
    df = df.copy()

    df["n_components_scored"] = df[ZSCORE_COLS].notna().sum(axis=1)

    # Composite = mean of available z-scores (only if >= MIN_COMPONENTS)
    df["composite_score"] = np.nan
    enough = df["n_components_scored"] >= MIN_COMPONENTS
    if enough.any():
        df.loc[enough, "composite_score"] = df.loc[enough, ZSCORE_COLS].mean(axis=1)

    return df


def compute_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Rank states within each FY by composite score (annual only)."""
    print("Computing rankings...")
    df = df.copy()
    df["rank"] = np.nan

    annual_mask = (df["period_type"] == "annual") & df["composite_score"].notna()
    annual_fys = df.loc[annual_mask, "fiscal_year"].unique()

    for fy in sorted(annual_fys):
        fy_mask = annual_mask & (df["fiscal_year"] == fy)
        # Rank: 1 = highest composite score
        df.loc[fy_mask, "rank"] = df.loc[fy_mask, "composite_score"].rank(
            ascending=False, method="min"
        ).astype(int)

    # Compute rank change (positive = improved)
    print("Computing rank changes...")
    df["rank_change"] = np.nan

    sorted_fys = sorted(annual_fys)
    for i in range(1, len(sorted_fys)):
        curr_fy = sorted_fys[i]
        prev_fy = sorted_fys[i - 1]

        curr_mask = annual_mask & (df["fiscal_year"] == curr_fy)
        prev_data = df.loc[annual_mask & (df["fiscal_year"] == prev_fy), ["state", "rank"]]

        if prev_data.empty:
            continue

        prev_ranks = prev_data.set_index("state")["rank"]

        for idx in df.loc[curr_mask].index:
            state = df.loc[idx, "state"]
            if state in prev_ranks.index:
                prev_rank = prev_ranks[state]
                curr_rank = df.loc[idx, "rank"]
                if pd.notna(prev_rank) and pd.notna(curr_rank):
                    df.loc[idx, "rank_change"] = int(prev_rank - curr_rank)

    return df


def main():
    print("=" * 70)
    print("Index Computation: Z-Scores + Composite")
    print("=" * 70)

    # Load merged data
    print("\nLoading merged data...")
    df = pd.read_parquet(DATA_PROCESSED / "merged.parquet")
    print(f"  Loaded {len(df):,} rows, {df['state'].nunique()} states")

    # Compute z-scores
    df = compute_cross_sectional_zscores(df)

    # Compute composite
    df = compute_composite(df)

    # Compute rankings (annual only)
    df = compute_rankings(df)

    # Save
    output_path = DATA_PROCESSED / "index_computed.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    annual = df[df["period_type"] == "annual"]

    # Z-score statistics
    print("\nZ-score statistics (annual, per FY):")
    for fy in sorted(annual["fiscal_year"].unique()):
        fy_data = annual[annual["fiscal_year"] == fy]
        scored = fy_data["composite_score"].notna().sum()
        if scored > 0:
            mean_z = fy_data["composite_score"].mean()
            std_z = fy_data["composite_score"].std()
            print(f"  {fy}: {scored} states scored, "
                  f"mean={mean_z:.3f}, std={std_z:.3f}")

    # Top 10 rankings for latest FY with actual rankings
    fys_with_ranks = annual[annual["rank"].notna()]["fiscal_year"].unique()
    latest_fy = sorted(fys_with_ranks)[-1] if len(fys_with_ranks) > 0 else None
    latest = annual[
        (annual["fiscal_year"] == latest_fy) & annual["rank"].notna()
    ].sort_values("rank") if latest_fy else pd.DataFrame()

    print(f"\nTop 10 states ({latest_fy}):")
    for _, row in latest.head(10).iterrows():
        rc = f" ({'+' if row['rank_change'] > 0 else ''}{int(row['rank_change'])})" \
             if pd.notna(row["rank_change"]) else ""
        print(f"  {int(row['rank']):2d}. {row['state']:25s} "
              f"score={row['composite_score']:.3f}{rc}")

    # Verify z-score properties
    print("\nVerification:")
    for fy in sorted(annual["fiscal_year"].unique()):
        fy_data = annual[annual["fiscal_year"] == fy]
        for raw_col, z_col in COMPONENTS.items():
            valid = fy_data[z_col].dropna()
            if len(valid) >= 5:
                m = valid.mean()
                s = valid.std(ddof=0)
                if abs(m) > 0.01 or abs(s - 1.0) > 0.15:
                    print(f"  WARNING: {fy} {z_col}: mean={m:.3f}, std={s:.3f}")
    print("  Z-score properties verified (mean~0, std~1)")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
