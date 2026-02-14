"""
10_compute_electricity_analysis.py
===================================
Compute electricity deep-dive metrics for the Li Keqiang Index dashboard.

Metrics computed:
  1. Electricity intensity (MU per crore GSDP) — annual
  2. National share of electricity by state — annual
  3. Monthly seasonality index (month / FY mean * 100) — monthly
  4. YoY monthly growth (same month previous year) — monthly
  5. Electricity-GSDP elasticity (log-log OLS per state) — panel
  6. Bivariate residuals (cross-sectional GSDP ~ electricity) — latest FY
  7. Monthly volatility (coefficient of variation within FY) — annual

Inputs:
  - data/processed/index_computed.parquet
  - data/processed/gsdp_clean.parquet
  - data/processed/electricity_clean.parquet

Output:
  - public/data/electricity.json
"""

import json
import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

from utils import (
    DATA_PROCESSED,
    PUBLIC_DATA,
    load_state_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nan_to_none(obj):
    """Recursively convert NaN/NaT to None for JSON serialization."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return round(float(obj), 4) if not np.isnan(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj


def write_json(data, path):
    """Write data to JSON file with NaN handling."""
    clean = nan_to_none(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    size_kb = path.stat().st_size / 1024
    print(f"  {path.name}: {size_kb:.1f} KB")


# ---------------------------------------------------------------------------
# 1. Electricity Intensity
# ---------------------------------------------------------------------------

def compute_intensity(annual_elec: pd.DataFrame, gsdp: pd.DataFrame) -> pd.DataFrame:
    """Compute electricity intensity = electricity_mu / gsdp_current_crore."""
    print("\n1. Computing electricity intensity (MU per crore GSDP)...")

    merged = annual_elec.merge(
        gsdp[["state", "fiscal_year", "gsdp_current_crore"]],
        on=["state", "fiscal_year"],
        how="inner",
    )

    valid = merged[
        merged["electricity_mu"].notna() & merged["gsdp_current_crore"].notna() &
        (merged["gsdp_current_crore"] > 0)
    ].copy()

    valid["intensity_mu_per_crore"] = valid["electricity_mu"] / valid["gsdp_current_crore"]

    print(f"  Computed intensity for {valid['state'].nunique()} states x "
          f"{valid['fiscal_year'].nunique()} FYs ({len(valid)} rows)")

    return valid


# ---------------------------------------------------------------------------
# 2. National Share
# ---------------------------------------------------------------------------

def compute_national_share(annual_elec: pd.DataFrame) -> pd.DataFrame:
    """Compute each state's share of total national electricity per FY."""
    print("\n2. Computing national electricity share...")

    valid = annual_elec[annual_elec["electricity_mu"].notna()].copy()
    fy_totals = valid.groupby("fiscal_year")["electricity_mu"].sum().rename("national_total_mu")
    valid = valid.merge(fy_totals, on="fiscal_year", how="left")
    valid["national_share_pct"] = valid["electricity_mu"] / valid["national_total_mu"] * 100

    print(f"  Computed shares for {valid['state'].nunique()} states x "
          f"{valid['fiscal_year'].nunique()} FYs")

    return valid


# ---------------------------------------------------------------------------
# 3. Monthly Seasonality
# ---------------------------------------------------------------------------

def compute_seasonality(monthly_elec: pd.DataFrame) -> pd.DataFrame:
    """Compute seasonality index: month_value / FY_mean * 100."""
    print("\n3. Computing monthly seasonality index...")

    valid = monthly_elec[monthly_elec["electricity_mu"].notna()].copy()

    # Extract fiscal year from month (YYYY-MM)
    def month_to_fy(yyyy_mm):
        parts = yyyy_mm.split("-")
        year, month = int(parts[0]), int(parts[1])
        if month >= 4:
            return f"{year}-{str(year + 1)[-2:]}"
        else:
            return f"{year - 1}-{str(year)[-2:]}"

    valid["fiscal_year_calc"] = valid["month"].apply(month_to_fy)

    # Compute FY mean per state
    fy_means = (
        valid.groupby(["state", "fiscal_year_calc"])["electricity_mu"]
        .mean()
        .rename("fy_mean_mu")
    )
    valid = valid.merge(fy_means, on=["state", "fiscal_year_calc"], how="left")

    valid["seasonality_index"] = np.where(
        valid["fy_mean_mu"] > 0,
        valid["electricity_mu"] / valid["fy_mean_mu"] * 100,
        np.nan,
    )

    # Extract calendar month number for aggregation
    valid["month_num"] = valid["month"].str.split("-").str[1].astype(int)

    print(f"  Computed seasonality for {valid['state'].nunique()} states, "
          f"{valid['month'].nunique()} months")

    return valid


# ---------------------------------------------------------------------------
# 4. YoY Monthly Growth
# ---------------------------------------------------------------------------

def compute_yoy_monthly_growth(monthly_elec: pd.DataFrame) -> pd.DataFrame:
    """Compute YoY growth: elec_month[t] / elec_month[t-12] - 1."""
    print("\n4. Computing YoY monthly growth...")

    valid = monthly_elec[monthly_elec["electricity_mu"].notna()].copy()
    valid = valid.sort_values(["state", "month"]).reset_index(drop=True)

    # Build lookup: state + month -> electricity_mu
    lookup = valid.set_index(["state", "month"])["electricity_mu"].to_dict()

    def get_prev_year_month(yyyy_mm):
        parts = yyyy_mm.split("-")
        year, month = int(parts[0]), int(parts[1])
        return f"{year - 1}-{month:02d}"

    yoy_records = []
    for _, row in valid.iterrows():
        prev_month = get_prev_year_month(row["month"])
        prev_val = lookup.get((row["state"], prev_month))
        if prev_val is not None and prev_val > 0:
            yoy_pct = (row["electricity_mu"] / prev_val - 1) * 100
            yoy_records.append({
                "state": row["state"],
                "month": row["month"],
                "electricity_mu": row["electricity_mu"],
                "yoy_pct": yoy_pct,
            })

    df_yoy = pd.DataFrame(yoy_records)
    print(f"  Computed YoY growth for {df_yoy['state'].nunique()} states, "
          f"{len(df_yoy)} month-state pairs")

    return df_yoy


# ---------------------------------------------------------------------------
# 5. Electricity-GSDP Elasticity (per-state log-log)
# ---------------------------------------------------------------------------

def compute_elasticity(annual_elec: pd.DataFrame, gsdp: pd.DataFrame) -> pd.DataFrame:
    """Per-state log-log regression: log(GSDP) ~ log(electricity).

    Requires statsmodels. Returns elasticity coefficient per state.
    Classification: >1 = industrial, 0.5-1 = services-transitioning, <0.5 = low-intensity.
    """
    print("\n5. Computing electricity-GSDP elasticity (log-log per state)...")

    try:
        import statsmodels.api as sm
    except ImportError:
        print("  WARNING: statsmodels not installed. Elasticity computation skipped.")
        return pd.DataFrame(columns=["state", "elasticity", "elasticity_label", "n_obs", "r_squared"])

    merged = annual_elec.merge(
        gsdp[["state", "fiscal_year", "gsdp_current_crore"]],
        on=["state", "fiscal_year"],
        how="inner",
    )

    valid = merged[
        merged["electricity_mu"].notna() &
        merged["gsdp_current_crore"].notna() &
        (merged["electricity_mu"] > 0) &
        (merged["gsdp_current_crore"] > 0)
    ].copy()

    results = []
    for state in sorted(valid["state"].unique()):
        state_data = valid[valid["state"] == state]
        if len(state_data) < 4:
            continue

        log_elec = np.log(state_data["electricity_mu"].values)
        log_gsdp = np.log(state_data["gsdp_current_crore"].values)

        X = sm.add_constant(log_elec)
        try:
            model = sm.OLS(log_gsdp, X).fit()
            elasticity = float(model.params[1])
            r_sq = float(model.rsquared)
        except Exception:
            continue

        if elasticity > 1:
            label = "industrial"
        elif elasticity >= 0.5:
            label = "services-transitioning"
        else:
            label = "low-intensity"

        results.append({
            "state": state,
            "elasticity": round(elasticity, 4),
            "elasticity_label": label,
            "n_obs": len(state_data),
            "r_squared": round(r_sq, 4),
        })

    df_elas = pd.DataFrame(results)
    print(f"  Computed elasticity for {len(df_elas)} states (>= 4 data points each)")
    for label in ["industrial", "services-transitioning", "low-intensity"]:
        n = (df_elas["elasticity_label"] == label).sum()
        if n > 0:
            print(f"    {label}: {n} states")

    return df_elas


# ---------------------------------------------------------------------------
# 6. Bivariate Residuals (cross-sectional)
# ---------------------------------------------------------------------------

def compute_bivariate_residuals(annual_elec: pd.DataFrame, gsdp: pd.DataFrame) -> tuple:
    """Cross-sectional OLS: GSDP ~ electricity for latest FY.

    Returns (residuals_df, regression_summary_dict).
    Positive residual = GSDP higher than electricity alone predicts.
    """
    print("\n6. Computing bivariate residuals (GSDP ~ electricity, cross-sectional)...")

    try:
        import statsmodels.api as sm
    except ImportError:
        print("  WARNING: statsmodels not installed. Bivariate residuals skipped.")
        return pd.DataFrame(), {}

    merged = annual_elec.merge(
        gsdp[["state", "fiscal_year", "gsdp_current_crore"]],
        on=["state", "fiscal_year"],
        how="inner",
    )

    valid = merged[
        merged["electricity_mu"].notna() &
        merged["gsdp_current_crore"].notna() &
        (merged["electricity_mu"] > 0) &
        (merged["gsdp_current_crore"] > 0)
    ].copy()

    if valid.empty:
        print("  WARNING: No overlapping data for bivariate regression.")
        return pd.DataFrame(), {}

    # Use latest FY with enough data
    fy_counts = valid.groupby("fiscal_year")["state"].nunique()
    eligible_fys = fy_counts[fy_counts >= 10].index.tolist()
    if not eligible_fys:
        print("  WARNING: No FY with >= 10 states for bivariate regression.")
        return pd.DataFrame(), {}

    latest_fy = sorted(eligible_fys)[-1]
    fy_data = valid[valid["fiscal_year"] == latest_fy].copy()
    n = len(fy_data)
    print(f"  Using FY {latest_fy}: {n} states")

    y = fy_data["gsdp_current_crore"].values
    X = sm.add_constant(fy_data["electricity_mu"].values)

    model = sm.OLS(y, X).fit()
    fy_data["predicted_gsdp"] = model.predict(X)
    fy_data["residual_crore"] = fy_data["gsdp_current_crore"] - fy_data["predicted_gsdp"]
    fy_data["residual_label"] = fy_data["residual_crore"].apply(
        lambda r: "GSDP higher than electricity predicts" if r > 0
        else "GSDP lower than electricity predicts"
    )

    regression_summary = {
        "r_squared": round(float(model.rsquared), 4),
        "n": n,
        "coef": round(float(model.params[1]), 4),
        "intercept": round(float(model.params[0]), 4),
        "latest_fy": latest_fy,
    }

    print(f"  R-squared: {model.rsquared:.4f}, coef: {model.params[1]:.4f}, "
          f"intercept: {model.params[0]:.4f}")

    return fy_data, regression_summary


# ---------------------------------------------------------------------------
# 7. Monthly Volatility (CoV within FY)
# ---------------------------------------------------------------------------

def compute_volatility(monthly_elec: pd.DataFrame) -> pd.DataFrame:
    """Coefficient of variation (std/mean) of monthly electricity within each FY per state."""
    print("\n7. Computing monthly volatility (CoV within FY)...")

    valid = monthly_elec[monthly_elec["electricity_mu"].notna()].copy()

    # Compute fiscal year from month
    def month_to_fy(yyyy_mm):
        parts = yyyy_mm.split("-")
        year, month = int(parts[0]), int(parts[1])
        if month >= 4:
            return f"{year}-{str(year + 1)[-2:]}"
        else:
            return f"{year - 1}-{str(year)[-2:]}"

    valid["fiscal_year_calc"] = valid["month"].apply(month_to_fy)

    # Need at least 6 months in a FY for meaningful CoV
    grouped = valid.groupby(["state", "fiscal_year_calc"])["electricity_mu"]
    stats_df = grouped.agg(["mean", "std", "count"]).reset_index()
    stats_df.columns = ["state", "fiscal_year", "mean_mu", "std_mu", "month_count"]

    stats_df = stats_df[stats_df["month_count"] >= 6].copy()
    stats_df["volatility_cov"] = np.where(
        stats_df["mean_mu"] > 0,
        stats_df["std_mu"] / stats_df["mean_mu"],
        np.nan,
    )

    print(f"  Computed volatility for {stats_df['state'].nunique()} states x "
          f"{stats_df['fiscal_year'].nunique()} FYs")

    return stats_df


# ---------------------------------------------------------------------------
# National Summary
# ---------------------------------------------------------------------------

def compute_national_summary(
    annual_elec: pd.DataFrame,
    gsdp: pd.DataFrame,
) -> dict:
    """Compute national-level summary statistics."""
    print("\nComputing national summary...")

    valid = annual_elec[annual_elec["electricity_mu"].notna()].copy()
    fy_totals = valid.groupby("fiscal_year")["electricity_mu"].sum().sort_index()

    if len(fy_totals) == 0:
        return {}

    latest_fy = fy_totals.index[-1]
    total_mu = float(fy_totals.iloc[-1])

    # YoY growth
    yoy_growth = None
    if len(fy_totals) >= 2:
        prev_total = float(fy_totals.iloc[-2])
        if prev_total > 0:
            yoy_growth = round((total_mu / prev_total - 1) * 100, 2)

    # GSDP correlation (latest FY cross-section)
    gsdp_corr_r = None
    merged = annual_elec.merge(
        gsdp[["state", "fiscal_year", "gsdp_current_crore"]],
        on=["state", "fiscal_year"],
        how="inner",
    )
    # Find latest FY with enough matched states
    merged_valid = merged[
        merged["electricity_mu"].notna() & merged["gsdp_current_crore"].notna()
    ]
    if not merged_valid.empty:
        fy_counts = merged_valid.groupby("fiscal_year")["state"].nunique()
        eligible = fy_counts[fy_counts >= 10].index.tolist()
        if eligible:
            corr_fy = sorted(eligible)[-1]
            corr_data = merged_valid[merged_valid["fiscal_year"] == corr_fy]
            r, _ = stats.pearsonr(
                corr_data["electricity_mu"], corr_data["gsdp_current_crore"]
            )
            gsdp_corr_r = round(float(r), 4)

    summary = {
        "total_mu_latest_fy": round(total_mu, 2),
        "yoy_growth_pct": yoy_growth,
        "gsdp_correlation_r": gsdp_corr_r,
        "latest_fy": latest_fy,
    }

    print(f"  Latest FY: {latest_fy}, Total MU: {total_mu:,.0f}")
    if yoy_growth is not None:
        print(f"  YoY growth: {yoy_growth:.1f}%")
    if gsdp_corr_r is not None:
        print(f"  GSDP correlation (Pearson r): {gsdp_corr_r:.4f}")

    return summary


# ---------------------------------------------------------------------------
# Assemble Output
# ---------------------------------------------------------------------------

def assemble_output(
    slug_map: dict,
    national_summary: dict,
    intensity_df: pd.DataFrame,
    share_df: pd.DataFrame,
    seasonality_df: pd.DataFrame,
    yoy_df: pd.DataFrame,
    elasticity_df: pd.DataFrame,
    residuals_df: pd.DataFrame,
    regression_summary: dict,
    volatility_df: pd.DataFrame,
    annual_elec: pd.DataFrame,
    gsdp: pd.DataFrame,
) -> dict:
    """Assemble the final JSON output structure."""
    print("\nAssembling output JSON...")

    # Reverse slug map: state -> slug
    state_to_slug = {v: k for k, v in {s: st for st, s in slug_map.items()}.items()}
    # Actually, slug_map is state_name -> slug
    state_to_slug = slug_map

    # Determine top 15 states by electricity_mu (latest FY)
    latest_fys = sorted(annual_elec[annual_elec["electricity_mu"].notna()]["fiscal_year"].unique())
    latest_fy = latest_fys[-1] if latest_fys else None

    top_states = []
    if latest_fy:
        latest_annual = annual_elec[
            (annual_elec["fiscal_year"] == latest_fy) & annual_elec["electricity_mu"].notna()
        ].nlargest(15, "electricity_mu")
        top_states = latest_annual["state"].tolist()

    # Determine latest 24 months
    all_months = sorted(yoy_df["month"].unique()) if not yoy_df.empty else []
    latest_24_months = all_months[-24:] if len(all_months) > 24 else all_months

    # --- State Profiles ---
    all_states = set()
    all_states.update(annual_elec["state"].unique())
    if not yoy_df.empty:
        all_states.update(yoy_df["state"].unique())

    # Latest FY intensity per state
    intensity_latest = {}
    if not intensity_df.empty and latest_fy:
        # Find latest FY in intensity (may not match if GSDP lags)
        intensity_fys = sorted(intensity_df["fiscal_year"].unique())
        int_fy = intensity_fys[-1] if intensity_fys else None
        if int_fy:
            for _, row in intensity_df[intensity_df["fiscal_year"] == int_fy].iterrows():
                intensity_latest[row["state"]] = round(float(row["intensity_mu_per_crore"]), 4)

    # Latest FY national share
    share_latest = {}
    if not share_df.empty and latest_fy:
        for _, row in share_df[share_df["fiscal_year"] == latest_fy].iterrows():
            share_latest[row["state"]] = round(float(row["national_share_pct"]), 4)

    # Elasticity lookup
    elasticity_lookup = {}
    if not elasticity_df.empty:
        for _, row in elasticity_df.iterrows():
            elasticity_lookup[row["state"]] = {
                "elasticity": row["elasticity"],
                "elasticity_label": row["elasticity_label"],
            }

    # Residual lookup
    residual_lookup = {}
    if not residuals_df.empty:
        for _, row in residuals_df.iterrows():
            residual_lookup[row["state"]] = {
                "residual_crore": round(float(row["residual_crore"]), 2),
                "residual_label": row["residual_label"],
            }

    # Volatility lookup (latest FY)
    volatility_lookup = {}
    if not volatility_df.empty:
        vol_fys = sorted(volatility_df["fiscal_year"].unique())
        vol_latest_fy = vol_fys[-1] if vol_fys else None
        if vol_latest_fy:
            for _, row in volatility_df[volatility_df["fiscal_year"] == vol_latest_fy].iterrows():
                if pd.notna(row["volatility_cov"]):
                    volatility_lookup[row["state"]] = round(float(row["volatility_cov"]), 4)

    state_profiles = {}
    for state in sorted(all_states):
        slug = state_to_slug.get(state)
        if not slug:
            continue

        profile = {}

        # Intensity
        if state in intensity_latest:
            profile["intensity_mu_per_crore"] = intensity_latest[state]

        # National share
        if state in share_latest:
            profile["national_share_pct"] = share_latest[state]

        # Elasticity
        if state in elasticity_lookup:
            profile["elasticity"] = elasticity_lookup[state]["elasticity"]
            profile["elasticity_label"] = elasticity_lookup[state]["elasticity_label"]

        # Residual
        if state in residual_lookup:
            profile["bivariate_residual_crore"] = residual_lookup[state]["residual_crore"]
            profile["residual_label"] = residual_lookup[state]["residual_label"]

        # Volatility
        if state in volatility_lookup:
            profile["volatility_cov"] = volatility_lookup[state]

        # Monthly growth (limited to top 15 states x latest 24 months)
        if state in top_states and not yoy_df.empty:
            state_yoy = yoy_df[
                (yoy_df["state"] == state) & (yoy_df["month"].isin(latest_24_months))
            ].sort_values("month")
            if not state_yoy.empty:
                profile["monthly_growth"] = [
                    {"month": row["month"], "yoy_pct": round(float(row["yoy_pct"]), 2)}
                    for _, row in state_yoy.iterrows()
                ]

        # Seasonality index (limited to top 15 states, averaged across FYs by month_num)
        if state in top_states and not seasonality_df.empty:
            state_season = seasonality_df[seasonality_df["state"] == state]
            if not state_season.empty:
                avg_season = (
                    state_season.groupby("month_num")["seasonality_index"]
                    .mean()
                    .reset_index()
                    .sort_values("month_num")
                )
                profile["seasonality_index"] = [
                    {"month": int(row["month_num"]), "index": round(float(row["seasonality_index"]), 2)}
                    for _, row in avg_season.iterrows()
                ]

        if profile:
            state_profiles[slug] = profile

    # --- Rankings by Intensity ---
    rankings_intensity = []
    if not intensity_df.empty:
        int_fys = sorted(intensity_df["fiscal_year"].unique())
        int_latest_fy = int_fys[-1] if int_fys else None
        if int_latest_fy:
            int_latest = intensity_df[intensity_df["fiscal_year"] == int_latest_fy].copy()
            int_latest = int_latest.sort_values("intensity_mu_per_crore", ascending=False)
            for _, row in int_latest.iterrows():
                slug = state_to_slug.get(row["state"])
                if slug:
                    rankings_intensity.append({
                        "state": row["state"],
                        "slug": slug,
                        "intensity": round(float(row["intensity_mu_per_crore"]), 4),
                        "gsdp_crore": round(float(row["gsdp_current_crore"]), 2),
                        "electricity_mu": round(float(row["electricity_mu"]), 2),
                    })

    # --- Rankings by Elasticity ---
    rankings_elasticity = []
    if not elasticity_df.empty:
        for _, row in elasticity_df.sort_values("elasticity", ascending=False).iterrows():
            slug = state_to_slug.get(row["state"])
            if slug:
                rankings_elasticity.append({
                    "state": row["state"],
                    "slug": slug,
                    "elasticity": row["elasticity"],
                    "label": row["elasticity_label"],
                })

    # --- Scatter: Electricity vs GSDP ---
    scatter_data = []
    if not residuals_df.empty:
        for _, row in residuals_df.iterrows():
            slug = state_to_slug.get(row["state"])
            elast_info = elasticity_lookup.get(row["state"], {})
            if slug:
                scatter_data.append({
                    "state": row["state"],
                    "slug": slug,
                    "electricity_mu": round(float(row["electricity_mu"]), 2),
                    "gsdp_crore": round(float(row["gsdp_current_crore"]), 2),
                    "elasticity_label": elast_info.get("elasticity_label"),
                    "residual_crore": round(float(row["residual_crore"]), 2),
                })

    # --- Final output ---
    output = {
        "generated_at": datetime.now().isoformat(),
        "national_summary": national_summary,
        "state_profiles": state_profiles,
        "rankings_by_intensity": rankings_intensity,
        "rankings_by_elasticity": rankings_elasticity,
        "electricity_vs_gsdp_scatter": scatter_data,
        "electricity_vs_gsdp_regression": regression_summary,
    }

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 70)
    print("Electricity Deep-Dive Analysis")
    print("=" * 70)

    # --- Load data ---
    print("\nLoading input data...")

    # Index computed (has annual + monthly rows with state, slug, fiscal_year, etc.)
    index_path = DATA_PROCESSED / "index_computed.parquet"
    if not index_path.exists():
        print(f"ERROR: {index_path} not found. Run 06_compute_index.py first.")
        sys.exit(1)
    df_index = pd.read_parquet(index_path)
    print(f"  index_computed.parquet: {len(df_index):,} rows, "
          f"{df_index['state'].nunique()} states")

    # GSDP data
    gsdp_path = DATA_PROCESSED / "gsdp_clean.parquet"
    if not gsdp_path.exists():
        print(f"ERROR: {gsdp_path} not found. Run 08_fetch_gsdp.py first.")
        sys.exit(1)
    df_gsdp = pd.read_parquet(gsdp_path)
    print(f"  gsdp_clean.parquet: {len(df_gsdp):,} rows, "
          f"{df_gsdp['state'].nunique()} states")

    # Electricity clean (monthly)
    elec_path = DATA_PROCESSED / "electricity_clean.parquet"
    if not elec_path.exists():
        print(f"ERROR: {elec_path} not found. Run 02_fetch_electricity.py first.")
        sys.exit(1)
    df_elec = pd.read_parquet(elec_path)
    print(f"  electricity_clean.parquet: {len(df_elec):,} rows, "
          f"{df_elec['state'].nunique()} states")

    # --- Load state metadata for slug mapping ---
    meta = load_state_metadata()
    slug_map = {row["canonical_name"]: row["slug"] for _, row in meta.iterrows()}
    print(f"  State metadata: {len(slug_map)} states")

    # --- Split index into annual and monthly ---
    annual_from_index = df_index[df_index["period_type"] == "annual"][
        ["state", "slug", "fiscal_year", "electricity_mu"]
    ].copy()
    monthly_from_index = df_index[df_index["period_type"] == "monthly"][
        ["state", "slug", "month", "electricity_mu"]
    ].copy()

    # Also use electricity_clean for monthly data (may have better coverage)
    # electricity_clean has: state, fiscal_year, month (YYYY-MM), electricity_mu, days_with_data
    monthly_elec = df_elec[["state", "month", "electricity_mu"]].copy() if "month" in df_elec.columns else monthly_from_index.copy()

    # Use index annual data for annual metrics
    annual_elec = annual_from_index.copy()

    print(f"\n  Annual electricity rows: {len(annual_elec):,}")
    print(f"  Monthly electricity rows: {len(monthly_elec):,}")

    # --- Compute all metrics ---

    # 1. Intensity
    intensity_df = compute_intensity(annual_elec, df_gsdp)

    # 2. National share
    share_df = compute_national_share(annual_elec)

    # 3. Seasonality
    seasonality_df = compute_seasonality(monthly_elec)

    # 4. YoY monthly growth
    yoy_df = compute_yoy_monthly_growth(monthly_elec)

    # 5. Elasticity
    elasticity_df = compute_elasticity(annual_elec, df_gsdp)

    # 6. Bivariate residuals
    residuals_df, regression_summary = compute_bivariate_residuals(annual_elec, df_gsdp)

    # 7. Volatility
    volatility_df = compute_volatility(monthly_elec)

    # National summary
    national_summary = compute_national_summary(annual_elec, df_gsdp)

    # --- Assemble and write output ---
    output = assemble_output(
        slug_map=slug_map,
        national_summary=national_summary,
        intensity_df=intensity_df,
        share_df=share_df,
        seasonality_df=seasonality_df,
        yoy_df=yoy_df,
        elasticity_df=elasticity_df,
        residuals_df=residuals_df,
        regression_summary=regression_summary,
        volatility_df=volatility_df,
        annual_elec=annual_elec,
        gsdp=df_gsdp,
    )

    output_path = PUBLIC_DATA / "electricity.json"
    write_json(output, output_path)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  State profiles: {len(output['state_profiles'])}")
    print(f"  Rankings by intensity: {len(output['rankings_by_intensity'])}")
    print(f"  Rankings by elasticity: {len(output['rankings_by_elasticity'])}")
    print(f"  Scatter data points: {len(output['electricity_vs_gsdp_scatter'])}")
    if output["electricity_vs_gsdp_regression"]:
        reg = output["electricity_vs_gsdp_regression"]
        print(f"  Regression R2: {reg.get('r_squared')}, N: {reg.get('n')}, "
              f"FY: {reg.get('latest_fy')}")

    # Count profiles with each metric
    has_intensity = sum(1 for p in output["state_profiles"].values() if "intensity_mu_per_crore" in p)
    has_elasticity = sum(1 for p in output["state_profiles"].values() if "elasticity" in p)
    has_growth = sum(1 for p in output["state_profiles"].values() if "monthly_growth" in p)
    has_season = sum(1 for p in output["state_profiles"].values() if "seasonality_index" in p)
    has_vol = sum(1 for p in output["state_profiles"].values() if "volatility_cov" in p)

    print(f"\n  Profiles with intensity: {has_intensity}")
    print(f"  Profiles with elasticity: {has_elasticity}")
    print(f"  Profiles with monthly growth: {has_growth}")
    print(f"  Profiles with seasonality: {has_season}")
    print(f"  Profiles with volatility: {has_vol}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
