"""
09_compute_insights.py
======================
Compute growth metrics, momentum tiers, COVID recovery, GSDP comparison,
component diagnostics, auto-generated insight text, correlations,
OLS regression with full diagnostics, panel fixed-effects regression,
log-log elasticities, lagged panel correlation, PCA weights robustness,
state gap explanations, and regional analysis.

Inputs:
  - data/processed/index_computed.parquet
  - data/processed/gsdp_clean.parquet
  - data/reference/brap_categories.csv

Outputs:
  - data/processed/insights_computed.parquet
  - public/data/regression.json
  - public/data/insights.json
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
import sys
import io

from utils import (
    DATA_PROCESSED,
    DATA_REF,
    PUBLIC_DATA,
    load_state_metadata,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Momentum tier thresholds (based on rank_momentum_3yr)
MOMENTUM_RISING_THRESHOLD = 3    # >= +3 positions improved
MOMENTUM_DECLINING_THRESHOLD = -3  # <= -3 positions dropped

# COVID fiscal years
COVID_PRE_FY = "2019-20"
COVID_DIP_FY = "2020-21"

# Minimum states for correlation/regression
MIN_STATES_FOR_CORR = 5
MIN_STATES_FOR_REGRESSION = 15

# Component z-score columns and labels
ZSCORE_COLS = {
    "gst_zscore": "GST collections (tax compliance and economic transactions)",
    "electricity_zscore": "electricity demand (physical/industrial activity)",
    "credit_zscore": "bank credit growth (financial depth and investment)",
    "epfo_zscore": "formal employment (EPFO payroll additions)",
}

ZSCORE_SHORT = {
    "gst_zscore": "GST",
    "electricity_zscore": "Electricity",
    "credit_zscore": "Credit",
    "epfo_zscore": "EPFO",
}

# Raw component columns for growth computation
RAW_COMPONENTS = ["gst_total", "electricity_mu", "bank_credit_yoy", "epfo_payroll"]


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
# Growth Metrics (computed from RAW values, never z-scores)
# ---------------------------------------------------------------------------

def compute_growth_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute YoY growth rates for each component from raw values."""
    print("Computing growth metrics from raw values...")
    df = df.copy()

    sorted_fys = sorted(df["fiscal_year"].unique())
    fy_to_idx = {fy: i for i, fy in enumerate(sorted_fys)}

    for comp in RAW_COMPONENTS:
        growth_col = comp.replace("gst_total", "gst") \
                         .replace("electricity_mu", "elec") \
                         .replace("bank_credit_yoy", "credit") \
                         .replace("epfo_payroll", "epfo") + "_yoy_pct"
        df[growth_col] = np.nan

        for state in df["state"].unique():
            state_mask = df["state"] == state
            state_data = df[state_mask].sort_values("fiscal_year")

            for _, row in state_data.iterrows():
                fy = row["fiscal_year"]
                idx = fy_to_idx.get(fy)
                if idx is None or idx == 0:
                    continue
                prev_fy = sorted_fys[idx - 1]
                prev_row = state_data[state_data["fiscal_year"] == prev_fy]
                if prev_row.empty:
                    continue
                curr_val = row[comp]
                prev_val = prev_row.iloc[0][comp]
                if pd.notna(curr_val) and pd.notna(prev_val) and prev_val != 0:
                    growth = (curr_val / prev_val - 1) * 100
                    df.loc[row.name, growth_col] = growth

    # Composite score change (absolute, not %)
    df["composite_yoy_change"] = np.nan
    for state in df["state"].unique():
        state_mask = df["state"] == state
        state_data = df[state_mask].sort_values("fiscal_year")
        for _, row in state_data.iterrows():
            fy = row["fiscal_year"]
            idx = fy_to_idx.get(fy)
            if idx is None or idx == 0:
                continue
            prev_fy = sorted_fys[idx - 1]
            prev_row = state_data[state_data["fiscal_year"] == prev_fy]
            if prev_row.empty:
                continue
            curr = row["composite_score"]
            prev = prev_row.iloc[0]["composite_score"]
            if pd.notna(curr) and pd.notna(prev):
                df.loc[row.name, "composite_yoy_change"] = curr - prev

    return df


# ---------------------------------------------------------------------------
# Momentum Tiers
# ---------------------------------------------------------------------------

def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 3-year rank momentum and tier classification."""
    print("Computing momentum tiers...")
    df = df.copy()
    sorted_fys = sorted(df["fiscal_year"].unique())
    fy_to_idx = {fy: i for i, fy in enumerate(sorted_fys)}

    df["rank_momentum_3yr"] = np.nan
    df["momentum_tier"] = None

    for state in df["state"].unique():
        state_mask = df["state"] == state
        state_data = df[state_mask].sort_values("fiscal_year")

        for _, row in state_data.iterrows():
            fy = row["fiscal_year"]
            idx = fy_to_idx.get(fy)
            if idx is None or idx < 3:
                continue
            prev_fy = sorted_fys[idx - 3]
            prev_row = state_data[state_data["fiscal_year"] == prev_fy]
            if prev_row.empty:
                continue
            curr_rank = row["rank"]
            prev_rank = prev_row.iloc[0]["rank"]
            if pd.notna(curr_rank) and pd.notna(prev_rank):
                momentum = int(prev_rank - curr_rank)  # positive = improved
                df.loc[row.name, "rank_momentum_3yr"] = momentum
                if momentum >= MOMENTUM_RISING_THRESHOLD:
                    df.loc[row.name, "momentum_tier"] = "rising"
                elif momentum <= MOMENTUM_DECLINING_THRESHOLD:
                    df.loc[row.name, "momentum_tier"] = "declining"
                else:
                    df.loc[row.name, "momentum_tier"] = "stable"

    return df


# ---------------------------------------------------------------------------
# COVID Recovery
# ---------------------------------------------------------------------------

def compute_covid_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """Compute COVID dip and recovery speed."""
    print("Computing COVID recovery metrics...")
    df = df.copy()
    df["covid_dip"] = np.nan
    df["recovery_fy"] = None
    df["recovery_speed"] = np.nan
    df["pre_covid_declining"] = False

    sorted_fys = sorted(df["fiscal_year"].unique())
    post_covid_fys = [fy for fy in sorted_fys if fy > COVID_DIP_FY]

    for state in df["state"].unique():
        state_data = df[df["state"] == state].sort_values("fiscal_year")

        pre_row = state_data[state_data["fiscal_year"] == COVID_PRE_FY]
        dip_row = state_data[state_data["fiscal_year"] == COVID_DIP_FY]

        if pre_row.empty or dip_row.empty:
            continue

        pre_score = pre_row.iloc[0]["composite_score"]
        dip_score = dip_row.iloc[0]["composite_score"]

        if pd.isna(pre_score) or pd.isna(dip_score):
            continue

        covid_dip = dip_score - pre_score

        # Apply to the latest FY row that has a composite score (not partial FYs)
        scored_rows = state_data[state_data["composite_score"].notna()]
        if scored_rows.empty:
            continue
        latest_idx = scored_rows.index[-1]
        df.loc[latest_idx, "covid_dip"] = covid_dip

        # Check if already declining pre-COVID
        pre_pre_fy = "2018-19"
        pre_pre_row = state_data[state_data["fiscal_year"] == pre_pre_fy]
        if not pre_pre_row.empty:
            pre_pre_score = pre_pre_row.iloc[0]["composite_score"]
            if pd.notna(pre_pre_score) and pre_score < pre_pre_score:
                df.loc[latest_idx, "pre_covid_declining"] = True

        # Find recovery FY
        for post_fy in post_covid_fys:
            post_row = state_data[state_data["fiscal_year"] == post_fy]
            if post_row.empty:
                continue
            post_score = post_row.iloc[0]["composite_score"]
            if pd.notna(post_score) and post_score >= pre_score:
                fy_idx_dip = sorted_fys.index(COVID_DIP_FY)
                fy_idx_recovery = sorted_fys.index(post_fy)
                speed = fy_idx_recovery - fy_idx_dip
                df.loc[latest_idx, "recovery_fy"] = post_fy
                df.loc[latest_idx, "recovery_speed"] = speed
                break

    return df


# ---------------------------------------------------------------------------
# GSDP Comparison
# ---------------------------------------------------------------------------

def merge_gsdp(df: pd.DataFrame) -> pd.DataFrame:
    """Merge GSDP data and compute rank gap."""
    print("Merging GSDP data...")
    gsdp_path = DATA_PROCESSED / "gsdp_clean.parquet"
    if not gsdp_path.exists():
        print("  WARNING: gsdp_clean.parquet not found. GSDP comparison disabled.")
        df["gsdp_rank"] = np.nan
        df["gsdp_current_crore"] = np.nan
        df["rank_gap"] = np.nan
        df["gap_label"] = None
        return df

    gsdp = pd.read_parquet(gsdp_path)
    print(f"  GSDP data: {len(gsdp)} rows, {gsdp['state'].nunique()} states")

    # Merge on state + fiscal_year
    df = df.merge(
        gsdp[["state", "fiscal_year", "gsdp_current_crore", "gsdp_rank"]],
        on=["state", "fiscal_year"],
        how="left",
        suffixes=("", "_gsdp")
    )

    # Compute rank gap: positive = index outperforms GDP
    df["rank_gap"] = np.nan
    mask = df["gsdp_rank"].notna() & df["rank"].notna()
    df.loc[mask, "rank_gap"] = df.loc[mask, "gsdp_rank"] - df.loc[mask, "rank"]

    # Gap label
    df["gap_label"] = None
    df.loc[df["rank_gap"] >= 3, "gap_label"] = "outperformer"
    df.loc[df["rank_gap"] <= -3, "gap_label"] = "underperformer"
    df.loc[(df["rank_gap"] > -3) & (df["rank_gap"] < 3) & df["rank_gap"].notna(), "gap_label"] = "aligned"

    gsdp_states = df[df["gsdp_rank"].notna()]["state"].nunique()
    print(f"  States with GSDP data: {gsdp_states}")

    return df


# ---------------------------------------------------------------------------
# Component Diagnostics
# ---------------------------------------------------------------------------

def compute_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute strongest/weakest component and diagnostic text."""
    print("Computing component diagnostics...")
    df = df.copy()

    z_cols = list(ZSCORE_COLS.keys())

    df["strongest_component"] = None
    df["weakest_component"] = None
    df["divergence_score"] = np.nan
    df["diagnostic_text"] = None

    for idx, row in df.iterrows():
        zscores = {col: row[col] for col in z_cols if pd.notna(row[col])}
        if len(zscores) < 3:
            continue

        strongest = max(zscores, key=zscores.get)
        weakest = min(zscores, key=zscores.get)
        divergence = max(zscores.values()) - min(zscores.values())

        df.loc[idx, "strongest_component"] = ZSCORE_SHORT[strongest]
        df.loc[idx, "weakest_component"] = ZSCORE_SHORT[weakest]
        df.loc[idx, "divergence_score"] = divergence

        state = row["state"]
        if divergence > 0.5:
            df.loc[idx, "diagnostic_text"] = (
                f"{state}: Strongest on {ZSCORE_COLS[strongest]}, "
                f"relatively weaker on {ZSCORE_COLS[weakest]}."
            )
        else:
            df.loc[idx, "diagnostic_text"] = (
                f"{state}: Balanced across all four components."
            )

    return df


# ---------------------------------------------------------------------------
# BRAP Data
# ---------------------------------------------------------------------------

def merge_brap(df: pd.DataFrame) -> pd.DataFrame:
    """Merge BRAP reform categories."""
    print("Merging BRAP data...")
    brap_path = DATA_REF / "brap_categories.csv"
    if not brap_path.exists():
        print("  WARNING: brap_categories.csv not found. BRAP data disabled.")
        df["brap_category"] = None
        return df

    brap = pd.read_csv(brap_path)
    print(f"  BRAP data: {len(brap)} states")

    df = df.merge(
        brap[["state", "brap_2020_category"]],
        on="state",
        how="left"
    )
    df = df.rename(columns={"brap_2020_category": "brap_category"})

    matched = df[df["brap_category"].notna()]["state"].nunique()
    print(f"  States with BRAP data: {matched}")

    return df


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def compute_correlations(df: pd.DataFrame) -> dict:
    """Compute Pearson correlations between components and GSDP."""
    print("Computing component-GSDP correlations...")

    if "gsdp_current_crore" not in df.columns or df["gsdp_current_crore"].isna().all():
        print("  WARNING: No GSDP data. Correlations skipped.")
        return {}

    # Use latest FY with both GSDP and index data
    fys_with_gsdp = df[df["gsdp_current_crore"].notna()]["fiscal_year"].unique()
    if len(fys_with_gsdp) == 0:
        return {}

    latest_fy = sorted(fys_with_gsdp)[-1]
    latest = df[(df["fiscal_year"] == latest_fy) & df["gsdp_current_crore"].notna()].copy()
    print(f"  Using FY {latest_fy}: {len(latest)} states")

    correlations = {"latest_fy": latest_fy, "note": "Cross-sectional Pearson r (across states, same FY)."}
    comp_map = {
        "gst_total": "gst_gsdp",
        "electricity_mu": "electricity_gsdp",
        "bank_credit_yoy": "credit_gsdp",
        "epfo_payroll": "epfo_gsdp",
        "composite_score": "composite_gsdp",
    }

    for comp_col, label in comp_map.items():
        valid = latest[[comp_col, "gsdp_current_crore"]].dropna()
        if len(valid) < MIN_STATES_FOR_CORR:
            correlations[label] = {"r": None, "p": None, "n": len(valid), "note": "Insufficient data"}
            continue
        r, p = stats.pearsonr(valid[comp_col], valid["gsdp_current_crore"])
        correlations[label] = {"r": round(r, 4), "p": round(p, 6), "n": len(valid)}

    return correlations


# ---------------------------------------------------------------------------
# OLS Regression
# ---------------------------------------------------------------------------

def compute_regression(df: pd.DataFrame) -> dict:
    """Run cross-sectional OLS and stepwise model comparison."""
    print("Computing OLS regression...")

    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.stats.stattools import durbin_watson
    except ImportError:
        print("  WARNING: statsmodels not installed. Regression analysis skipped.")
        return {"skipped": True, "reason": "statsmodels not installed"}

    if "gsdp_current_crore" not in df.columns or df["gsdp_current_crore"].isna().all():
        print("  WARNING: No GSDP data. Regression skipped.")
        return {"skipped": True, "reason": "No GSDP data available"}

    result = {
        "generated_at": datetime.now().isoformat(),
        "note": "Cross-sectional OLS: GSDP regressed on 4 activity indicators. NOT a causal model.",
    }

    # Find FYs with both GSDP and index data
    regressors = ["gst_total", "electricity_mu", "bank_credit_yoy", "epfo_payroll"]
    fys_with_data = []
    for fy in sorted(df["fiscal_year"].unique()):
        fy_data = df[(df["fiscal_year"] == fy) & df["gsdp_current_crore"].notna()]
        # Need enough states with all 4 components + GSDP
        valid = fy_data[regressors + ["gsdp_current_crore"]].dropna()
        if len(valid) >= MIN_STATES_FOR_REGRESSION:
            fys_with_data.append(fy)

    if not fys_with_data:
        print("  WARNING: No FY with enough data for regression.")
        return {"skipped": True, "reason": f"Fewer than {MIN_STATES_FOR_REGRESSION} states with complete data in any FY"}

    latest_fy = fys_with_data[-1]
    result["latest_fy_with_gsdp"] = latest_fy
    print(f"  FYs with sufficient data: {fys_with_data}")
    print(f"  Running regression for FY {latest_fy}")

    # Get clean data for latest FY
    fy_df = df[(df["fiscal_year"] == latest_fy)].copy()
    reg_data = fy_df[["state"] + regressors + ["gsdp_current_crore"]].dropna()
    n = len(reg_data)
    print(f"  N = {n} states")

    y = reg_data["gsdp_current_crore"].values
    X = reg_data[regressors].values
    X_with_const = sm.add_constant(X)

    # --- Full model ---
    model = sm.OLS(y, X_with_const).fit()
    print(f"  R-squared: {model.rsquared:.4f}, Adj R-squared: {model.rsquared_adj:.4f}")

    # Coefficients
    coefs = {}
    var_names = ["const"] + regressors
    for i, name in enumerate(var_names):
        coefs[name] = {
            "coef": round(float(model.params[i]), 6),
            "se": round(float(model.bse[i]), 6),
            "t": round(float(model.tvalues[i]), 4),
            "p": round(float(model.pvalues[i]), 6),
            "ci_low": round(float(model.conf_int()[i, 0]), 4),
            "ci_high": round(float(model.conf_int()[i, 1]), 4),
        }
    # Beta weights (standardized coefficients)
    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    y_std = (y - y.mean()) / y.std(ddof=1)
    model_std = sm.OLS(y_std, sm.add_constant(X_std)).fit()
    for i, name in enumerate(regressors):
        coefs[name]["beta"] = round(float(model_std.params[i + 1]), 4)

    # --- Diagnostics ---
    diagnostics = {}

    # VIF
    try:
        vif = {}
        for i, name in enumerate(regressors):
            vif[name] = round(float(variance_inflation_factor(X_with_const, i + 1)), 2)
        diagnostics["vif"] = vif
        high_vif = [k for k, v in vif.items() if v > 10]
        if high_vif:
            diagnostics["vif_warning"] = f"VIF > 10 for {', '.join(high_vif)}. Individual coefficient estimates unreliable."
    except Exception:
        diagnostics["vif"] = "Could not compute"

    # Breusch-Pagan
    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, X_with_const)
        diagnostics["breusch_pagan_stat"] = round(float(bp_stat), 4)
        diagnostics["breusch_pagan_p"] = round(float(bp_p), 4)
        if bp_p < 0.05:
            diagnostics["heteroscedasticity_warning"] = "Heteroscedasticity detected. Robust standard errors recommended."
            # Re-fit with HC3
            model_robust = sm.OLS(y, X_with_const).fit(cov_type="HC3")
            diagnostics["robust_se_used"] = True
            for i, name in enumerate(var_names):
                coefs[name]["robust_se"] = round(float(model_robust.bse[i]), 6)
                coefs[name]["robust_t"] = round(float(model_robust.tvalues[i]), 4)
                coefs[name]["robust_p"] = round(float(model_robust.pvalues[i]), 6)
    except Exception:
        diagnostics["breusch_pagan_p"] = "Could not compute"

    # Shapiro-Wilk
    try:
        sw_stat, sw_p = stats.shapiro(model.resid)
        diagnostics["shapiro_wilk_stat"] = round(float(sw_stat), 4)
        diagnostics["shapiro_wilk_p"] = round(float(sw_p), 4)
        if sw_p < 0.05:
            diagnostics["normality_warning"] = "Residuals are not normally distributed. F-tests and t-tests are approximate."
    except Exception:
        diagnostics["shapiro_wilk_p"] = "Could not compute"

    # Cook's Distance
    try:
        influence = model.get_influence()
        cooks = influence.cooks_distance[0]
        threshold = 4 / n
        flagged = reg_data.iloc[cooks > threshold]["state"].tolist()
        diagnostics["cooks_distance_threshold"] = round(threshold, 4)
        diagnostics["cooks_distance_flagged"] = flagged
    except Exception:
        diagnostics["cooks_distance_flagged"] = []

    # Durbin-Watson
    try:
        dw = durbin_watson(model.resid)
        diagnostics["durbin_watson"] = round(float(dw), 4)
    except Exception:
        diagnostics["durbin_watson"] = None

    cross_sectional_result = {
        "n": n,
        "r_squared": round(float(model.rsquared), 4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "f_statistic": round(float(model.fvalue), 4),
        "f_pvalue": round(float(model.f_pvalue), 6),
        "degrees_of_freedom": {"residual": int(model.df_resid), "model": int(model.df_model)},
        "coefficients": coefs,
        "diagnostics": diagnostics,
    }

    # --- Without Maharashtra ---
    maha_mask = reg_data["state"] != "Maharashtra"
    if maha_mask.sum() >= MIN_STATES_FOR_REGRESSION:
        y_no_mh = reg_data.loc[maha_mask, "gsdp_current_crore"].values
        X_no_mh = reg_data.loc[maha_mask, regressors].values
        X_no_mh_c = sm.add_constant(X_no_mh)
        model_no_mh = sm.OLS(y_no_mh, X_no_mh_c).fit()
        cross_sectional_result["without_maharashtra"] = {
            "n": int(maha_mask.sum()),
            "r_squared": round(float(model_no_mh.rsquared), 4),
            "adj_r_squared": round(float(model_no_mh.rsquared_adj), 4),
            "f_statistic": round(float(model_no_mh.fvalue), 4),
            "f_pvalue": round(float(model_no_mh.f_pvalue), 6),
        }
        r2_change = model_no_mh.rsquared - model.rsquared
        if r2_change > 0.01:
            cross_sectional_result["without_maharashtra"]["note"] = (
                f"R-squared increases by {r2_change:.3f} without Maharashtra. "
                f"Maharashtra is an outlier (likely due to Mumbai's inflated bank credit) "
                f"that slightly worsens overall model fit."
            )
        elif r2_change < -0.05:
            cross_sectional_result["without_maharashtra"]["note"] = (
                f"R-squared drops by {abs(r2_change):.3f} without Maharashtra, "
                f"suggesting it is highly influential."
            )

    result["cross_sectional"] = {latest_fy: cross_sectional_result}

    # --- Stepwise Model Comparison ---
    print("  Running stepwise model comparison...")
    stepwise = {}
    step_vars = [
        ("model_1_gst_only", ["gst_total"]),
        ("model_2_gst_elec", ["gst_total", "electricity_mu"]),
        ("model_3_gst_elec_credit", ["gst_total", "electricity_mu", "bank_credit_yoy"]),
        ("model_4_full", ["gst_total", "electricity_mu", "bank_credit_yoy", "epfo_payroll"]),
    ]

    prev_r2 = 0
    prev_model = None
    for name, vars_list in step_vars:
        X_step = reg_data[vars_list].values
        X_step_c = sm.add_constant(X_step)
        m = sm.OLS(y, X_step_c).fit()

        entry = {
            "r_squared": round(float(m.rsquared), 4),
            "adj_r_squared": round(float(m.rsquared_adj), 4),
            "aic": round(float(m.aic), 2),
            "bic": round(float(m.bic), 2),
        }

        if prev_model is not None:
            entry["delta_r2"] = round(float(m.rsquared - prev_r2), 4)
            # Partial F-test
            rss_reduced = prev_model.ssr
            rss_full = m.ssr
            df_diff = prev_model.df_resid - m.df_resid
            if df_diff > 0 and m.df_resid > 0:
                partial_f = ((rss_reduced - rss_full) / df_diff) / (rss_full / m.df_resid)
                partial_f_p = 1 - stats.f.cdf(partial_f, df_diff, m.df_resid)
                entry["partial_f"] = round(float(partial_f), 4)
                entry["partial_f_p"] = round(float(partial_f_p), 6)

        stepwise[name] = entry
        prev_r2 = m.rsquared
        prev_model = m

    result["stepwise"] = stepwise

    # --- Pooled Panel Regression (if enough data) ---
    all_panel = []
    for fy in fys_with_data:
        fy_df = df[df["fiscal_year"] == fy][["state"] + regressors + ["gsdp_current_crore", "fiscal_year"]].dropna()
        all_panel.append(fy_df)

    if all_panel:
        panel = pd.concat(all_panel, ignore_index=True)
        if len(panel) >= 60:
            print(f"  Running pooled panel regression ({len(panel)} obs)...")
            y_panel = panel["gsdp_current_crore"].values
            X_panel = panel[regressors].values
            # Year dummies
            year_dummies = pd.get_dummies(panel["fiscal_year"], drop_first=True, dtype=float)
            X_panel_full = np.column_stack([X_panel, year_dummies.values])
            X_panel_full = sm.add_constant(X_panel_full)
            panel_model = sm.OLS(y_panel, X_panel_full).fit()
            result["pooled_panel"] = {
                "n_observations": len(panel),
                "n_years": len(fys_with_data),
                "r_squared": round(float(panel_model.rsquared), 4),
                "adj_r_squared": round(float(panel_model.rsquared_adj), 4),
                "f_statistic": round(float(panel_model.fvalue), 4),
                "f_pvalue": round(float(panel_model.f_pvalue), 6),
                "note": "Year dummies included to absorb national trends."
            }
        else:
            result["pooled_panel"] = {"skipped": True, "reason": f"Only {len(panel)} observations (need 60+)"}

    # --- Interpretation ---
    r2 = model.rsquared
    f_stat = model.fvalue
    f_p = model.f_pvalue
    interp_parts = [
        f"The four activity indicators jointly explain ~{r2*100:.0f}% of cross-sectional "
        f"variation in state GSDP (F={f_stat:.1f}, p<{max(f_p, 0.001):.3f})."
    ]
    if stepwise:
        sig_steps = sum(1 for k, v in stepwise.items() if v.get("partial_f_p", 1) < 0.05)
        if sig_steps > 0:
            interp_parts.append(f"{sig_steps} of 3 additional components significantly improve fit (partial F-test, p<0.05).")

    vif_vals = diagnostics.get("vif", {})
    if isinstance(vif_vals, dict) and vif_vals:
        max_vif = max(vif_vals.values())
        min_vif = min(vif_vals.values())
        if max_vif > 10:
            interp_parts.append(
                f"VIF values ({min_vif:.1f}-{max_vif:.1f}) indicate high multicollinearity; "
                f"individual coefficient estimates are unreliable, but joint significance (F-test) remains valid."
            )
        elif max_vif > 5:
            interp_parts.append(f"VIF values ({min_vif:.1f}-{max_vif:.1f}) indicate moderate multicollinearity.")

    if "without_maharashtra" in cross_sectional_result:
        mh_r2 = cross_sectional_result["without_maharashtra"]["r_squared"]
        r2_diff = mh_r2 - r2
        if r2_diff > 0:
            interp_parts.append(
                f"R-squared increases to {mh_r2:.2f} without Maharashtra, "
                f"indicating it is an outlier that slightly worsens fit "
                f"(likely due to Mumbai's inflated bank credit)."
            )
        elif abs(r2_diff) > 0.05:
            interp_parts.append(
                f"R-squared drops to {mh_r2:.2f} without Maharashtra, "
                f"suggesting it is highly influential."
            )

    interp_parts.append("These are associations, not causal relationships.")
    result["interpretation"] = " ".join(interp_parts)

    return result


# ---------------------------------------------------------------------------
# Panel Fixed-Effects Regression
# ---------------------------------------------------------------------------

def compute_panel_fe_regression(df: pd.DataFrame) -> dict:
    """Run state FE and RE panel regressions with Hausman test."""
    print("Computing panel fixed-effects regression...")

    regressors = ["gst_total", "electricity_mu", "bank_credit_yoy", "epfo_payroll"]

    if "gsdp_current_crore" not in df.columns or df["gsdp_current_crore"].isna().all():
        print("  WARNING: No GSDP data. Panel FE skipped.")
        return {"skipped": True, "reason": "No GSDP data"}

    # Build panel dataset: all FYs with GSDP + all 4 regressors
    panel = df[["state", "fiscal_year"] + regressors + ["gsdp_current_crore"]].dropna().copy()
    if len(panel) < 30:
        print(f"  WARNING: Only {len(panel)} complete obs. Need 30+. Panel FE skipped.")
        return {"skipped": True, "reason": f"Only {len(panel)} complete observations"}

    n_states = panel["state"].nunique()
    n_years = panel["fiscal_year"].nunique()
    print(f"  Panel: {len(panel)} obs, {n_states} states, {n_years} years")

    try:
        from linearmodels.panel import PanelOLS, RandomEffects
    except ImportError:
        print("  WARNING: linearmodels not installed. Falling back to manual FE.")
        return _manual_panel_fe(panel, regressors)

    # Set up panel index — linearmodels needs numeric or date-like time index
    # Convert fiscal_year "2023-24" -> 2023 (start year)
    panel["year_num"] = panel["fiscal_year"].apply(lambda fy: int(fy.split("-")[0]))
    panel = panel.set_index(["state", "year_num"])

    y = panel["gsdp_current_crore"]
    X = panel[regressors]

    # Fixed Effects
    fe_model = PanelOLS(y, X, entity_effects=True, time_effects=True,
                        check_rank=False).fit(cov_type="clustered", cluster_entity=True)

    # Random Effects
    try:
        re_model = RandomEffects(y, X, check_rank=False).fit()
        re_r2 = round(float(re_model.rsquared), 4)
    except Exception:
        re_model = None
        re_r2 = None

    result = {
        "n": len(panel),
        "n_states": n_states,
        "n_years": n_years,
        "within_r_squared": round(float(fe_model.rsquared_within), 4),
        "between_r_squared": round(float(fe_model.rsquared_between), 4),
        "overall_r_squared": round(float(fe_model.rsquared_overall), 4),
        "coefficients": {},
        "note": "State FE absorbs time-invariant state characteristics. "
                "Year FE absorbs national trends. Clustered SEs by state.",
    }

    for var in regressors:
        if var in fe_model.params.index:
            result["coefficients"][var] = {
                "coef": round(float(fe_model.params[var]), 4),
                "se": round(float(fe_model.std_errors[var]), 4),
                "t": round(float(fe_model.tstats[var]), 4),
                "p": round(float(fe_model.pvalues[var]), 6),
            }

    # Hausman test (FE vs RE)
    if re_model is not None:
        try:
            # Manual Hausman: H = (b_FE - b_RE)' [V_FE - V_RE]^-1 (b_FE - b_RE)
            common_vars = [v for v in regressors if v in fe_model.params.index and v in re_model.params.index]
            b_fe = fe_model.params[common_vars].values
            b_re = re_model.params[common_vars].values
            v_fe = fe_model.cov[common_vars].loc[common_vars].values
            v_re = re_model.cov[common_vars].loc[common_vars].values
            diff = b_fe - b_re
            v_diff = v_fe - v_re
            try:
                h_stat = float(diff @ np.linalg.inv(v_diff) @ diff)
                h_p = float(1 - stats.chi2.cdf(h_stat, len(common_vars)))
                result["hausman_test"] = {
                    "stat": round(h_stat, 4),
                    "p": round(h_p, 4),
                    "preferred": "FE" if h_p < 0.05 else "RE",
                    "note": "p<0.05 favors FE (systematic differences between states)."
                }
            except np.linalg.LinAlgError:
                result["hausman_test"] = {"note": "Could not compute (singular covariance difference)."}
        except Exception as e:
            result["hausman_test"] = {"note": f"Could not compute: {str(e)[:100]}"}

    return result


def _manual_panel_fe(panel: pd.DataFrame, regressors: list) -> dict:
    """Fallback panel FE using statsmodels OLS with dummy variables."""
    import statsmodels.api as sm

    n_states = panel["state"].nunique()
    n_years = panel["fiscal_year"].nunique()

    y = panel["gsdp_current_crore"].values
    X = panel[regressors].values

    # Add state dummies
    state_dummies = pd.get_dummies(panel["state"], drop_first=True, dtype=float)
    year_dummies = pd.get_dummies(panel["fiscal_year"], drop_first=True, dtype=float)
    X_full = np.column_stack([X, state_dummies.values, year_dummies.values])
    X_full = sm.add_constant(X_full)

    model = sm.OLS(y, X_full).fit(cov_type="HC1")

    result = {
        "n": len(panel),
        "n_states": n_states,
        "n_years": n_years,
        "within_r_squared": round(float(model.rsquared), 4),
        "between_r_squared": None,
        "overall_r_squared": round(float(model.rsquared), 4),
        "coefficients": {},
        "note": "Manual FE via state+year dummies (linearmodels not available). HC1 robust SEs.",
    }

    for i, var in enumerate(regressors):
        result["coefficients"][var] = {
            "coef": round(float(model.params[i + 1]), 4),
            "se": round(float(model.bse[i + 1]), 4),
            "t": round(float(model.tvalues[i + 1]), 4),
            "p": round(float(model.pvalues[i + 1]), 6),
        }

    return result


# ---------------------------------------------------------------------------
# Log-Log Regression
# ---------------------------------------------------------------------------

def compute_log_log_regression(df: pd.DataFrame) -> dict:
    """Run log-log regression: coefficients are elasticities."""
    print("Computing log-log regression...")

    import statsmodels.api as sm

    regressors = ["gst_total", "electricity_mu", "bank_credit_yoy", "epfo_payroll"]

    if "gsdp_current_crore" not in df.columns or df["gsdp_current_crore"].isna().all():
        return {"skipped": True, "reason": "No GSDP data"}

    # Filter to positive values only (log requires >0)
    panel = df[["state", "fiscal_year"] + regressors + ["gsdp_current_crore"]].dropna().copy()
    for col in regressors + ["gsdp_current_crore"]:
        panel = panel[panel[col] > 0]

    if len(panel) < 20:
        return {"skipped": True, "reason": f"Only {len(panel)} obs with all positive values"}

    # Log transform
    log_cols = {}
    for col in regressors + ["gsdp_current_crore"]:
        log_name = f"log_{col}"
        panel[log_name] = np.log(panel[col])
        log_cols[col] = log_name

    result = {}

    # --- Cross-sectional (latest FY) ---
    fys_sorted = sorted(panel["fiscal_year"].unique())
    for fy in reversed(fys_sorted):
        fy_data = panel[panel["fiscal_year"] == fy]
        if len(fy_data) >= 15:
            y = fy_data[log_cols["gsdp_current_crore"]].values
            X = fy_data[[log_cols[r] for r in regressors]].values
            X_c = sm.add_constant(X)
            m = sm.OLS(y, X_c).fit()
            elasticities = {}
            for i, r in enumerate(regressors):
                elasticities[r] = {
                    "elasticity": round(float(m.params[i + 1]), 4),
                    "p": round(float(m.pvalues[i + 1]), 6),
                }
            result["cross_sectional"] = {
                "n": len(fy_data),
                "fiscal_year": fy,
                "r_squared": round(float(m.rsquared), 4),
                "adj_r_squared": round(float(m.rsquared_adj), 4),
                "elasticities": elasticities,
                "note": "Log-log coefficients = elasticities. A 1% increase in X is associated with beta% increase in GSDP.",
            }
            break

    # --- Panel (all FYs, with year dummies) ---
    if len(panel) >= 50:
        y_panel = panel[log_cols["gsdp_current_crore"]].values
        X_vars = [log_cols[r] for r in regressors]
        X_panel = panel[X_vars].values
        year_dummies = pd.get_dummies(panel["fiscal_year"], drop_first=True, dtype=float)
        X_full = np.column_stack([X_panel, year_dummies.values])
        X_full = sm.add_constant(X_full)
        m_panel = sm.OLS(y_panel, X_full).fit()

        panel_elasticities = {}
        for i, r in enumerate(regressors):
            panel_elasticities[r] = {
                "elasticity": round(float(m_panel.params[i + 1]), 4),
                "p": round(float(m_panel.pvalues[i + 1]), 6),
            }

        result["panel"] = {
            "n": len(panel),
            "n_years": panel["fiscal_year"].nunique(),
            "r_squared": round(float(m_panel.rsquared), 4),
            "elasticities": panel_elasticities,
            "note": "Pooled log-log with year dummies.",
        }

    print(f"  Cross-sectional R2: {result.get('cross_sectional', {}).get('r_squared', 'N/A')}")
    return result


# ---------------------------------------------------------------------------
# Lagged Panel Correlation
# ---------------------------------------------------------------------------

def compute_lagged_correlation(df: pd.DataFrame) -> dict:
    """Test if electricity growth leads GSDP growth by 1 year."""
    print("Computing 1-year lagged panel correlation...")

    import statsmodels.api as sm

    if "gsdp_current_crore" not in df.columns or df["gsdp_current_crore"].isna().all():
        return {"skipped": True, "reason": "No GSDP data"}

    gsdp_path = DATA_PROCESSED / "gsdp_clean.parquet"
    if not gsdp_path.exists():
        return {"skipped": True, "reason": "gsdp_clean.parquet not found"}

    gsdp = pd.read_parquet(gsdp_path)

    # Compute electricity YoY growth per state
    annual = df[df["period_type"] == "annual"][["state", "fiscal_year", "electricity_mu"]].dropna().copy()
    annual = annual.sort_values(["state", "fiscal_year"])
    annual["elec_growth"] = annual.groupby("state")["electricity_mu"].pct_change(fill_method=None) * 100

    # Merge GSDP growth
    if "gsdp_growth_pct" in gsdp.columns:
        growth_merge = gsdp[["state", "fiscal_year", "gsdp_growth_pct"]].dropna()
    else:
        return {"skipped": True, "reason": "No GSDP growth data"}

    merged = annual.merge(growth_merge, on=["state", "fiscal_year"], how="inner")

    # Create lagged variable: electricity growth at t-1 vs GSDP growth at t
    sorted_fys = sorted(merged["fiscal_year"].unique())
    fy_to_idx = {fy: i for i, fy in enumerate(sorted_fys)}

    lagged_rows = []
    for state in merged["state"].unique():
        state_data = merged[merged["state"] == state].sort_values("fiscal_year")
        for _, row in state_data.iterrows():
            fy = row["fiscal_year"]
            idx = fy_to_idx.get(fy)
            if idx is None or idx == 0:
                continue
            prev_fy = sorted_fys[idx - 1]
            prev = state_data[state_data["fiscal_year"] == prev_fy]
            if prev.empty:
                continue
            prev_elec_growth = prev.iloc[0]["elec_growth"]
            if pd.notna(prev_elec_growth) and pd.notna(row["gsdp_growth_pct"]):
                lagged_rows.append({
                    "state": state,
                    "fiscal_year": fy,
                    "elec_growth_lag1": prev_elec_growth,
                    "gsdp_growth": row["gsdp_growth_pct"],
                })

    if len(lagged_rows) < 20:
        return {"skipped": True, "reason": f"Only {len(lagged_rows)} lagged obs (need 20+)"}

    lag_df = pd.DataFrame(lagged_rows)
    n = len(lag_df)
    n_states = lag_df["state"].nunique()
    print(f"  Lagged panel: {n} obs, {n_states} states")

    # Simple pooled regression with state FE
    y = lag_df["gsdp_growth"].values
    X = lag_df["elec_growth_lag1"].values.reshape(-1, 1)

    # Add state dummies for FE
    state_dummies = pd.get_dummies(lag_df["state"], drop_first=True, dtype=float)
    X_full = np.column_stack([X, state_dummies.values])
    X_full = sm.add_constant(X_full)

    model = sm.OLS(y, X_full).fit()

    # Also simple correlation without FE
    r, p_corr = stats.pearsonr(lag_df["elec_growth_lag1"], lag_df["gsdp_growth"])

    result = {
        "n": n,
        "n_states": n_states,
        "lag_years": 1,
        "simple_correlation": {"r": round(float(r), 4), "p": round(float(p_corr), 4)},
        "panel_fe": {
            "electricity_growth_coef": round(float(model.params[1]), 4),
            "se": round(float(model.bse[1]), 4),
            "t": round(float(model.tvalues[1]), 4),
            "p": round(float(model.pvalues[1]), 6),
        },
        "note": "Lagged correlation, not a causal test. "
                "Tests whether electricity growth in year t-1 is associated with GSDP growth in year t. "
                "Reverse causality remains possible.",
    }

    if model.pvalues[1] < 0.05:
        result["interpretation"] = (
            f"States where electricity grew faster had significantly higher GSDP growth "
            f"the following year (coef={model.params[1]:.2f}, p={model.pvalues[1]:.3f}). "
            f"This is consistent with electricity being a leading indicator, but is not a causal claim."
        )
    else:
        result["interpretation"] = (
            f"No significant leading relationship detected between electricity growth and "
            f"subsequent GSDP growth (p={model.pvalues[1]:.3f}). Limited time depth ({n} obs) "
            f"may reduce statistical power."
        )

    print(f"  Lagged coef: {model.params[1]:.4f}, p={model.pvalues[1]:.4f}")
    return result


# ---------------------------------------------------------------------------
# PCA Weights Robustness Check
# ---------------------------------------------------------------------------

def compute_pca_weights(df: pd.DataFrame) -> dict:
    """Check if PCA-derived weights validate equal-weight approach."""
    print("Computing PCA weights robustness check...")

    z_cols = list(ZSCORE_COLS.keys())

    # Use latest FY with full data
    scored_fys = df[df["composite_score"].notna()]["fiscal_year"].unique()
    if len(scored_fys) == 0:
        return {"skipped": True, "reason": "No scored FYs"}

    latest_fy = sorted(scored_fys)[-1]
    latest = df[(df["fiscal_year"] == latest_fy) & df["composite_score"].notna()].copy()

    # Need at least 10 states with all 4 z-scores
    complete = latest[z_cols].dropna()
    if len(complete) < 10:
        return {"skipped": True, "reason": f"Only {len(complete)} states with all 4 z-scores"}

    from sklearn.decomposition import PCA as SklearnPCA

    try:
        X = complete[z_cols].values
        pca = SklearnPCA(n_components=min(4, X.shape[1]))
        pca.fit(X)

        pc1_loadings = {}
        for i, col in enumerate(z_cols):
            short = ZSCORE_SHORT[col].lower()
            pc1_loadings[short] = round(float(pca.components_[0][i]), 4)

        pc1_var = round(float(pca.explained_variance_ratio_[0]) * 100, 1)
        all_var = [round(float(v) * 100, 1) for v in pca.explained_variance_ratio_]

        # Compute PCA-weighted composite score
        weights = np.abs(pca.components_[0])
        weights = weights / weights.sum()
        pca_scores = X @ weights
        equal_scores = latest.loc[complete.index, "composite_score"].values

        # Rank correlation
        from scipy.stats import spearmanr
        rho, rho_p = spearmanr(pca_scores, equal_scores)

        # Build implied weights
        implied_weights = {}
        for i, col in enumerate(z_cols):
            short = ZSCORE_SHORT[col].lower()
            implied_weights[short] = round(float(weights[i]), 4)

        result = {
            "n": len(complete),
            "fiscal_year": latest_fy,
            "pc1_loadings": pc1_loadings,
            "pc1_variance_explained_pct": pc1_var,
            "all_variance_explained_pct": all_var,
            "implied_weights": implied_weights,
            "rank_correlation_with_equal_weights": round(float(rho), 4),
            "rank_correlation_p": round(float(rho_p), 6),
        }

        if rho > 0.95:
            result["interpretation"] = (
                f"PCA and equal weights produce nearly identical rankings (Spearman rho={rho:.3f}). "
                f"The equal-weight approach is validated by the data."
            )
        elif rho > 0.85:
            result["interpretation"] = (
                f"PCA and equal weights produce similar but not identical rankings (Spearman rho={rho:.3f}). "
                f"Moderate sensitivity to weighting scheme."
            )
        else:
            result["interpretation"] = (
                f"PCA weights diverge meaningfully from equal weights (Spearman rho={rho:.3f}). "
                f"PC1 loads most heavily on {max(pc1_loadings, key=pc1_loadings.get)}."
            )

        print(f"  PC1 explains {pc1_var}% of variance. Rank correlation with equal weights: {rho:.3f}")
        return result

    except ImportError:
        # Fallback using numpy eigendecomposition
        print("  sklearn not available, using numpy PCA fallback")
        X = complete[z_cols].values
        X_centered = X - X.mean(axis=0)
        cov_matrix = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        total_var = eigenvalues.sum()
        pc1_loadings = {}
        for i, col in enumerate(z_cols):
            short = ZSCORE_SHORT[col].lower()
            pc1_loadings[short] = round(float(eigenvectors[i, 0]), 4)

        pc1_var = round(float(eigenvalues[0] / total_var * 100), 1)

        weights = np.abs(eigenvectors[:, 0])
        weights = weights / weights.sum()
        pca_scores = X @ weights
        equal_scores = latest.loc[complete.index, "composite_score"].values
        rho, rho_p = stats.spearmanr(pca_scores, equal_scores)

        implied_weights = {}
        for i, col in enumerate(z_cols):
            short = ZSCORE_SHORT[col].lower()
            implied_weights[short] = round(float(weights[i]), 4)

        return {
            "n": len(complete),
            "fiscal_year": latest_fy,
            "pc1_loadings": pc1_loadings,
            "pc1_variance_explained_pct": pc1_var,
            "implied_weights": implied_weights,
            "rank_correlation_with_equal_weights": round(float(rho), 4),
            "rank_correlation_p": round(float(rho_p), 6),
            "interpretation": f"Spearman rho={rho:.3f} between PCA and equal-weight rankings.",
        }


# ---------------------------------------------------------------------------
# State Gap Explanations (Curated Narratives)
# ---------------------------------------------------------------------------

# Domain-knowledge explanations for states with |rank_gap| >= 3
STATE_GAP_TEMPLATES = {
    "Haryana": {
        "direction": "outperformer",
        "explanation": (
            "Haryana ranks significantly higher on the activity index than on official GSDP. "
            "Its proximity to Delhi drives high formal employment (EPFO) and GST-registered "
            "commercial activity in the Gurgaon/NCR belt, while moderate agriculture suppresses "
            "official GSDP less than states like Punjab or MP."
        ),
        "key_drivers": ["EPFO: NCR proximity drives formal employment", "GST: Gurgaon commercial hub"],
        "key_drags": ["Moderate agricultural share not captured by index"],
    },
    "Delhi": {
        "direction": "outperformer",
        "explanation": (
            "Delhi ranks higher on our index than on GSDP. As a city-state with near-zero agriculture, "
            "all four index components capture its activity well. High GST (services + retail), "
            "concentrated bank credit, and large formal workforce make it an index-friendly economy."
        ),
        "key_drivers": ["All 4 components well-represented", "No agriculture dilution"],
        "key_drags": [],
    },
    "Telangana": {
        "direction": "outperformer",
        "explanation": (
            "Telangana outperforms its GSDP rank on our index. Hyderabad's IT corridor drives "
            "high GST collections and EPFO registrations, while the state's rapid industrialization "
            "shows in electricity demand growth."
        ),
        "key_drivers": ["GST: Hyderabad services economy", "EPFO: IT sector formalization"],
        "key_drags": [],
    },
    "Rajasthan": {
        "direction": "outperformer",
        "explanation": (
            "Rajasthan ranks higher on the activity index than on GDP. Strong mining and "
            "industrial activity drives electricity demand, while recent GST growth suggests "
            "increasing formalization of its economy."
        ),
        "key_drivers": ["Electricity: mining and industrial activity", "GST growth"],
        "key_drags": ["Large agricultural sector partially invisible to index"],
    },
    "Odisha": {
        "direction": "outperformer",
        "explanation": (
            "Odisha outperforms its GDP rank on our index. Its heavy industry base (steel, aluminium, mining) "
            "drives high electricity intensity, and industrial expansion shows in GST and credit growth."
        ),
        "key_drivers": ["Electricity: heavy industry (steel, aluminium)", "Credit: industrial investment"],
        "key_drags": [],
    },
    "Kerala": {
        "direction": "underperformer",
        "explanation": (
            "Kerala ranks lower on our index than on official GSDP. Its economy is remittance-driven "
            "and services-heavy (tourism, healthcare, education) -- sectors that don't register "
            "strongly in GST collections or electricity demand. The index misses Kerala's large "
            "informal services sector and NRI remittance economy."
        ),
        "key_drivers": [],
        "key_drags": ["Remittance economy invisible to index", "Services-heavy (low electricity intensity)"],
    },
    "Punjab": {
        "direction": "underperformer",
        "explanation": (
            "Punjab's GSDP rank exceeds its index rank because agriculture (which our index doesn't "
            "capture) contributes significantly to its economy. Farm power is subsidized and partly "
            "unmeasured, and agricultural transactions are largely GST-exempt."
        ),
        "key_drivers": [],
        "key_drags": ["Agriculture invisible to all 4 components", "Subsidized farm electricity unmeasured"],
    },
    "Madhya Pradesh": {
        "direction": "underperformer",
        "explanation": (
            "Madhya Pradesh ranks lower on our index than on GSDP. Its large agricultural sector "
            "(~25% of GSDP) is invisible to our four indicators. Additionally, its economy has a "
            "significant informal component that GST and EPFO don't capture."
        ),
        "key_drivers": [],
        "key_drags": ["Large agricultural share (~25% GSDP)", "Informal sector activity"],
    },
    "West Bengal": {
        "direction": "underperformer",
        "explanation": (
            "West Bengal ranks lower on our index than on GDP. Its large informal economy, "
            "significant agricultural base, and small-enterprise-dominated manufacturing "
            "are underrepresented by formal indicators like GST and EPFO."
        ),
        "key_drivers": [],
        "key_drags": ["Large informal economy", "Small-enterprise manufacturing", "Agricultural base"],
    },
    "Tripura": {
        "direction": "underperformer",
        "explanation": (
            "Tripura shows the largest negative gap between GDP rank and index rank. "
            "As a small northeastern state, its economy relies heavily on government spending, "
            "agriculture, and informal activity -- none of which our four indicators capture well."
        ),
        "key_drivers": [],
        "key_drags": ["Government-spending driven", "Agriculture", "Very small formal sector"],
    },
    "Arunachal Pradesh": {
        "direction": "underperformer",
        "explanation": (
            "Arunachal Pradesh ranks much lower on our index than on GDP. Its economy is dominated "
            "by government expenditure and hydropower revenue that flow through channels our index "
            "doesn't fully capture. Very low formal sector presence limits EPFO and GST signals."
        ),
        "key_drivers": [],
        "key_drags": ["Government expenditure-driven", "Hydropower revenue accounting", "Minimal formal sector"],
    },
    "Maharashtra": {
        "direction": "aligned",
        "explanation": (
            "Maharashtra ranks #1 on both the activity index and official GSDP. Mumbai's role as "
            "India's financial capital inflates bank credit (corporate HQs book credit here), "
            "but its genuinely massive GST, electricity, and EPFO numbers make it the clear leader."
        ),
        "key_drivers": ["All 4 components: dominant across the board"],
        "key_drags": ["Bank credit inflated by Mumbai financial hub effect"],
    },
    "Tamil Nadu": {
        "direction": "aligned",
        "explanation": (
            "Tamil Nadu is well-aligned between index and GDP ranks. Its diversified economy -- "
            "strong manufacturing (autos, electronics), services, and formal employment -- is "
            "well-captured by all four indicators."
        ),
        "key_drivers": ["EPFO: #1 in formal employment", "Diversified economy"],
        "key_drags": [],
    },
    "Gujarat": {
        "direction": "aligned",
        "explanation": (
            "Gujarat is well-aligned between index and GDP ranks. Its industrial economy -- "
            "refining, chemicals, textiles -- drives high electricity demand and GST collections. "
            "Strong formal sector presence shows in EPFO numbers."
        ),
        "key_drivers": ["Electricity: heavy industry", "GST: manufacturing + trade"],
        "key_drags": [],
    },
    "Karnataka": {
        "direction": "aligned",
        "explanation": (
            "Karnataka is roughly aligned between index and GDP. Bangalore's IT sector drives "
            "EPFO and GST, while southern Karnataka's industry supports electricity demand. "
            "Slight divergence possible as IT services are less electricity-intensive."
        ),
        "key_drivers": ["EPFO: Bangalore IT employment", "GST: services economy"],
        "key_drags": ["IT services less electricity-intensive"],
    },
    "Uttar Pradesh": {
        "direction": "aligned",
        "explanation": (
            "Uttar Pradesh is roughly aligned despite being India's most populous state. "
            "Its sheer scale in electricity demand, GST, and EPFO numbers places it in the top 5, "
            "though per-capita metrics would tell a different story."
        ),
        "key_drivers": ["Scale: India's largest population"],
        "key_drags": ["Per-capita metrics would rank much lower"],
    },
}


def generate_gap_explanations(df: pd.DataFrame) -> dict:
    """Generate curated gap explanations for each state."""
    print("Generating state gap explanations...")

    meta = load_state_metadata()
    slug_map = {row["canonical_name"]: row["slug"] for _, row in meta.iterrows()}

    # Get latest FY with scores
    scored_fys = df[df["composite_score"].notna()]["fiscal_year"].unique()
    if len(scored_fys) == 0:
        return {}

    latest_fy = sorted(scored_fys)[-1]
    latest = df[df["fiscal_year"] == latest_fy].copy()

    explanations = {}
    for _, row in latest.iterrows():
        state = row["state"]
        slug = slug_map.get(state, "")
        if not slug:
            continue

        rank_gap = row.get("rank_gap")
        index_rank = row.get("rank")
        gsdp_rank = row.get("gsdp_rank")

        template = STATE_GAP_TEMPLATES.get(state)

        if template:
            entry = {
                "state": state,
                "slug": slug,
                "index_rank": int(index_rank) if pd.notna(index_rank) else None,
                "gsdp_rank": int(gsdp_rank) if pd.notna(gsdp_rank) else None,
                "rank_gap": int(rank_gap) if pd.notna(rank_gap) else None,
                "direction": template["direction"],
                "explanation": template["explanation"],
                "key_drivers": template["key_drivers"],
                "key_drags": template["key_drags"],
                "strongest_component": row.get("strongest_component") if pd.notna(row.get("strongest_component")) else None,
                "weakest_component": row.get("weakest_component") if pd.notna(row.get("weakest_component")) else None,
            }
        elif pd.notna(rank_gap) and abs(rank_gap) >= 3:
            # Auto-generate for uncurated states with large gap
            direction = "outperformer" if rank_gap > 0 else "underperformer"
            strongest = row.get("strongest_component", "")
            weakest = row.get("weakest_component", "")
            if direction == "outperformer":
                explanation = (
                    f"{state} ranks {int(abs(rank_gap))} positions higher on the activity index "
                    f"than on official GSDP. "
                    f"Strongest on {strongest.lower() if strongest else 'N/A'}, "
                    f"suggesting higher formal economic activity than GDP estimates capture."
                )
            else:
                explanation = (
                    f"{state} ranks {int(abs(rank_gap))} positions lower on the activity index "
                    f"than on official GSDP. "
                    f"The gap likely reflects economic activity (agriculture, informal sector, "
                    f"government spending) that our four indicators do not capture."
                )
            entry = {
                "state": state,
                "slug": slug,
                "index_rank": int(index_rank) if pd.notna(index_rank) else None,
                "gsdp_rank": int(gsdp_rank) if pd.notna(gsdp_rank) else None,
                "rank_gap": int(rank_gap) if pd.notna(rank_gap) else None,
                "direction": direction,
                "explanation": explanation,
                "key_drivers": [],
                "key_drags": [],
                "strongest_component": strongest if pd.notna(row.get("strongest_component")) else None,
                "weakest_component": weakest if pd.notna(row.get("weakest_component")) else None,
            }
        elif pd.notna(rank_gap):
            # Aligned state — short note
            entry = {
                "state": state,
                "slug": slug,
                "index_rank": int(index_rank) if pd.notna(index_rank) else None,
                "gsdp_rank": int(gsdp_rank) if pd.notna(gsdp_rank) else None,
                "rank_gap": int(rank_gap) if pd.notna(rank_gap) else None,
                "direction": "aligned",
                "explanation": f"{state} is closely aligned between the activity index and official GSDP rankings.",
                "key_drivers": [],
                "key_drags": [],
                "strongest_component": row.get("strongest_component") if pd.notna(row.get("strongest_component")) else None,
                "weakest_component": row.get("weakest_component") if pd.notna(row.get("weakest_component")) else None,
            }
        else:
            continue

        explanations[slug] = entry

    outperformers = sorted(
        [e for e in explanations.values() if e["direction"] == "outperformer"],
        key=lambda x: -(x.get("rank_gap") or 0)
    )
    underperformers = sorted(
        [e for e in explanations.values() if e["direction"] == "underperformer"],
        key=lambda x: (x.get("rank_gap") or 0)
    )

    print(f"  {len(outperformers)} outperformers, {len(underperformers)} underperformers, "
          f"{len(explanations) - len(outperformers) - len(underperformers)} aligned")

    return {
        "latest_fy": latest_fy,
        "all": explanations,
        "top_outperformers": [e["slug"] for e in outperformers[:5]],
        "top_underperformers": [e["slug"] for e in underperformers[:5]],
    }


# ---------------------------------------------------------------------------
# Regional Analysis
# ---------------------------------------------------------------------------

def compute_regional_analysis(df: pd.DataFrame) -> dict:
    """Compute regional aggregations and trends."""
    print("Computing regional analysis...")

    meta = load_state_metadata()
    region_map = {row["canonical_name"]: row["region"] for _, row in meta.iterrows()}

    scored = df[df["composite_score"].notna()].copy()
    scored["region"] = scored["state"].map(region_map)
    scored = scored[scored["region"].notna()]

    if scored.empty:
        return {}

    # Per-region summary for latest FY
    scored_fys = sorted(scored["fiscal_year"].unique())
    latest_fy = scored_fys[-1]
    latest = scored[scored["fiscal_year"] == latest_fy]

    regions = {}
    for region in sorted(latest["region"].unique()):
        region_data = latest[latest["region"] == region]
        regions[region] = {
            "mean_composite": round(float(region_data["composite_score"].mean()), 4),
            "median_composite": round(float(region_data["composite_score"].median()), 4),
            "n_states": int(len(region_data)),
            "states": sorted(region_data["state"].tolist()),
            "best_state": region_data.loc[region_data["composite_score"].idxmax(), "state"],
            "worst_state": region_data.loc[region_data["composite_score"].idxmin(), "state"],
        }

        # Add GSDP metrics if available
        if "gsdp_rank" in region_data.columns:
            gsdp_valid = region_data[region_data["gsdp_rank"].notna()]
            if len(gsdp_valid) > 0:
                regions[region]["mean_gsdp_rank"] = round(float(gsdp_valid["gsdp_rank"].mean()), 1)

    # Regional trends over time
    trends = {}
    for region in sorted(scored["region"].unique()):
        region_trend = []
        for fy in scored_fys:
            fy_region = scored[(scored["fiscal_year"] == fy) & (scored["region"] == region)]
            if len(fy_region) > 0:
                region_trend.append({
                    "fiscal_year": fy,
                    "mean_composite": round(float(fy_region["composite_score"].mean()), 4),
                    "n_states": int(len(fy_region)),
                })
        trends[region] = region_trend

    # Classify trend direction for each region
    for region in regions:
        trend_data = trends.get(region, [])
        if len(trend_data) >= 3:
            first_half = np.mean([t["mean_composite"] for t in trend_data[:len(trend_data)//2]])
            second_half = np.mean([t["mean_composite"] for t in trend_data[len(trend_data)//2:]])
            if second_half - first_half > 0.1:
                regions[region]["trend"] = "rising"
            elif first_half - second_half > 0.1:
                regions[region]["trend"] = "declining"
            else:
                regions[region]["trend"] = "stable"

    result = {
        "latest_fy": latest_fy,
        "regions": regions,
        "trends": trends,
    }

    print(f"  {len(regions)} regions analyzed")
    for r, data in sorted(regions.items(), key=lambda x: -x[1]["mean_composite"]):
        print(f"    {r}: mean={data['mean_composite']:.3f}, n={data['n_states']}")

    return result


# ---------------------------------------------------------------------------
# Key Findings Auto-Generation
# ---------------------------------------------------------------------------

def generate_key_findings(df: pd.DataFrame, correlations: dict) -> list:
    """Generate auto-generated key findings with threshold guards."""
    print("Generating key findings...")
    findings = []

    # Get latest FY with scores
    scored_fys = df[df["composite_score"].notna()]["fiscal_year"].unique()
    latest_fy = sorted(scored_fys)[-1] if len(scored_fys) > 0 else sorted(df["fiscal_year"].unique())[-1]
    latest = df[df["fiscal_year"] == latest_fy].copy()

    meta = load_state_metadata()
    slug_map = {row["canonical_name"]: row["slug"] for _, row in meta.iterrows()}

    # 1. Fastest risers by rank_momentum_3yr
    risers = latest[latest["rank_momentum_3yr"] >= MOMENTUM_RISING_THRESHOLD].nlargest(3, "rank_momentum_3yr")
    if len(risers) > 0:
        states_list = [{"state": r["state"], "slug": slug_map.get(r["state"], ""), "momentum": int(r["rank_momentum_3yr"])} for _, r in risers.iterrows()]
        findings.append({
            "type": "fastest_rising",
            "title": "Fastest Rising States",
            "detail": f"{states_list[0]['state']} climbed {states_list[0]['momentum']} positions in 3 years, "
                     f"showing the strongest upward trajectory among all states.",
            "states": [s["slug"] for s in states_list],
        })

    # 2. Top outperformers (index >> GSDP)
    outperformers = latest[latest["rank_gap"] >= 3].nlargest(3, "rank_gap")
    if len(outperformers) > 0:
        states_list = [{"state": r["state"], "slug": slug_map.get(r["state"], ""), "gap": int(r["rank_gap"])} for _, r in outperformers.iterrows()]
        findings.append({
            "type": "outperformer",
            "title": "Index Outperformers vs GDP",
            "detail": f"{states_list[0]['state']} ranks {states_list[0]['gap']} positions higher on our activity "
                     f"index than on official GSDP, suggesting stronger recent economic momentum.",
            "states": [s["slug"] for s in states_list],
        })

    # 3. Top underperformers (GSDP >> index)
    underperformers = latest[latest["rank_gap"] <= -3].nsmallest(3, "rank_gap")
    if len(underperformers) > 0:
        states_list = [{"state": r["state"], "slug": slug_map.get(r["state"], ""), "gap": int(r["rank_gap"])} for _, r in underperformers.iterrows()]
        findings.append({
            "type": "underperformer",
            "title": "GDP Outperformers vs Index",
            "detail": f"{states_list[0]['state']} ranks {abs(states_list[0]['gap'])} positions higher on official "
                     f"GSDP than on our index, possibly due to agriculture or informal sector activity.",
            "states": [s["slug"] for s in states_list],
        })

    # 4. Fastest COVID recoverer
    recovered = latest[latest["recovery_speed"].notna() & (latest["recovery_speed"] > 0)]
    if len(recovered) > 0:
        fastest = recovered.nsmallest(1, "recovery_speed").iloc[0]
        findings.append({
            "type": "fastest_recovery",
            "title": "Fastest COVID Recovery",
            "detail": f"{fastest['state']} recovered to pre-COVID activity levels in {int(fastest['recovery_speed'])} year(s) "
                     f"after FY 2020-21.",
            "states": [slug_map.get(fastest["state"], "")],
        })

    # 5. Most unbalanced state
    divergent = latest[latest["divergence_score"].notna()].nlargest(1, "divergence_score")
    if len(divergent) > 0:
        d = divergent.iloc[0]
        if d["divergence_score"] > 1.0:  # Only highlight if meaningfully unbalanced
            findings.append({
                "type": "most_unbalanced",
                "title": "Most Unbalanced Economic Profile",
                "detail": f"{d['state']} shows the widest gap between its strongest component "
                         f"({d['strongest_component']}) and weakest ({d['weakest_component']}), "
                         f"with a divergence score of {d['divergence_score']:.2f}.",
                "states": [slug_map.get(d["state"], "")],
            })

    # 6. Strongest GSDP correlation
    if correlations:
        comp_corrs = {k: v for k, v in correlations.items()
                     if isinstance(v, dict) and v.get("r") is not None}
        if comp_corrs:
            best = max(comp_corrs.items(), key=lambda x: abs(x[1]["r"]))
            comp_name = best[0].replace("_gsdp", "").replace("_", " ").title()
            findings.append({
                "type": "strongest_correlation",
                "title": f"{comp_name} Most Correlated with GDP",
                "detail": f"{comp_name} shows the strongest association with state GSDP "
                         f"(r={best[1]['r']:.2f}, p<{max(best[1]['p'], 0.001):.3f}), "
                         f"suggesting states with higher {comp_name.lower()} also have higher economic output. "
                         f"Correlation does not imply causation.",
                "states": [],
            })

    print(f"  Generated {len(findings)} key findings")
    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 70)
    print("Insights Computation")
    print("=" * 70)

    # Load index data (annual only for insights)
    print("\nLoading computed index...")
    df_all = pd.read_parquet(DATA_PROCESSED / "index_computed.parquet")
    df = df_all[df_all["period_type"] == "annual"].copy()
    print(f"  {len(df)} annual rows, {df['state'].nunique()} states")

    # Step 1: Growth metrics
    df = compute_growth_metrics(df)

    # Step 2: Momentum
    df = compute_momentum(df)

    # Step 3: COVID recovery
    df = compute_covid_recovery(df)

    # Step 4: GSDP comparison
    df = merge_gsdp(df)

    # Step 5: Component diagnostics
    df = compute_diagnostics(df)

    # Step 6: BRAP
    df = merge_brap(df)

    # Step 7: Correlations
    correlations = compute_correlations(df)

    # Step 8: OLS Regression
    regression = compute_regression(df)

    # Step 9: Panel FE regression
    panel_fe = compute_panel_fe_regression(df)
    if not panel_fe.get("skipped"):
        regression["panel_fe"] = panel_fe

    # Step 10: Log-log regression
    log_log = compute_log_log_regression(df)
    if not log_log.get("skipped"):
        regression["log_log"] = log_log

    # Step 11: Lagged panel correlation
    lagged = compute_lagged_correlation(df)
    if not lagged.get("skipped"):
        regression["lagged"] = lagged

    # Step 12: PCA weights check
    pca_result = compute_pca_weights(df)
    if not pca_result.get("skipped"):
        regression["pca"] = pca_result

    # Step 13: State gap explanations
    gap_explanations = generate_gap_explanations(df)

    # Step 14: Regional analysis
    regional_analysis = compute_regional_analysis(df)

    # Step 15: Key findings
    key_findings = generate_key_findings(df, correlations)

    # Save insights parquet
    print("\nSaving insights_computed.parquet...")
    OUTPUT_PARQUET = DATA_PROCESSED / "insights_computed.parquet"
    # Select only serializable columns (drop diagnostic_text for parquet, put in JSON)
    parquet_cols = [c for c in df.columns if c not in ["diagnostic_text", "gap_label",
                                                        "momentum_tier", "recovery_fy",
                                                        "strongest_component", "weakest_component",
                                                        "brap_category"]]
    # Actually, keep string cols too - parquet handles them fine
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"  Saved {len(df)} rows to {OUTPUT_PARQUET}")

    # Save regression JSON
    print("\nSaving regression.json...")
    write_json(regression, PUBLIC_DATA / "regression.json")

    # Save insights JSON
    print("\nBuilding insights.json...")
    # Pick latest FY with actual composite scores (exclude partial years like 2025-26)
    scored_fys = df[df["composite_score"].notna()]["fiscal_year"].unique()
    latest_fy = sorted(scored_fys)[-1] if len(scored_fys) > 0 else sorted(df["fiscal_year"].unique())[-1]
    latest = df[df["fiscal_year"] == latest_fy].copy()
    print(f"  Using FY {latest_fy} ({len(latest)} states)")

    meta = load_state_metadata()
    slug_map = {row["canonical_name"]: row["slug"] for _, row in meta.iterrows()}

    # Growth rankings
    growth_rankings = []
    for _, row in latest.sort_values("rank").iterrows():
        growth_rankings.append({
            "state": row["state"],
            "slug": slug_map.get(row["state"], ""),
            "rank": int(row["rank"]) if pd.notna(row["rank"]) else None,
            "momentum_tier": row.get("momentum_tier"),
            "rank_momentum_3yr": row.get("rank_momentum_3yr"),
            "gst_yoy_pct": row.get("gst_yoy_pct"),
            "elec_yoy_pct": row.get("elec_yoy_pct"),
            "credit_yoy_pct": row.get("credit_yoy_pct"),
            "epfo_yoy_pct": row.get("epfo_yoy_pct"),
        })

    # GSDP comparison
    gsdp_comparison = []
    for _, row in latest[latest["gsdp_rank"].notna()].sort_values("rank").iterrows():
        gsdp_comparison.append({
            "state": row["state"],
            "slug": slug_map.get(row["state"], ""),
            "index_rank": int(row["rank"]) if pd.notna(row["rank"]) else None,
            "gsdp_rank": int(row["gsdp_rank"]),
            "rank_gap": int(row["rank_gap"]) if pd.notna(row["rank_gap"]) else None,
            "gap_label": row.get("gap_label"),
        })

    # COVID recovery
    covid_recovery = []
    for _, row in latest.sort_values("rank").iterrows():
        if pd.notna(row.get("covid_dip")):
            covid_recovery.append({
                "state": row["state"],
                "slug": slug_map.get(row["state"], ""),
                "covid_dip": row["covid_dip"],
                "recovery_speed": row.get("recovery_speed"),
                "recovery_fy": row.get("recovery_fy"),
                "never_recovered": pd.isna(row.get("recovery_speed")),
                "pre_covid_declining": bool(row.get("pre_covid_declining", False)),
            })
    # Sort: recovered states first (by speed), then never-recovered
    covid_recovery.sort(key=lambda x: (x["never_recovered"], x.get("recovery_speed") or 999))

    # Component diagnostics (keyed by slug)
    component_diagnostics = {}
    for _, row in latest.iterrows():
        slug = slug_map.get(row["state"], "")
        if slug:
            component_diagnostics[slug] = {
                "strongest": row.get("strongest_component"),
                "weakest": row.get("weakest_component"),
                "divergence_score": row.get("divergence_score"),
                "diagnostic": row.get("diagnostic_text"),
                "brap_category": row.get("brap_category"),
            }

    insights_data = {
        "generated_at": datetime.now().isoformat(),
        "latest_fy": latest_fy,
        "key_findings": key_findings,
        "correlations": correlations,
        "growth_rankings": growth_rankings,
        "gsdp_comparison": gsdp_comparison,
        "covid_recovery": covid_recovery,
        "component_diagnostics": component_diagnostics,
        "gap_explanations": gap_explanations,
        "regional_analysis": regional_analysis,
    }

    write_json(insights_data, PUBLIC_DATA / "insights.json")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Key findings: {len(key_findings)}")
    print(f"Growth rankings: {len(growth_rankings)} states")
    print(f"GSDP comparison: {len(gsdp_comparison)} states")
    print(f"COVID recovery: {len(covid_recovery)} states")
    print(f"Correlations computed: {len([v for v in correlations.values() if isinstance(v, dict) and v.get('r') is not None])}")
    if not regression.get("skipped"):
        cs = regression.get("cross_sectional", {})
        for fy, data in cs.items():
            print(f"Regression (FY {fy}): R2={data['r_squared']}, F={data['f_statistic']}, N={data['n']}")
    if "panel_fe" in regression:
        pf = regression["panel_fe"]
        print(f"Panel FE: within-R2={pf.get('within_r_squared')}, N={pf.get('n')}")
    if "log_log" in regression:
        ll = regression["log_log"]
        cs_ll = ll.get("cross_sectional", {})
        print(f"Log-log: R2={cs_ll.get('r_squared')}")
    if "pca" in regression:
        pc = regression["pca"]
        print(f"PCA: PC1 explains {pc.get('pc1_variance_explained_pct')}%, rank corr={pc.get('rank_correlation_with_equal_weights')}")
    if gap_explanations:
        n_gaps = len(gap_explanations.get("all", {}))
        print(f"Gap explanations: {n_gaps} states")
    if regional_analysis:
        n_regions = len(regional_analysis.get("regions", {}))
        print(f"Regional analysis: {n_regions} regions")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
