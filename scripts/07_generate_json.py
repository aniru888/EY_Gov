"""
07_generate_json.py

Generates static JSON files for the Next.js dashboard.

Input:  data/processed/index_computed.parquet
Output:
  - public/data/rankings.json     - latest FY leaderboard
  - public/data/trends.json       - all states x all FYs + monthly
  - public/data/states/{slug}.json - one file per state
  - public/data/metadata.json     - source info, methodology
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime

from utils import (
    DATA_PROCESSED,
    PUBLIC_DATA,
    load_state_metadata,
)


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


def load_insights_for_enrichment():
    """Load insights data for enriching rankings and state files."""
    insights_path = DATA_PROCESSED / "insights_computed.parquet"
    if not insights_path.exists():
        print("  NOTE: insights_computed.parquet not found. Enrichment skipped.")
        return None
    return pd.read_parquet(insights_path)


def load_electricity_profiles():
    """Load electricity analysis data for enriching state files."""
    elec_path = PUBLIC_DATA / "electricity.json"
    if not elec_path.exists():
        print("  NOTE: electricity.json not found. Electricity enrichment skipped.")
        return None
    import json as json_mod
    with open(elec_path, "r", encoding="utf-8") as f:
        return json_mod.load(f)


def load_insights_json():
    """Load insights.json for gap explanations and regional analysis."""
    ins_path = PUBLIC_DATA / "insights.json"
    if not ins_path.exists():
        return None
    import json as json_mod
    with open(ins_path, "r", encoding="utf-8") as f:
        return json_mod.load(f)


def generate_rankings(df: pd.DataFrame, output_dir, insights_df=None):
    """Generate rankings.json for the latest fiscal year."""
    print("\nGenerating rankings.json...")

    annual = df[df["period_type"] == "annual"].copy()
    # Pick the latest FY that actually has ranked states
    fys_with_scores = annual[annual["composite_score"].notna()]["fiscal_year"].unique()
    latest_fy = sorted(fys_with_scores)[-1] if len(fys_with_scores) > 0 else sorted(annual["fiscal_year"].unique())[-1]
    latest = annual[
        (annual["fiscal_year"] == latest_fy) & annual["composite_score"].notna()
    ].sort_values("rank")

    # Build insights lookup for enrichment
    insights_lookup = {}
    if insights_df is not None:
        ins_latest = insights_df[insights_df["fiscal_year"] == latest_fy]
        for _, row in ins_latest.iterrows():
            insights_lookup[row["state"]] = row

    rankings = []
    for _, row in latest.iterrows():
        entry = {
            "state": row["state"],
            "slug": row["slug"],
            "region": row["region"],
            "rank": int(row["rank"]) if pd.notna(row["rank"]) else None,
            "rank_change": int(row["rank_change"]) if pd.notna(row["rank_change"]) else None,
            "composite_score": row["composite_score"],
            "gst_zscore": row["gst_zscore"],
            "electricity_zscore": row["electricity_zscore"],
            "credit_zscore": row["credit_zscore"],
            "epfo_zscore": row["epfo_zscore"],
            "n_components": int(row["n_components_scored"]),
        }
        # Enrich with insights
        ins = insights_lookup.get(row["state"])
        if ins is not None:
            entry["momentum_tier"] = ins.get("momentum_tier") if pd.notna(ins.get("momentum_tier")) else None
            entry["gsdp_rank"] = int(ins["gsdp_rank"]) if pd.notna(ins.get("gsdp_rank")) else None
            entry["rank_gap"] = int(ins["rank_gap"]) if pd.notna(ins.get("rank_gap")) else None
            entry["brap_category"] = ins.get("brap_category") if pd.notna(ins.get("brap_category")) else None
        rankings.append(entry)

    data = {
        "fiscal_year": latest_fy,
        "generated_at": datetime.now().isoformat(),
        "count": len(rankings),
        "rankings": rankings,
    }

    write_json(data, output_dir / "rankings.json")


def generate_trends(df: pd.DataFrame, output_dir):
    """Generate trends.json with all states x all FYs + monthly."""
    print("\nGenerating trends.json...")

    # Annual trends
    annual = df[df["period_type"] == "annual"].copy()
    annual_data = {}
    for state in sorted(annual["state"].unique()):
        state_df = annual[annual["state"] == state].sort_values("fiscal_year")
        slug = state_df.iloc[0]["slug"] if not state_df.empty else ""
        annual_data[slug] = {
            "state": state,
            "fiscal_years": state_df["fiscal_year"].tolist(),
            "composite_score": state_df["composite_score"].tolist(),
            "rank": state_df["rank"].tolist(),
            "gst_total": state_df["gst_total"].tolist(),
            "electricity_mu": state_df["electricity_mu"].tolist(),
            "bank_credit_yoy": state_df["bank_credit_yoy"].tolist(),
            "epfo_payroll": state_df["epfo_payroll"].tolist(),
        }

    # Monthly sub-indices (GST + electricity z-scores only)
    monthly = df[df["period_type"] == "monthly"].copy()
    monthly_data = {}
    for state in sorted(monthly["state"].unique()):
        state_df = monthly[monthly["state"] == state].sort_values("month")
        slug = state_df.iloc[0]["slug"] if not state_df.empty else ""
        monthly_data[slug] = {
            "state": state,
            "months": state_df["month"].tolist(),
            "gst_total": state_df["gst_total"].tolist(),
            "electricity_mu": state_df["electricity_mu"].tolist(),
            "gst_zscore": state_df["gst_zscore"].tolist(),
            "electricity_zscore": state_df["electricity_zscore"].tolist(),
        }

    data = {
        "generated_at": datetime.now().isoformat(),
        "fiscal_years": sorted(annual["fiscal_year"].unique()),
        "annual": annual_data,
        "monthly": monthly_data,
    }

    write_json(data, output_dir / "trends.json")


def generate_state_files(df: pd.DataFrame, output_dir, insights_df=None,
                         electricity_data=None, insights_json=None):
    """Generate one JSON file per state."""
    print("\nGenerating per-state JSON files...")

    states_dir = output_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    meta = load_state_metadata()
    meta_dict = {
        row["canonical_name"]: {"slug": row["slug"], "region": row["region"]}
        for _, row in meta.iterrows()
    }

    # Build insights lookup (latest scored FY per state)
    state_insights = {}
    if insights_df is not None:
        for state in insights_df["state"].unique():
            state_ins = insights_df[
                (insights_df["state"] == state) &
                insights_df["composite_score"].notna()
            ].sort_values("fiscal_year")
            if not state_ins.empty:
                state_insights[state] = state_ins.iloc[-1]

    count = 0
    for state in sorted(df["state"].unique()):
        info = meta_dict.get(state, {"slug": state.lower().replace(" ", "-"), "region": "Unknown"})
        slug = info["slug"]

        # Annual data
        annual = df[(df["state"] == state) & (df["period_type"] == "annual")].sort_values("fiscal_year")

        # Monthly data
        monthly = df[(df["state"] == state) & (df["period_type"] == "monthly")].sort_values("month")

        # Find peer states (nearest by composite score in latest FY with scores)
        peers = []
        if not annual.empty:
            scored_fys = annual[annual["composite_score"].notna()]["fiscal_year"]
            latest_fy = scored_fys.max() if not scored_fys.empty else None
            if latest_fy is not None:
                latest_row = annual[annual["fiscal_year"] == latest_fy].iloc[0]
                if pd.notna(latest_row.get("composite_score")):
                    all_latest = df[
                        (df["period_type"] == "annual") &
                        (df["fiscal_year"] == latest_fy) &
                        (df["composite_score"].notna()) &
                        (df["state"] != state)
                    ].copy()
                    all_latest["score_diff"] = abs(
                        all_latest["composite_score"] - latest_row["composite_score"]
                    )
                    nearest = all_latest.nsmallest(3, "score_diff")
                    peers = [
                        {"state": r["state"], "slug": meta_dict.get(r["state"], {}).get("slug", ""),
                         "composite_score": r["composite_score"], "rank": int(r["rank"]) if pd.notna(r["rank"]) else None}
                        for _, r in nearest.iterrows()
                    ]

        state_data = {
            "state": state,
            "slug": slug,
            "region": info["region"],
            "annual": {
                "fiscal_years": annual["fiscal_year"].tolist(),
                "composite_score": annual["composite_score"].tolist(),
                "rank": annual["rank"].tolist(),
                "rank_change": annual["rank_change"].tolist(),
                "gst_total": annual["gst_total"].tolist(),
                "electricity_mu": annual["electricity_mu"].tolist(),
                "bank_credit_yoy": annual["bank_credit_yoy"].tolist(),
                "epfo_payroll": annual["epfo_payroll"].tolist(),
                "gst_zscore": annual["gst_zscore"].tolist(),
                "electricity_zscore": annual["electricity_zscore"].tolist(),
                "credit_zscore": annual["credit_zscore"].tolist(),
                "epfo_zscore": annual["epfo_zscore"].tolist(),
                "n_components": annual["n_components_scored"].tolist(),
            },
            "monthly": {
                "months": monthly["month"].tolist(),
                "gst_total": monthly["gst_total"].tolist(),
                "electricity_mu": monthly["electricity_mu"].tolist(),
                "gst_zscore": monthly["gst_zscore"].tolist(),
                "electricity_zscore": monthly["electricity_zscore"].tolist(),
            },
            "peers": peers,
        }

        # Add insights block if available
        ins = state_insights.get(state)
        if ins is not None:
            state_data["insights"] = {
                "diagnostic_text": ins.get("diagnostic_text") if pd.notna(ins.get("diagnostic_text")) else None,
                "strongest_component": ins.get("strongest_component") if pd.notna(ins.get("strongest_component")) else None,
                "weakest_component": ins.get("weakest_component") if pd.notna(ins.get("weakest_component")) else None,
                "divergence_score": ins.get("divergence_score"),
                "momentum_tier": ins.get("momentum_tier") if pd.notna(ins.get("momentum_tier")) else None,
                "rank_momentum_3yr": ins.get("rank_momentum_3yr"),
                "gsdp_rank": int(ins["gsdp_rank"]) if pd.notna(ins.get("gsdp_rank")) else None,
                "rank_gap": int(ins["rank_gap"]) if pd.notna(ins.get("rank_gap")) else None,
                "gap_label": ins.get("gap_label") if pd.notna(ins.get("gap_label")) else None,
                "brap_category": ins.get("brap_category") if pd.notna(ins.get("brap_category")) else None,
                "covid_dip": ins.get("covid_dip"),
                "recovery_speed": ins.get("recovery_speed"),
                "gst_yoy_pct": ins.get("gst_yoy_pct"),
                "elec_yoy_pct": ins.get("elec_yoy_pct"),
                "credit_yoy_pct": ins.get("credit_yoy_pct"),
                "epfo_yoy_pct": ins.get("epfo_yoy_pct"),
            }

        # Add electricity enrichment if available
        if electricity_data and "state_profiles" in electricity_data:
            elec_profile = electricity_data["state_profiles"].get(slug)
            if elec_profile:
                state_data["electricity"] = {
                    "intensity_mu_per_crore": elec_profile.get("intensity_mu_per_crore"),
                    "national_share_pct": elec_profile.get("national_share_pct"),
                    "elasticity": elec_profile.get("elasticity"),
                    "elasticity_label": elec_profile.get("elasticity_label"),
                    "volatility_cov": elec_profile.get("volatility_cov"),
                }

        # Add gap explanation if available
        if insights_json and "gap_explanations" in insights_json:
            gap_all = insights_json["gap_explanations"].get("all", {})
            gap_entry = gap_all.get(slug)
            if gap_entry:
                state_data["gap_explanation"] = {
                    "direction": gap_entry.get("direction"),
                    "explanation": gap_entry.get("explanation"),
                    "key_drivers": gap_entry.get("key_drivers", []),
                    "key_drags": gap_entry.get("key_drags", []),
                }

        write_json(state_data, states_dir / f"{slug}.json")
        count += 1

    print(f"  Generated {count} state files")


def generate_metadata(df: pd.DataFrame, output_dir):
    """Generate metadata.json with source info and methodology."""
    print("\nGenerating metadata.json...")

    annual = df[df["period_type"] == "annual"]
    fys = sorted(annual["fiscal_year"].unique())

    data = {
        "generated_at": datetime.now().isoformat(),
        "methodology": {
            "name": "Li Keqiang Index for Indian States",
            "version": "1.0",
            "description": "Composite State Economic Activity Index using 4 hard data indicators",
            "normalization": "Cross-sectional z-scores within each fiscal year",
            "weights": "Equal weights (0.25 each)",
            "min_components": MIN_COMPONENTS,
            "components": [
                {
                    "name": "GST Collections",
                    "column": "gst_total",
                    "unit": "INR crore",
                    "type": "flow",
                    "frequency": "monthly",
                    "source": "gst.gov.in",
                },
                {
                    "name": "Electricity Demand",
                    "column": "electricity_mu",
                    "unit": "Million Units (MU)",
                    "type": "flow",
                    "frequency": "daily (aggregated to monthly)",
                    "source": "POSOCO/Grid India via Robbie Andrew",
                    "license": "CC-BY-4.0",
                },
                {
                    "name": "Bank Credit (YoY Delta)",
                    "column": "bank_credit_yoy",
                    "unit": "INR crore (year-over-year change)",
                    "type": "stock-to-flow",
                    "frequency": "annual",
                    "source": "RBI Handbook of Statistics, Table 156",
                },
                {
                    "name": "EPFO Net Payroll",
                    "column": "epfo_payroll",
                    "unit": "Number of persons",
                    "type": "flow",
                    "frequency": "annual",
                    "source": "epfindia.gov.in",
                },
            ],
        },
        "data_coverage": {
            "fiscal_years": fys,
            "latest_fy": fys[-1] if fys else None,
            "states_count": int(annual["state"].nunique()),
            "states_with_4_components": int((annual["n_components"] >= 4).sum()),
        },
    }

    write_json(data, output_dir / "metadata.json")


MIN_COMPONENTS = 3


def main():
    print("=" * 70)
    print("JSON Generation for Dashboard")
    print("=" * 70)

    # Load computed index
    print("\nLoading computed index...")
    df = pd.read_parquet(DATA_PROCESSED / "index_computed.parquet")
    print(f"  Loaded {len(df):,} rows, {df['state'].nunique()} states")

    output_dir = PUBLIC_DATA

    # Load insights data for enrichment
    insights_df = load_insights_for_enrichment()
    electricity_data = load_electricity_profiles()
    insights_json = load_insights_json()

    # Generate all JSON files
    generate_rankings(df, output_dir, insights_df)
    generate_trends(df, output_dir)
    generate_state_files(df, output_dir, insights_df, electricity_data, insights_json)
    generate_metadata(df, output_dir)

    # Verify all JSON files are valid
    print("\nVerifying JSON files...")
    import glob
    json_files = list(output_dir.glob("**/*.json"))
    valid = 0
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                json.load(f)
            valid += 1
        except json.JSONDecodeError as e:
            print(f"  INVALID: {jf} - {e}")

    print(f"  {valid}/{len(json_files)} JSON files valid")

    # File size summary
    print("\nFile sizes:")
    total_kb = 0
    for jf in sorted(json_files):
        kb = jf.stat().st_size / 1024
        total_kb += kb
    print(f"  Total: {total_kb:.1f} KB across {len(json_files)} files")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
