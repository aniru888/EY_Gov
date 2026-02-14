"""
04b_fetch_population.py

Reads state population projections from data/reference/state_population.csv,
maps to canonical names, and outputs data/processed/population_clean.parquet.

Source: National Commission on Population projections (Census 2011 base),
published 2020. Inter-census projections for 2017-2025.

FY mapping convention: FY "2023-24" uses CY 2023 population.

Output schema:
  - state: str (canonical name)
  - fiscal_year: str ("2017-18", etc.)
  - population: int
"""

import pandas as pd
import sys
import io

from utils import (
    DATA_REF,
    DATA_PROCESSED,
    load_state_mapper,
    canonicalize_state,
)

# FYs we need: 2017-18 through 2024-25
# FY "2023-24" -> CY 2023 population
FY_TO_CY = {
    "2017-18": "2017",
    "2018-19": "2018",
    "2019-20": "2019",
    "2020-21": "2020",
    "2021-22": "2021",
    "2022-23": "2022",
    "2023-24": "2023",
    "2024-25": "2024",
}


def main():
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 70)
    print("Population Data: Parse Reference CSV")
    print("=" * 70)

    pop_path = DATA_REF / "state_population.csv"
    if not pop_path.exists():
        print(f"  ERROR: {pop_path} not found.")
        sys.exit(1)

    raw = pd.read_csv(pop_path)
    print(f"  Loaded {len(raw)} rows, columns: {list(raw.columns)}")

    mapper = load_state_mapper()

    rows = []
    unmatched = []

    for _, row in raw.iterrows():
        name = str(row["canonical_name"]).strip()
        canonical = canonicalize_state(name, mapper)
        if not canonical:
            unmatched.append(name)
            continue

        for fy, cy in FY_TO_CY.items():
            if cy in raw.columns:
                pop = row[cy]
                if pd.notna(pop) and pop > 0:
                    rows.append({
                        "state": canonical,
                        "fiscal_year": fy,
                        "population": int(pop),
                    })

    if unmatched:
        print(f"  WARNING: {len(unmatched)} unmatched states: {unmatched}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["state", "fiscal_year"]).reset_index(drop=True)

    # Validate
    n_states = df["state"].nunique()
    n_fys = df["fiscal_year"].nunique()
    print(f"\n  States: {n_states}")
    print(f"  Fiscal years: {sorted(df['fiscal_year'].unique())}")
    print(f"  Total rows: {len(df)}")

    # Save
    output_path = DATA_PROCESSED / "population_clean.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\n  Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Spot check
    print("\n  Spot check - Top 5 by population (FY 2024-25):")
    latest = df[df["fiscal_year"] == "2024-25"].nlargest(5, "population")
    for _, r in latest.iterrows():
        print(f"    {r['state']:25s} {r['population']:>12,}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
