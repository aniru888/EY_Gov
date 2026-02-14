# State Economic Activity Index

A composite index tracking economic activity across Indian states, inspired by the [Li Keqiang Index](https://en.wikipedia.org/wiki/Li_Keqiang_index) — the idea that hard data like electricity consumption, freight volumes, and bank lending reveal more about real economic activity than headline GDP alone.

**[Live Dashboard](https://eygovindex.vercel.app)** &nbsp;&middot;&nbsp; Built for EY Government Advisory

<!-- Screenshot: replace with actual screenshot -->
<!-- ![Dashboard Screenshot](docs/screenshot.png) -->

---

## What is this?

India's state-level GDP estimates lag by 1-2 years and are frequently revised. This project constructs a near-real-time composite index from four independently collected, hard-to-manipulate data sources: GST tax collections, electricity consumption, bank credit growth, and formal employment (EPFO payroll). The result is a single score per state per fiscal year that tracks observable economic activity patterns.

This is a **descriptive composite indicator** — it tracks patterns in observable data, not a predictive model of GDP.

## Key Numbers

| Metric | Value |
|--------|-------|
| States & UTs ranked | 34 |
| Fiscal years covered | 9 (FY 2017-18 to FY 2025-26) |
| Index components | 4 |
| Static JSON output | ~1 MB across 39 files |
| Pipeline runtime | 3.5 seconds end-to-end |
| Dashboard pages | 42 (statically generated) |

## The Four Components

Each component receives equal weight (0.25) in the composite score. Values are z-score normalized within each fiscal year.

| Component | Source | Frequency | Unit | Coverage |
|-----------|--------|-----------|------|----------|
| **GST Collections** | [gst.gov.in](https://gst.gov.in) | Monthly | INR crore | 2017-18 onwards |
| **Electricity Demand** | [POSOCO/Grid India](https://robbieandrew.github.io/india/) via Robbie Andrew | Daily (aggregated monthly) | Million Units (MU) | 2013 onwards |
| **Bank Credit (YoY)** | [RBI Handbook of Statistics](https://rbi.org.in), Table 156 | Annual | INR crore (year-over-year change) | 2004 onwards |
| **EPFO Net Payroll** | [epfindia.gov.in](https://www.epfindia.gov.in) | Annual | Number of persons | 2017-18 onwards |

## Top 5 States (FY 2024-25)

| Rank | State | Composite Score |
|------|-------|----------------|
| 1 | Maharashtra | 4.01 |
| 2 | Tamil Nadu | 1.38 |
| 3 | Gujarat | 1.32 |
| 4 | Karnataka | 1.16 |
| 5 | Uttar Pradesh | 1.05 |

## Pages

| Route | Description |
|-------|-------------|
| `/` | Rankings table (sortable, color-coded tiers), trend chart, radar breakdown |
| `/methodology` | Full methodology — data sources, normalization, weights, limitations |
| `/states/[slug]` | State detail — score trend, component breakdown, peer comparison (36 states) |
| `/compare` | Side-by-side multi-state comparison with component-level drill-down |

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Framework** | [Next.js](https://nextjs.org) 16.1.6 (App Router, Turbopack) |
| **UI** | React 19, [Tailwind CSS](https://tailwindcss.com) v4 |
| **Charts** | [Recharts](https://recharts.org) 3.7.0 |
| **Data Pipeline** | Python 3.10+, pandas, numpy, scipy, openpyxl |
| **Data Storage** | Static JSON files (no database) |
| **Deployment** | [Vercel](https://vercel.com) |

## Project Structure

```
state-economic-index/
├── scripts/                  # Python data pipeline (run in order)
│   ├── 01_fetch_gst.py       # Download GST state collections
│   ├── 02_fetch_electricity.py
│   ├── 03_fetch_rbi.py       # Download RBI bank credit data
│   ├── 04_fetch_epfo.py      # Download EPFO payroll data
│   ├── 05_clean_and_merge.py # Standardize names, merge on state x time
│   ├── 06_compute_index.py   # Z-score normalization, composite index
│   ├── 07_generate_json.py   # Produce 39 JSON files for frontend
│   ├── run_pipeline.py       # Run all steps in sequence
│   └── utils.py              # State name mapping, shared helpers
├── src/
│   ├── app/                  # Next.js App Router pages
│   │   ├── page.tsx          # Dashboard home
│   │   ├── methodology/      # Methodology explainer
│   │   ├── states/[slug]/    # Per-state detail (36 states)
│   │   └── compare/          # Multi-state comparison
│   ├── components/           # 19 React components
│   │   ├── charts/           # TrendChart, ComponentBreakdown, PeerComparison, ...
│   │   ├── rankings/         # RankingsTable
│   │   ├── methodology/      # PipelineDiagram, TableOfContents
│   │   ├── comparison/       # ComparisonTable
│   │   └── common/           # Header, Footer, MetricCard, Breadcrumbs, ...
│   └── lib/                  # Types, constants, data loading utilities
├── public/data/              # Static JSON consumed by frontend
│   ├── rankings.json         # Latest fiscal year rankings
│   ├── trends.json           # Multi-year time series (all states)
│   ├── metadata.json         # Data freshness, methodology metadata
│   └── states/               # 36 per-state detail files
├── data/
│   ├── raw/                  # Downloaded source files (read-only)
│   └── processed/            # Cleaned parquet files, computed index
└── requirements.txt          # Python dependencies
```

## Data Pipeline

Seven scripts run in strict sequence. Each is idempotent — re-running produces the same output.

```
01_fetch_gst → 02_fetch_electricity → 03_fetch_rbi → 04_fetch_epfo
                                                          ↓
                        05_clean_and_merge (standardize state names, merge)
                                                          ↓
                        06_compute_index (z-score normalize, equal-weight composite)
                                                          ↓
                        07_generate_json (39 JSON files → public/data/)
```

Raw government data goes in, static JSON comes out. Python never touches React. React never runs pandas.

## Getting Started

### Frontend (dashboard)

```bash
cd state-economic-index
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The dashboard reads pre-built JSON from `public/data/`.

### Data Pipeline (rebuild from source data)

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
```

This re-runs all 7 pipeline steps and regenerates the JSON files in `public/data/`.

> **Note**: EPFO data requires manual browser download due to WAF restrictions. See `scripts/04_fetch_epfo.py` for details.

### Build for production

```bash
npm run build    # Statically generates all 42 pages
npm run start    # Serve locally
```

## Data Sources & Licenses

| Source | URL | License |
|--------|-----|---------|
| GST State Collections | [gst.gov.in](https://gst.gov.in/download/gststatistics) | Government of India Open Data |
| Electricity Demand (POSOCO) | [robbieandrew.github.io/india](https://robbieandrew.github.io/india/) | CC-BY-4.0 |
| RBI Bank Credit | [rbi.org.in](https://rbi.org.in) — Handbook Table 156 | Government of India Open Data |
| EPFO Net Payroll | [epfindia.gov.in](https://www.epfindia.gov.in/site_en/Estimate_of_Payroll.php) | Government of India Open Data |

### Attribution

Electricity demand data sourced from [Robbie Andrew's compilation](https://robbieandrew.github.io/india/) of POSOCO/Grid India data, licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

## Methodology

The index uses **cross-sectional z-score normalization** within each fiscal year — each state's raw value is compared to the mean and standard deviation across all states in that year. This prevents older periods with lower absolute values from dominating.

The four z-scores are combined with **equal weights (0.25 each)** into a single composite score. States with fewer than 3 available components are excluded. Partial indices (3 of 4 components) are flagged.

For the full methodology including data collection details, normalization math, handling of missing data, and known limitations, see the [Methodology page](https://eygovindex.vercel.app/methodology).
