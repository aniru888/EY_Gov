# State Economic Activity Index for India

A Li Keqiang-style composite tracking real economic activity across 34 Indian states using four hard-to-fake indicators: GST collections, electricity demand, bank credit growth, and formal employment (EPFO payroll).

**[Live Dashboard](https://eygovindex.vercel.app)** &nbsp;&middot;&nbsp; Built for EY Government Advisory

---

## Key Findings (FY 2024-25)

- **93% of cross-state GDP variation** is explained by our four indicators (R&sup2;=0.926, F=59.3, p<0.001, N=24 states)
- **Electricity demand** has the strongest correlation with state GDP (r=0.94), followed by the composite index itself (r=0.94) and GST (r=0.90)
- **Each additional indicator adds value**: GST alone explains 80% of GDP variation; adding electricity jumps to 93%
- The **top 5** (Maharashtra, Tamil Nadu, Gujarat, Karnataka, Uttar Pradesh) are stable across all 8 fiscal years
- Several states rank significantly **higher on our index than on official GDP** &mdash; potentially reflecting formalization and recent momentum that lagging GDP estimates haven't captured

## How the Index Relates to Official GDP

Our index rankings correlate strongly with official GSDP (r=0.94) but diverge for specific states. Key reasons for gaps:

| Factor | Effect |
|--------|--------|
| **Agriculture** | GSDP captures it; our index doesn't. Farm-heavy states (Punjab, MP) rank lower on our index |
| **Formalization** | EPFO captures formal jobs that GSDP may undercount. Rapidly formalizing states outperform their GDP rank |
| **Financial hub bias** | Maharashtra's bank credit is inflated by Mumbai's HQ role |
| **Time lag** | GSDP lags 1-2 years; our index uses more recent data (GST monthly, electricity daily) |

See the [Insights page](https://eygovindex.vercel.app/insights) for the full scatter plot, regression results, and COVID recovery analysis.

## Top 5 States (FY 2024-25)

| Rank | State | Composite Score |
|------|-------|----------------|
| 1 | Maharashtra | 4.01 |
| 2 | Tamil Nadu | 1.38 |
| 3 | Gujarat | 1.32 |
| 4 | Karnataka | 1.16 |
| 5 | Uttar Pradesh | 1.05 |

## The Four Components

Each component receives equal weight (0.25) in the composite score. Values are z-score normalized within each fiscal year.

| Component | Source | Frequency | Unit | Coverage |
|-----------|--------|-----------|------|----------|
| **GST Collections** | [gst.gov.in](https://gst.gov.in) | Monthly | INR crore | 2017-18 onwards |
| **Electricity Demand** | [POSOCO/Grid India](https://robbieandrew.github.io/india/) via Robbie Andrew | Daily | Million Units (MU) | 2013 onwards |
| **Bank Credit (YoY)** | [RBI Handbook](https://rbi.org.in), Table 156 | Annual | INR crore (YoY delta) | 2004 onwards |
| **EPFO Net Payroll** | [epfindia.gov.in](https://www.epfindia.gov.in) | Annual | Persons | 2017-18 onwards |

## Dashboard Pages

| Route | Description |
|-------|-------------|
| `/` | Rankings table (sortable, color-coded tiers), trend chart, radar breakdown |
| `/insights` | Key findings, GSDP scatter plot, regression validation, COVID recovery, growth dynamics |
| `/methodology` | Data sources, normalization, statistical validation, limitations |
| `/states/[slug]` | State detail with diagnostics, growth metrics, component trends, peer comparison |
| `/compare` | Side-by-side multi-state comparison |

## Key Numbers

| Metric | Value |
|--------|-------|
| States & UTs ranked | 34 |
| Fiscal years covered | 9 (FY 2017-18 to FY 2025-26) |
| Index components | 4 |
| Static JSON output | ~1 MB across 41 files |
| Pipeline runtime | ~6 seconds end-to-end |
| Dashboard pages | ~43 (statically generated) |

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Framework** | [Next.js](https://nextjs.org) 16.1.6 (App Router, Turbopack) |
| **UI** | React 19, [Tailwind CSS](https://tailwindcss.com) v4 |
| **Charts** | [Recharts](https://recharts.org) 3.7.0 |
| **Data Pipeline** | Python 3.10+, pandas, numpy, scipy, statsmodels, openpyxl |
| **Data Storage** | Static JSON files (no database) |
| **Deployment** | [Vercel](https://vercel.com) |

## Data Pipeline

Nine scripts run in strict sequence. Each is idempotent.

```
01_fetch_gst → 02_fetch_electricity → 03_fetch_rbi → 04_fetch_epfo → 08_fetch_gsdp
                                                                          ↓
                              05_clean_and_merge (standardize names, merge)
                                                                          ↓
                              06_compute_index (z-score normalize, composite)
                                                                          ↓
                              09_compute_insights (growth, regression, diagnostics)
                                                                          ↓
                              07_generate_json (41 JSON files → public/data/)
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

This re-runs all 9 pipeline steps and regenerates the JSON files in `public/data/`.

> **Note**: EPFO data requires manual browser download due to WAF restrictions. See `scripts/04_fetch_epfo.py` for details.

### Build for production

```bash
npm run build    # Statically generates all pages
npm run start    # Serve locally
```

## Data Sources & Licenses

| Source | URL | License |
|--------|-----|---------|
| GST State Collections | [gst.gov.in](https://gst.gov.in/download/gststatistics) | Government of India Open Data |
| Electricity Demand (POSOCO) | [robbieandrew.github.io/india](https://robbieandrew.github.io/india/) | CC-BY-4.0 |
| RBI Bank Credit (Table 156) | [rbi.org.in](https://rbi.org.in) | Government of India Open Data |
| RBI GSDP (Tables 21 & 22) | [rbi.org.in](https://rbi.org.in) | Government of India Open Data |
| EPFO Net Payroll | [epfindia.gov.in](https://www.epfindia.gov.in/site_en/Estimate_of_Payroll.php) | Government of India Open Data |
| BRAP Categories | [DPIIT](https://dpiit.gov.in) via PIB press releases | Government of India Open Data |

### Attribution

Electricity demand data sourced from [Robbie Andrew's compilation](https://robbieandrew.github.io/india/) of POSOCO/Grid India data, licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

## Methodology

The index uses **cross-sectional z-score normalization** within each fiscal year. The four z-scores are combined with **equal weights (0.25 each)** into a single composite score. States with fewer than 3 available components are excluded.

Statistical validation via OLS regression shows the four indicators jointly explain 93% of cross-state GDP variation. Each component adds incremental explanatory power (partial F-tests significant). Results are sensitive to Maharashtra's inclusion and are associations, not causal claims.

For the full methodology, see the [Methodology page](https://eygovindex.vercel.app/methodology).
