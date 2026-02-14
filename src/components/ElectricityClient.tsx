"use client";

import { useMemo } from "react";
import dynamic from "next/dynamic";
import type { ElectricityData } from "@/lib/types";

const ElectricityGsdpScatter = dynamic(
  () => import("@/components/charts/ElectricityGsdpScatter"),
  {
    ssr: false,
    loading: () => (
      <div className="h-[400px] bg-gray-50 rounded-lg animate-pulse" />
    ),
  }
);

const IntensityTable = dynamic(
  () => import("@/components/charts/IntensityTable"),
  {
    ssr: false,
    loading: () => (
      <div className="h-64 bg-gray-50 rounded-lg animate-pulse" />
    ),
  }
);

const MonthlyMomentumHeatmap = dynamic(
  () => import("@/components/charts/MonthlyMomentumHeatmap"),
  {
    ssr: false,
    loading: () => (
      <div className="h-64 bg-gray-50 rounded-lg animate-pulse" />
    ),
  }
);

const SeasonalityChart = dynamic(
  () => import("@/components/charts/SeasonalityChart"),
  {
    ssr: false,
    loading: () => (
      <div className="h-[350px] bg-gray-50 rounded-lg animate-pulse" />
    ),
  }
);

interface Props {
  data: ElectricityData;
}

export default function ElectricityClient({ data }: Props) {
  // Derive stateNames mapping from scatter data + elasticity rankings
  const stateNames = useMemo(() => {
    const map: Record<string, string> = {};
    for (const entry of data.electricity_vs_gsdp_scatter) {
      map[entry.slug] = entry.state;
    }
    for (const entry of data.rankings_by_elasticity) {
      map[entry.slug] = entry.state;
    }
    for (const entry of data.rankings_by_intensity) {
      map[entry.slug] = entry.state;
    }
    return map;
  }, [data]);

  // Derive top 5 slugs by electricity_mu from scatter data
  const topSlugs = useMemo(() => {
    return [...data.electricity_vs_gsdp_scatter]
      .sort((a, b) => b.electricity_mu - a.electricity_mu)
      .slice(0, 5)
      .map((d) => d.slug);
  }, [data]);

  // Derive elasticities mapping keyed by slug
  const elasticities = useMemo(() => {
    const map: Record<string, { elasticity: number; label: string }> = {};
    for (const entry of data.rankings_by_elasticity) {
      map[entry.slug] = {
        elasticity: entry.elasticity,
        label: entry.label,
      };
    }
    return map;
  }, [data]);

  // Build profiles subset for heatmap and seasonality
  // (only pass profiles that have data we need)
  const heatmapProfiles = useMemo(() => {
    const result: Record<
      string,
      { monthly_growth: Array<{ month: string; yoy_pct: number }> }
    > = {};
    for (const [slug, profile] of Object.entries(data.state_profiles)) {
      if (profile.monthly_growth && profile.monthly_growth.length > 0) {
        result[slug] = { monthly_growth: profile.monthly_growth };
      }
    }
    return result;
  }, [data]);

  const seasonalityProfiles = useMemo(() => {
    const result: Record<
      string,
      { seasonality_index: Array<{ month: number; index: number }> }
    > = {};
    for (const [slug, profile] of Object.entries(data.state_profiles)) {
      if (profile.seasonality_index && profile.seasonality_index.length > 0) {
        result[slug] = { seasonality_index: profile.seasonality_index };
      }
    }
    return result;
  }, [data]);

  return (
    <div className="space-y-12">
      {/* Section 1: Scatter Plot */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Electricity Demand vs GSDP
        </h2>
        <p className="text-sm text-gray-600 mb-4">
          Cross-sectional relationship between electricity consumption and state
          GDP. States are colored by their electricity-GDP elasticity
          classification. The dashed line shows the linear regression fit (R
          {"\u00B2"} = {data.electricity_vs_gsdp_regression.r_squared.toFixed(3)}
          ).
        </p>
        <ElectricityGsdpScatter
          data={data.electricity_vs_gsdp_scatter}
          regression={data.electricity_vs_gsdp_regression}
        />
      </section>

      {/* Section 2: Intensity Table */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Electricity Intensity Rankings
        </h2>
        <p className="text-sm text-gray-600 mb-4">
          States ranked by electricity intensity (MU consumed per crore of GSDP).
          Higher intensity suggests a more electricity-dependent economic
          structure, often associated with industrial activity. Elasticity
          measures how much electricity demand grows for each unit of GDP growth.
        </p>
        <IntensityTable
          data={data.rankings_by_intensity}
          elasticities={elasticities}
        />
      </section>

      {/* Section 3: Heatmap */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Monthly Demand Momentum
        </h2>
        <p className="text-sm text-gray-600 mb-4">
          Year-over-year monthly electricity demand growth for the top 15 states
          (by average growth). Green cells indicate growing demand, red cells
          indicate contraction. Hover over any cell for the exact value.
        </p>
        <MonthlyMomentumHeatmap
          profiles={heatmapProfiles}
          stateNames={stateNames}
        />
      </section>

      {/* Section 4: Seasonality */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Seasonal Demand Patterns
        </h2>
        <p className="text-sm text-gray-600 mb-4">
          Monthly seasonality index for the top 5 states by electricity demand.
          An index of 100 represents the fiscal year average. Summer months
          (Apr--Jun) typically show peaks due to cooling demand, while winter
          months tend to be below average.
        </p>
        <SeasonalityChart
          profiles={seasonalityProfiles}
          stateNames={stateNames}
          topSlugs={topSlugs}
        />
      </section>
    </div>
  );
}
