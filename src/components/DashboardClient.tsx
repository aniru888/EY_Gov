"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import type { RankingEntry, TrendChartPoint } from "@/lib/types";
import RankingsTable from "@/components/rankings/RankingsTable";

const TrendChart = dynamic(() => import("@/components/charts/TrendChart"), {
  ssr: false,
  loading: () => (
    <div className="h-80 bg-gray-50 rounded-lg animate-pulse" />
  ),
});

const ComponentBreakdown = dynamic(
  () => import("@/components/charts/ComponentBreakdown"),
  {
    ssr: false,
    loading: () => (
      <div className="h-80 bg-gray-50 rounded-lg animate-pulse" />
    ),
  }
);

interface Props {
  rankings: RankingEntry[];
  trendData: TrendChartPoint[];
  trendSlugs: string[];
  stateNameMap: Record<string, string>;
}

export default function DashboardClient({
  rankings,
  trendData,
  trendSlugs,
  stateNameMap,
}: Props) {
  const [selectedSlug, setSelectedSlug] = useState(
    rankings[0]?.slug || ""
  );

  const selectedRanking = rankings.find((r) => r.slug === selectedSlug);

  return (
    <div className="space-y-8">
      {/* Rankings Table */}
      <section>
        <h2 className="text-xl font-bold text-gray-900 mb-4">
          State Rankings
        </h2>
        <RankingsTable
          rankings={rankings}
          selectedSlug={selectedSlug}
          onSelectState={setSelectedSlug}
        />
      </section>

      {/* Charts side by side on large screens */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Trend Chart */}
        <section>
          <h2 className="text-xl font-bold text-gray-900 mb-4">
            Composite Score Trends
          </h2>
          <TrendChart
            data={trendData}
            slugs={trendSlugs}
            stateNameMap={stateNameMap}
          />
        </section>

        {/* Component Breakdown */}
        <section>
          <h2 className="text-xl font-bold text-gray-900 mb-4">
            Component Breakdown
            {selectedRanking && (
              <span className="text-base font-normal text-gray-500 ml-2">
                &mdash; {selectedRanking.state}
              </span>
            )}
          </h2>
          {selectedRanking && (
            <ComponentBreakdown ranking={selectedRanking} />
          )}
        </section>
      </div>
    </div>
  );
}
