"use client";

import { useState } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import type { RankingEntry, PerformanceRankingEntry, TrendChartPoint } from "@/lib/types";
import RankingsTable from "./RankingsTable";

const TrendChart = dynamic(() => import("@/components/charts/TrendChart"), {
  ssr: false,
  loading: () => <div className="h-80 bg-gray-50 rounded-lg animate-pulse" />,
});

const ComponentBreakdown = dynamic(
  () => import("@/components/charts/ComponentBreakdown"),
  {
    ssr: false,
    loading: () => <div className="h-80 bg-gray-50 rounded-lg animate-pulse" />,
  }
);

function ZScoreBar({ value }: { value: number | null }) {
  if (value === null) return <span className="text-gray-300 text-xs">--</span>;
  const clamped = Math.max(-3, Math.min(3, value));
  const pct = ((clamped + 3) / 6) * 100;
  const color = value >= 0 ? "bg-emerald-400" : "bg-red-400";
  return (
    <div className="flex items-center gap-1">
      <div className="w-16 h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full ${color} rounded-full`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs text-gray-500 w-8 text-right tabular-nums">
        {value.toFixed(1)}
      </span>
    </div>
  );
}

function GapBadge({ gap }: { gap: number | null }) {
  if (gap === null) return <span className="text-gray-400 text-xs">&mdash;</span>;
  const color =
    gap > 0
      ? "text-emerald-700 bg-emerald-50"
      : gap < 0
        ? "text-amber-700 bg-amber-50"
        : "text-gray-600 bg-gray-50";
  return (
    <span className={`inline-block text-xs font-medium px-1.5 py-0.5 rounded ${color} tabular-nums`}>
      {gap > 0 ? "+" : ""}{gap}
    </span>
  );
}

function getTierColor(rank: number, total: number): string {
  const third = Math.ceil(total / 3);
  if (rank <= third) return "bg-emerald-50";
  if (rank <= third * 2) return "bg-amber-50";
  return "bg-red-50";
}

interface Props {
  rankings: RankingEntry[];
  performance: PerformanceRankingEntry[];
  trendData: TrendChartPoint[];
  trendSlugs: string[];
  stateNameMap: Record<string, string>;
}

export default function RankingsToggle({
  rankings,
  performance,
  trendData,
  trendSlugs,
  stateNameMap,
}: Props) {
  const [mode, setMode] = useState<"activity" | "performance">("activity");
  const [selectedSlug, setSelectedSlug] = useState(rankings[0]?.slug || "");

  const selectedRanking = rankings.find((r) => r.slug === selectedSlug);

  return (
    <div className="space-y-8">
      {/* Toggle */}
      <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1 w-fit">
        <button
          onClick={() => setMode("activity")}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            mode === "activity"
              ? "bg-white text-gray-900 shadow-sm"
              : "text-gray-600 hover:text-gray-900"
          }`}
        >
          Activity Index
        </button>
        <button
          onClick={() => setMode("performance")}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            mode === "performance"
              ? "bg-white text-gray-900 shadow-sm"
              : "text-gray-600 hover:text-gray-900"
          }`}
        >
          Performance Index
          <span className="ml-1.5 text-xs text-blue-600 font-normal">(per capita)</span>
        </button>
      </div>

      {mode === "activity" ? (
        <>
          {/* Activity: existing table + charts */}
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

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
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
            <section>
              <h2 className="text-xl font-bold text-gray-900 mb-4">
                Component Breakdown
                {selectedRanking && (
                  <span className="text-base font-normal text-gray-500 ml-2">
                    &mdash; {selectedRanking.state}
                  </span>
                )}
              </h2>
              {selectedRanking && <ComponentBreakdown ranking={selectedRanking} />}
            </section>
          </div>
        </>
      ) : (
        <>
          {/* Performance Index table */}
          <section>
            <div className="mb-4">
              <h2 className="text-xl font-bold text-gray-900">
                Performance Rankings (Per Capita)
              </h2>
              <p className="text-sm text-gray-600 mt-1">
                Same four indicators, normalized by population. Removes the size bias that makes the Activity Index
                83% correlated with population.
              </p>
            </div>

            {/* Desktop table */}
            <div className="hidden md:block overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-12">
                      #
                    </th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      State
                    </th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Score
                    </th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Activity Rank
                    </th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Gap
                    </th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      GST/cap
                    </th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Elec/cap
                    </th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Credit/cap
                    </th>
                    <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      EPFO/cap
                    </th>
                    <th className="px-3 py-2 w-10" />
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {performance.map((r) => (
                    <tr
                      key={r.slug}
                      className={`hover:bg-gray-100 transition-colors ${getTierColor(r.perf_rank, performance.length)}`}
                    >
                      <td className="px-3 py-2 text-sm font-semibold text-gray-700 tabular-nums">
                        {r.perf_rank}
                      </td>
                      <td className="px-3 py-2 text-sm font-medium text-gray-900">
                        {r.state}
                      </td>
                      <td className="px-3 py-2 text-sm font-semibold text-gray-800 tabular-nums">
                        {r.perf_score.toFixed(2)}
                      </td>
                      <td className="px-3 py-2 text-sm text-gray-500 tabular-nums">
                        #{r.activity_rank}
                      </td>
                      <td className="px-3 py-2">
                        <GapBadge gap={r.activity_perf_gap} />
                      </td>
                      <td className="px-3 py-2">
                        <ZScoreBar value={r.gst_pc_zscore} />
                      </td>
                      <td className="px-3 py-2">
                        <ZScoreBar value={r.electricity_pc_zscore} />
                      </td>
                      <td className="px-3 py-2">
                        <ZScoreBar value={r.credit_pc_zscore} />
                      </td>
                      <td className="px-3 py-2">
                        <ZScoreBar value={r.epfo_pc_zscore} />
                      </td>
                      <td className="px-3 py-2">
                        <Link
                          href={`/states/${r.slug}`}
                          className="text-gray-400 hover:text-blue-600 transition-colors"
                          title={`View ${r.state} details`}
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                          </svg>
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Mobile cards */}
            <div className="md:hidden space-y-2">
              {performance.map((r) => (
                <Link
                  key={r.slug}
                  href={`/states/${r.slug}`}
                  className={`block p-3 rounded-lg border border-gray-200 transition-colors hover:border-gray-300 ${getTierColor(r.perf_rank, performance.length)}`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-gray-200 text-xs font-bold text-gray-700">
                        {r.perf_rank}
                      </span>
                      <span className="text-sm font-semibold text-gray-900">{r.state}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-bold text-gray-800 tabular-nums">
                        {r.perf_score.toFixed(2)}
                      </span>
                      <GapBadge gap={r.activity_perf_gap} />
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          </section>

          {/* Size dependency note */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-sm text-blue-800">
              <strong>Why per capita?</strong> The Activity Index is 83% correlated with population
              (Spearman rho = -0.83). The Performance Index removes this: rho drops to ~0.11.
              City-states (Delhi, Chandigarh, Goa) naturally rank higher per capita; large states
              (UP, Bihar) drop. Neither view is &ldquo;right&rdquo; &mdash; they answer different questions.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
