"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import type { InsightsData, GsdpComparisonEntry } from "@/lib/types";

const GsdpScatterPlot = dynamic(
  () => import("@/components/charts/GsdpScatterPlot"),
  {
    ssr: false,
    loading: () => (
      <div className="h-80 bg-gray-50 rounded-lg animate-pulse" />
    ),
  }
);

const CovidRecoveryChart = dynamic(
  () => import("@/components/charts/CovidRecoveryChart"),
  {
    ssr: false,
    loading: () => (
      <div className="h-80 bg-gray-50 rounded-lg animate-pulse" />
    ),
  }
);

type GrowthSort =
  | "rank"
  | "rank_momentum_3yr"
  | "gst_yoy_pct"
  | "epfo_yoy_pct";

function MomentumBadge({ tier }: { tier: string | null }) {
  if (!tier) return null;
  const colors: Record<string, string> = {
    rising: "bg-emerald-100 text-emerald-700",
    declining: "bg-red-100 text-red-700",
    stable: "bg-gray-100 text-gray-600",
  };
  return (
    <span
      className={`inline-block text-xs font-medium px-2 py-0.5 rounded-full ${
        colors[tier] || colors.stable
      }`}
    >
      {tier}
    </span>
  );
}

function pctFmt(val: number | null): string {
  if (val === null || val === undefined) return "--";
  const sign = val >= 0 ? "+" : "";
  return `${sign}${val.toFixed(1)}%`;
}

interface Props {
  insights: InsightsData;
}

export default function InsightsClient({ insights }: Props) {
  const [growthSort, setGrowthSort] = useState<GrowthSort>("rank_momentum_3yr");
  const [growthAsc, setGrowthAsc] = useState(false);

  const sortedGrowth = useMemo(() => {
    return [...insights.growth_rankings].sort((a, b) => {
      const av = a[growthSort];
      const bv = b[growthSort];
      if (av === null && bv === null) return 0;
      if (av === null) return 1;
      if (bv === null) return -1;
      return growthAsc
        ? (av as number) - (bv as number)
        : (bv as number) - (av as number);
    });
  }, [insights.growth_rankings, growthSort, growthAsc]);

  function handleGrowthSort(key: GrowthSort) {
    if (growthSort === key) {
      setGrowthAsc(!growthAsc);
    } else {
      setGrowthSort(key);
      setGrowthAsc(key === "rank");
    }
  }

  return (
    <div className="space-y-12">
      {/* GSDP Scatter Plot */}
      {insights.gsdp_comparison.length > 0 && (
        <section>
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Index Rank vs GDP Rank
          </h2>
          <p className="text-sm text-gray-600 mb-4">
            States above the diagonal line rank higher on our activity index
            than on official GSDP. States below rank lower.
          </p>
          <GsdpScatterPlot data={insights.gsdp_comparison} />
        </section>
      )}

      {/* Fastest Growing States */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Growth & Momentum Rankings
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border border-gray-200 rounded-lg">
            <thead className="bg-gray-50">
              <tr>
                <th
                  className="text-left px-3 py-2 font-medium text-gray-500 text-xs uppercase cursor-pointer hover:text-gray-900"
                  onClick={() => handleGrowthSort("rank")}
                >
                  State{" "}
                  {growthSort === "rank" && (growthAsc ? "\u2191" : "\u2193")}
                </th>
                <th className="text-center px-3 py-2 font-medium text-gray-500 text-xs uppercase">
                  Rank
                </th>
                <th className="text-center px-3 py-2 font-medium text-gray-500 text-xs uppercase">
                  Momentum
                </th>
                <th
                  className="text-right px-3 py-2 font-medium text-gray-500 text-xs uppercase cursor-pointer hover:text-gray-900"
                  onClick={() => handleGrowthSort("rank_momentum_3yr")}
                >
                  3yr Change{" "}
                  {growthSort === "rank_momentum_3yr" &&
                    (growthAsc ? "\u2191" : "\u2193")}
                </th>
                <th
                  className="text-right px-3 py-2 font-medium text-gray-500 text-xs uppercase cursor-pointer hover:text-gray-900"
                  onClick={() => handleGrowthSort("gst_yoy_pct")}
                >
                  GST YoY{" "}
                  {growthSort === "gst_yoy_pct" &&
                    (growthAsc ? "\u2191" : "\u2193")}
                </th>
                <th
                  className="text-right px-3 py-2 font-medium text-gray-500 text-xs uppercase cursor-pointer hover:text-gray-900"
                  onClick={() => handleGrowthSort("epfo_yoy_pct")}
                >
                  EPFO YoY{" "}
                  {growthSort === "epfo_yoy_pct" &&
                    (growthAsc ? "\u2191" : "\u2193")}
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {sortedGrowth.map((r) => (
                <tr key={r.slug} className="hover:bg-gray-50">
                  <td className="px-3 py-2 font-medium text-gray-900">
                    <Link
                      href={`/states/${r.slug}`}
                      className="hover:text-blue-600"
                    >
                      {r.state}
                    </Link>
                  </td>
                  <td className="px-3 py-2 text-center tabular-nums text-gray-600">
                    {r.rank ?? "--"}
                  </td>
                  <td className="px-3 py-2 text-center">
                    <MomentumBadge tier={r.momentum_tier} />
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums">
                    {r.rank_momentum_3yr != null ? (
                      <span
                        className={
                          r.rank_momentum_3yr >= 3
                            ? "text-emerald-600 font-medium"
                            : r.rank_momentum_3yr <= -3
                              ? "text-red-600 font-medium"
                              : "text-gray-500"
                        }
                      >
                        {r.rank_momentum_3yr > 0 ? "+" : ""}
                        {r.rank_momentum_3yr}
                      </span>
                    ) : (
                      <span className="text-gray-300">--</span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-gray-600">
                    {pctFmt(r.gst_yoy_pct)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-gray-600">
                    {pctFmt(r.epfo_yoy_pct)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* COVID Recovery */}
      {insights.covid_recovery.length > 0 && (
        <section>
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            COVID Recovery
          </h2>
          <p className="text-sm text-gray-600 mb-4">
            How quickly states recovered to pre-COVID (FY 2019-20) activity
            levels after the FY 2020-21 dip.
          </p>
          <CovidRecoveryChart data={insights.covid_recovery} />
        </section>
      )}
    </div>
  );
}
