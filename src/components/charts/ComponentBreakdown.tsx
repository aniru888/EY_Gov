"use client";

import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { RankingEntry } from "@/lib/types";

interface Props {
  ranking: RankingEntry;
}

export default function ComponentBreakdown({ ranking }: Props) {
  const data = [
    {
      component: "GST",
      value: ranking.gst_zscore ?? 0,
      fullName: "GST Collections",
    },
    {
      component: "Electricity",
      value: ranking.electricity_zscore ?? 0,
      fullName: "Electricity Demand",
    },
    {
      component: "Credit",
      value: ranking.credit_zscore ?? 0,
      fullName: "Bank Credit (YoY)",
    },
    {
      component: "EPFO",
      value: ranking.epfo_zscore ?? 0,
      fullName: "EPFO Payroll",
    },
  ];

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="text-center mb-2">
        <span className="text-2xl font-bold text-gray-900 tabular-nums">
          {ranking.composite_score.toFixed(2)}
        </span>
        <span className="text-sm text-gray-500 ml-2">composite score</span>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <RadarChart data={data} cx="50%" cy="50%" outerRadius="70%">
          <PolarGrid />
          <PolarAngleAxis dataKey="component" tick={{ fontSize: 12 }} />
          <PolarRadiusAxis
            tick={{ fontSize: 10 }}
            tickFormatter={(v: number) => v.toFixed(1)}
            domain={["auto", "auto"]}
          />
          <Tooltip
            formatter={(value, _name, props) => {
              const v = typeof value === "number" ? value.toFixed(3) : String(value);
              const label = (props?.payload as Record<string, string>)?.fullName || "";
              return [v, label];
            }}
            contentStyle={{ fontSize: 13 }}
          />
          <Radar
            dataKey="value"
            stroke="#2563eb"
            fill="#3b82f6"
            fillOpacity={0.3}
            strokeWidth={2}
          />
        </RadarChart>
      </ResponsiveContainer>
      <div className="mt-2 text-xs text-gray-500 text-center">
        Z-scores: higher = stronger relative to other states
        {ranking.n_components < 4 && (
          <span className="text-amber-600 ml-2">
            ({ranking.n_components}/4 components available)
          </span>
        )}
      </div>
    </div>
  );
}
