"use client";

import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";

interface SeasonalityPoint {
  month: number;
  index: number;
}

interface StateProfile {
  seasonality_index: SeasonalityPoint[];
}

interface Props {
  profiles: Record<string, StateProfile>;
  stateNames: Record<string, string>;
  topSlugs: string[];
}

const COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed"];

/**
 * Fiscal year month order: Apr(4), May(5), ..., Dec(12), Jan(1), Feb(2), Mar(3)
 * The data has month 1-12 where 1=Jan. We reorder to fiscal year.
 */
const FISCAL_ORDER = [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3];
const MONTH_LABELS: Record<number, string> = {
  1: "Jan",
  2: "Feb",
  3: "Mar",
  4: "Apr",
  5: "May",
  6: "Jun",
  7: "Jul",
  8: "Aug",
  9: "Sep",
  10: "Oct",
  11: "Nov",
  12: "Dec",
};

interface ChartRow {
  month_label: string;
  month_num: number;
  [slug: string]: string | number;
}

export default function SeasonalityChart({
  profiles,
  stateNames,
  topSlugs,
}: Props) {
  const chartData = useMemo(() => {
    // Build seasonality lookup per state
    const stateSeasonality: Record<string, Record<number, number>> = {};
    for (const slug of topSlugs) {
      const profile = profiles[slug];
      if (!profile?.seasonality_index) continue;
      stateSeasonality[slug] = {};
      for (const pt of profile.seasonality_index) {
        stateSeasonality[slug][pt.month] = pt.index;
      }
    }

    // Build rows in fiscal year order
    const rows: ChartRow[] = FISCAL_ORDER.map((monthNum) => {
      const row: ChartRow = {
        month_label: MONTH_LABELS[monthNum] || String(monthNum),
        month_num: monthNum,
      };
      for (const slug of topSlugs) {
        row[slug] = stateSeasonality[slug]?.[monthNum] ?? 0;
      }
      return row;
    });

    return rows;
  }, [profiles, topSlugs]);

  // Filter to slugs that actually have data
  const activeSlugs = topSlugs.filter(
    (slug) => profiles[slug]?.seasonality_index?.length > 0
  );

  if (activeSlugs.length === 0) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-8 text-center text-gray-500">
        No seasonality data available.
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <ResponsiveContainer width="100%" height={350}>
        <LineChart
          data={chartData}
          margin={{ top: 10, right: 20, bottom: 10, left: 10 }}
        >
          <XAxis
            dataKey="month_label"
            tick={{ fontSize: 11 }}
            tickLine={false}
          />
          <YAxis
            tick={{ fontSize: 11 }}
            tickFormatter={(v: number) => v.toFixed(0)}
            domain={["auto", "auto"]}
            label={{
              value: "Seasonality Index",
              angle: -90,
              position: "insideLeft",
              style: { fontSize: 12, fill: "#6b7280" },
              offset: 0,
            }}
          />
          <Tooltip
            formatter={(value, name) => [
              typeof value === "number" ? value.toFixed(1) : String(value),
              stateNames[String(name)] || String(name),
            ]}
            labelFormatter={(label) => `Month: ${label}`}
            contentStyle={{ fontSize: 12 }}
          />
          <Legend
            formatter={(value: string) => stateNames[value] || value}
            wrapperStyle={{ fontSize: 11 }}
          />
          <ReferenceLine
            y={100}
            stroke="#d1d5db"
            strokeDasharray="4 4"
            strokeWidth={1}
            label={{
              value: "FY Avg = 100",
              position: "right",
              fill: "#9ca3af",
              fontSize: 10,
            }}
          />
          {activeSlugs.map((slug, i) => (
            <Line
              key={slug}
              type="monotone"
              dataKey={slug}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={2}
              dot={{ r: 3, fill: COLORS[i % COLORS.length] }}
              connectNulls={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <div className="mt-2 text-xs text-gray-500 text-center">
        Seasonality index: 100 = fiscal year average. Values above 100 indicate
        above-average demand in that month. Source: Robbie Andrew / POSOCO
        daily demand data.
      </div>
    </div>
  );
}
