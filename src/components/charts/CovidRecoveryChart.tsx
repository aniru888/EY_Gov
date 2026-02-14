"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { CovidRecoveryEntry } from "@/lib/types";

interface Props {
  data: CovidRecoveryEntry[];
}

function getRecoveryColor(entry: CovidRecoveryEntry): string {
  if (entry.never_recovered) return "#dc2626"; // red
  if (entry.recovery_speed === null) return "#9ca3af"; // gray
  if (entry.recovery_speed <= 1) return "#16a34a"; // green - fast
  if (entry.recovery_speed <= 2) return "#65a30d"; // lime
  if (entry.recovery_speed <= 3) return "#d97706"; // amber
  return "#dc2626"; // red - slow
}

function getRecoveryLabel(entry: CovidRecoveryEntry): string {
  if (entry.never_recovered) return "Not recovered";
  if (entry.recovery_speed === null) return "No data";
  if (entry.recovery_speed <= 1) return `${entry.recovery_speed} FY (fast)`;
  return `${entry.recovery_speed} FYs`;
}

interface TooltipProps {
  active?: boolean;
  payload?: Array<{
    payload: CovidRecoveryEntry;
  }>;
}

function RecoveryTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-md p-3 text-sm">
      <div className="font-semibold text-gray-900">{d.state}</div>
      <div className="text-gray-600 mt-1">
        COVID Dip: {d.covid_dip.toFixed(2)}
      </div>
      {d.recovery_fy && (
        <div className="text-gray-600">Recovered by: FY {d.recovery_fy}</div>
      )}
      <div
        className={`font-medium mt-1 ${
          d.never_recovered ? "text-red-600" : "text-emerald-600"
        }`}
      >
        {getRecoveryLabel(d)}
      </div>
      {d.pre_covid_declining && (
        <div className="text-amber-600 text-xs mt-1">
          Already declining before COVID
        </div>
      )}
    </div>
  );
}

export default function CovidRecoveryChart({ data }: Props) {
  if (data.length === 0) return null;

  // Sort: recovered states first (by speed), then not-recovered
  const sorted = [...data].sort((a, b) => {
    if (a.never_recovered && !b.never_recovered) return 1;
    if (!a.never_recovered && b.never_recovered) return -1;
    if (a.recovery_speed === null && b.recovery_speed === null) return 0;
    if (a.recovery_speed === null) return 1;
    if (b.recovery_speed === null) return -1;
    return a.recovery_speed - b.recovery_speed;
  });

  // For the bar chart, use covid_dip magnitude (negative = bigger dip)
  // Show recovery_speed as the bar value, with never_recovered as max
  const maxSpeed = Math.max(
    ...sorted
      .filter((d) => d.recovery_speed !== null)
      .map((d) => d.recovery_speed!)
  );
  const chartMax = maxSpeed + 1;

  const chartData = sorted.map((d) => ({
    ...d,
    bar_value: d.never_recovered
      ? chartMax
      : d.recovery_speed ?? chartMax,
    label: getRecoveryLabel(d),
  }));

  const barHeight = 28;
  const chartHeight = Math.max(300, chartData.length * barHeight + 60);

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <ResponsiveContainer width="100%" height={chartHeight}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
        >
          <XAxis
            type="number"
            domain={[0, chartMax]}
            tick={{ fontSize: 11 }}
            tickFormatter={(v: number) =>
              v === chartMax ? "N/R" : `${v} FY`
            }
          />
          <YAxis
            type="category"
            dataKey="state"
            tick={{ fontSize: 11 }}
            width={95}
          />
          <Tooltip content={<RecoveryTooltip />} />
          <Bar dataKey="bar_value" radius={[0, 4, 4, 0]}>
            {chartData.map((entry, i) => (
              <Cell key={entry.slug} fill={getRecoveryColor(entry)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex justify-center gap-6 mt-2 text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-600" />
          Fast (1 FY)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-amber-600" />
          Slow (3+ FYs)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-red-600" />
          Not recovered
        </span>
      </div>
    </div>
  );
}
