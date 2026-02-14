"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { TrendChartPoint } from "@/lib/types";

const COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed"];

interface Props {
  data: TrendChartPoint[];
  slugs: string[];
  stateNameMap: Record<string, string>;
}

export default function TrendChart({ data, slugs, stateNameMap }: Props) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data}>
          <XAxis
            dataKey="fiscal_year"
            tick={{ fontSize: 12 }}
            tickFormatter={(v: string) => v.replace("20", "'").replace("-", "-'")}
          />
          <YAxis
            tick={{ fontSize: 12 }}
            tickFormatter={(v: number) => v.toFixed(1)}
            label={{
              value: "Composite Score",
              angle: -90,
              position: "insideLeft",
              style: { fontSize: 12, fill: "#6b7280" },
            }}
          />
          <Tooltip
            formatter={(value, name) => [
              typeof value === "number" ? value.toFixed(3) : "N/A",
              stateNameMap[String(name)] || String(name),
            ]}
            labelFormatter={(label) => `FY ${label}`}
            contentStyle={{ fontSize: 13 }}
          />
          <Legend
            formatter={(value: string) => stateNameMap[value] || value}
            wrapperStyle={{ fontSize: 12 }}
          />
          {slugs.map((slug, i) => (
            <Line
              key={slug}
              type="monotone"
              dataKey={slug}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={2}
              dot={{ r: 3 }}
              connectNulls={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
