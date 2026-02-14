"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface Props {
  fiscalYears: string[];
  compositeScore: (number | null)[];
}

export default function StateScoreTrend({
  fiscalYears,
  compositeScore,
}: Props) {
  const data = fiscalYears.map((fy, i) => ({
    fiscal_year: fy,
    score: compositeScore[i],
  }));

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-900 mb-3">
        Composite Score Over Time
      </h3>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data}>
          <XAxis
            dataKey="fiscal_year"
            tick={{ fontSize: 11 }}
            tickFormatter={(v: string) =>
              v.replace("20", "'").replace("-", "-'")
            }
          />
          <YAxis
            tick={{ fontSize: 11 }}
            tickFormatter={(v: number) => v.toFixed(1)}
          />
          <Tooltip
            formatter={(value) => [
              typeof value === "number" ? value.toFixed(3) : "N/A",
              "Composite Score",
            ]}
            labelFormatter={(label) => `FY ${label}`}
            contentStyle={{ fontSize: 13 }}
          />
          <ReferenceLine y={0} stroke="#d1d5db" strokeDasharray="3 3" />
          <Line
            type="monotone"
            dataKey="score"
            stroke="#2563eb"
            strokeWidth={2.5}
            dot={{ r: 4, fill: "#2563eb" }}
            connectNulls={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
