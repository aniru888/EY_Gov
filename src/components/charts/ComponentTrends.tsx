"use client";

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

interface Props {
  fiscalYears: string[];
  gstZscore: (number | null)[];
  electricityZscore: (number | null)[];
  creditZscore: (number | null)[];
  epfoZscore: (number | null)[];
}

const COMPONENTS = [
  { key: "gst", label: "GST", color: "#2563eb" },
  { key: "electricity", label: "Electricity", color: "#d97706" },
  { key: "credit", label: "Credit", color: "#16a34a" },
  { key: "epfo", label: "EPFO", color: "#7c3aed" },
] as const;

export default function ComponentTrends({
  fiscalYears,
  gstZscore,
  electricityZscore,
  creditZscore,
  epfoZscore,
}: Props) {
  const data = fiscalYears.map((fy, i) => ({
    fiscal_year: fy,
    gst: gstZscore[i],
    electricity: electricityZscore[i],
    credit: creditZscore[i],
    epfo: epfoZscore[i],
  }));

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-900 mb-3">
        Component Z-Scores Over Time
      </h3>
      <ResponsiveContainer width="100%" height={300}>
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
            label={{
              value: "Z-Score",
              angle: -90,
              position: "insideLeft",
              style: { fontSize: 11, fill: "#6b7280" },
            }}
          />
          <Tooltip
            formatter={(value, name) => [
              typeof value === "number" ? value.toFixed(3) : "N/A",
              COMPONENTS.find((c) => c.key === name)?.label || String(name),
            ]}
            labelFormatter={(label) => `FY ${label}`}
            contentStyle={{ fontSize: 13 }}
          />
          <Legend
            formatter={(value: string) =>
              COMPONENTS.find((c) => c.key === value)?.label || value
            }
            wrapperStyle={{ fontSize: 12 }}
          />
          <ReferenceLine y={0} stroke="#d1d5db" strokeDasharray="3 3" />
          {COMPONENTS.map((c) => (
            <Line
              key={c.key}
              type="monotone"
              dataKey={c.key}
              stroke={c.color}
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
