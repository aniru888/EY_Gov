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

interface Props {
  months: string[];
  gstTotal: (number | null)[];
  electricityMu: (number | null)[];
}

export default function MonthlySubIndex({
  months,
  gstTotal,
  electricityMu,
}: Props) {
  const data = months.map((m, i) => ({
    month: m,
    gst: gstTotal[i],
    electricity: electricityMu[i],
  }));

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-900 mb-1">
        Monthly GST & Electricity
      </h3>
      <p className="text-xs text-gray-500 mb-3">
        Raw values (not z-scores). Only GST + electricity available at monthly
        granularity.
      </p>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data}>
          <XAxis
            dataKey="month"
            tick={{ fontSize: 10 }}
            interval="preserveStartEnd"
            tickFormatter={(v: string) => {
              const [y, m] = v.split("-");
              return `${m}/${y.slice(2)}`;
            }}
          />
          <YAxis
            yAxisId="gst"
            tick={{ fontSize: 10 }}
            tickFormatter={(v: number) =>
              v >= 1000 ? `${(v / 1000).toFixed(0)}K` : String(v)
            }
            label={{
              value: "GST (Cr)",
              angle: -90,
              position: "insideLeft",
              style: { fontSize: 10, fill: "#2563eb" },
            }}
          />
          <YAxis
            yAxisId="elec"
            orientation="right"
            tick={{ fontSize: 10 }}
            tickFormatter={(v: number) =>
              v >= 1000 ? `${(v / 1000).toFixed(0)}K` : String(v)
            }
            label={{
              value: "Elec (MU)",
              angle: 90,
              position: "insideRight",
              style: { fontSize: 10, fill: "#d97706" },
            }}
          />
          <Tooltip
            formatter={(value, name) => {
              const v =
                typeof value === "number" ? value.toLocaleString("en-IN") : "N/A";
              const label = name === "gst" ? "GST (Cr)" : "Electricity (MU)";
              return [v, label];
            }}
            labelFormatter={(label) => label}
            contentStyle={{ fontSize: 12 }}
          />
          <Legend
            formatter={(value: string) =>
              value === "gst" ? "GST (Cr)" : "Electricity (MU)"
            }
            wrapperStyle={{ fontSize: 11 }}
          />
          <Line
            yAxisId="gst"
            type="monotone"
            dataKey="gst"
            stroke="#2563eb"
            strokeWidth={1.5}
            dot={false}
            connectNulls={false}
          />
          <Line
            yAxisId="elec"
            type="monotone"
            dataKey="electricity"
            stroke="#d97706"
            strokeWidth={1.5}
            dot={false}
            connectNulls={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
