"use client";

import { useMemo } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Line,
  ComposedChart,
} from "recharts";

interface ScatterPoint {
  state: string;
  slug: string;
  electricity_mu: number;
  gsdp_crore: number;
  elasticity_label: string | null;
  residual_crore: number | null;
}

interface RegressionInfo {
  r_squared: number;
  coef: number;
  intercept: number;
  n: number;
  latest_fy: string;
}

interface Props {
  data: ScatterPoint[];
  regression: RegressionInfo;
}

const LABEL_COLORS: Record<string, string> = {
  industrial: "#ef4444",
  "services-transitioning": "#3b82f6",
  "low-intensity": "#f59e0b",
};
const DEFAULT_COLOR = "#9ca3af";

function getColor(label: string | null): string {
  if (!label) return DEFAULT_COLOR;
  return LABEL_COLORS[label] || DEFAULT_COLOR;
}

interface ScatterTooltipProps {
  active?: boolean;
  payload?: Array<{
    payload: ScatterPoint;
  }>;
}

function ScatterTooltip({ active, payload }: ScatterTooltipProps) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-md p-3 text-sm">
      <div className="font-semibold text-gray-900">{d.state}</div>
      <div className="text-gray-600 mt-1">
        Electricity: {d.electricity_mu.toLocaleString("en-IN")} MU
      </div>
      <div className="text-gray-600">
        GSDP: {(d.gsdp_crore / 1e5).toFixed(2)} L Cr
      </div>
      {d.elasticity_label && (
        <div className="mt-1">
          <span
            className="inline-block text-xs font-medium px-2 py-0.5 rounded-full"
            style={{
              backgroundColor: getColor(d.elasticity_label) + "20",
              color: getColor(d.elasticity_label),
            }}
          >
            {d.elasticity_label}
          </span>
        </div>
      )}
      {d.residual_crore !== null && (
        <div className="text-gray-500 text-xs mt-1">
          Residual: {d.residual_crore > 0 ? "+" : ""}
          {(d.residual_crore / 1e5).toFixed(2)} L Cr
        </div>
      )}
    </div>
  );
}

interface CustomLabelProps {
  x?: number;
  y?: number;
  index?: number;
}

export default function ElectricityGsdpScatter({ data, regression }: Props) {
  // Identify outlier states: top 5 by absolute residual
  const outlierSlugs = useMemo(() => {
    const withResidual = data
      .filter((d) => d.residual_crore !== null)
      .sort(
        (a, b) =>
          Math.abs(b.residual_crore as number) -
          Math.abs(a.residual_crore as number)
      );
    return new Set(withResidual.slice(0, 5).map((d) => d.slug));
  }, [data]);

  // Build regression line points spanning the data range
  const regressionLine = useMemo(() => {
    const elecValues = data.map((d) => d.electricity_mu).filter((v) => v > 0);
    if (elecValues.length === 0) return [];
    const minElec = Math.min(...elecValues);
    const maxElec = Math.max(...elecValues);
    const steps = 50;
    const points: Array<{ electricity_mu: number; regression_gsdp: number }> =
      [];
    for (let i = 0; i <= steps; i++) {
      const elec = minElec * Math.pow(maxElec / minElec, i / steps);
      const gsdp = regression.intercept + regression.coef * elec;
      if (gsdp > 0) {
        points.push({ electricity_mu: elec, regression_gsdp: gsdp });
      }
    }
    return points;
  }, [data, regression]);

  // Custom label renderer for outlier states
  function OutlierLabel({ x, y, index }: CustomLabelProps) {
    if (x === undefined || y === undefined || index === undefined) return null;
    const entry = data[index];
    if (!entry || !outlierSlugs.has(entry.slug)) return null;
    return (
      <text
        x={x}
        y={y - 10}
        textAnchor="middle"
        fill="#374151"
        fontSize={10}
        fontWeight={500}
      >
        {entry.state.length > 14
          ? entry.state.slice(0, 13) + "..."
          : entry.state}
      </text>
    );
  }

  const validData = data.filter(
    (d) => d.electricity_mu > 0 && d.gsdp_crore > 0
  );

  if (validData.length === 0) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-8 text-center text-gray-500">
        No scatter data available.
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart
          margin={{ top: 20, right: 30, bottom: 40, left: 40 }}
        >
          <XAxis
            type="number"
            dataKey="electricity_mu"
            scale="log"
            domain={["auto", "auto"]}
            tick={{ fontSize: 11 }}
            tickFormatter={(v: number) =>
              v >= 1000
                ? `${(v / 1000).toFixed(0)}K`
                : v.toFixed(0)
            }
            label={{
              value: "Electricity Demand (MU, log scale)",
              position: "insideBottom",
              offset: -20,
              style: { fontSize: 12, fill: "#6b7280" },
            }}
            allowDuplicatedCategory={false}
          />
          <YAxis
            type="number"
            dataKey="gsdp_crore"
            scale="log"
            domain={["auto", "auto"]}
            tick={{ fontSize: 11 }}
            tickFormatter={(v: number) => {
              if (v >= 1e7) return `${(v / 1e7).toFixed(0)} Cr`;
              if (v >= 1e5) return `${(v / 1e5).toFixed(0)} L`;
              if (v >= 1e3) return `${(v / 1e3).toFixed(0)}K`;
              return v.toFixed(0);
            }}
            label={{
              value: "GSDP (Cr, log scale)",
              angle: -90,
              position: "insideLeft",
              offset: -20,
              style: { fontSize: 12, fill: "#6b7280" },
            }}
          />
          <Tooltip content={<ScatterTooltip />} />

          {/* Regression line overlay */}
          <Line
            data={regressionLine}
            type="monotone"
            dataKey="regression_gsdp"
            stroke="#9ca3af"
            strokeWidth={1.5}
            strokeDasharray="6 3"
            dot={false}
            legendType="none"
            isAnimationActive={false}
          />

          {/* Scatter plot points */}
          <Scatter
            data={validData}
            label={<OutlierLabel />}
          >
            {validData.map((entry) => (
              <Cell
                key={entry.slug}
                fill={getColor(entry.elasticity_label)}
                r={5}
              />
            ))}
          </Scatter>
        </ComposedChart>
      </ResponsiveContainer>

      {/* R-squared annotation */}
      <div className="flex items-center justify-between mt-2 px-2">
        <div className="text-xs text-gray-500">
          R{"\u00B2"} = {regression.r_squared.toFixed(3)} | n ={" "}
          {regression.n} states | FY {regression.latest_fy}
        </div>
        <div className="text-xs text-gray-400">
          GSDP = {regression.intercept.toLocaleString("en-IN", { maximumFractionDigits: 0 })} +{" "}
          {regression.coef.toFixed(1)} x Electricity
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap justify-center gap-4 mt-3 text-xs text-gray-500">
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: LABEL_COLORS.industrial }}
          />
          Industrial
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: LABEL_COLORS["services-transitioning"] }}
          />
          Services-transitioning
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: LABEL_COLORS["low-intensity"] }}
          />
          Low-intensity
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: DEFAULT_COLOR }}
          />
          Unclassified
        </span>
      </div>
    </div>
  );
}
