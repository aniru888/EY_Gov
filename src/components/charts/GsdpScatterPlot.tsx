"use client";

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
  Label,
} from "recharts";
import type { GsdpComparisonEntry } from "@/lib/types";

interface Props {
  data: GsdpComparisonEntry[];
}

const GAP_COLORS: Record<string, string> = {
  outperformer: "#16a34a",
  underperformer: "#dc2626",
  aligned: "#9ca3af",
};

function shouldLabel(entry: GsdpComparisonEntry, idx: number): boolean {
  // Label top 10 by index rank + any with large gap
  if (entry.index_rank !== null && entry.index_rank <= 10) return true;
  if (entry.rank_gap !== null && Math.abs(entry.rank_gap) >= 5) return true;
  return false;
}

interface CustomLabelProps {
  x?: number;
  y?: number;
  index?: number;
  data: GsdpComparisonEntry[];
}

function CustomDotLabel({ x, y, index, data }: CustomLabelProps) {
  if (x === undefined || y === undefined || index === undefined) return null;
  const entry = data[index];
  if (!entry || !shouldLabel(entry, index)) return null;
  return (
    <text
      x={x}
      y={y - 8}
      textAnchor="middle"
      fill="#374151"
      fontSize={10}
      fontWeight={500}
    >
      {entry.state.length > 12
        ? entry.state.slice(0, 11) + "..."
        : entry.state}
    </text>
  );
}

interface TooltipProps {
  active?: boolean;
  payload?: Array<{
    payload: GsdpComparisonEntry;
  }>;
}

function ScatterTooltip({ active, payload }: TooltipProps) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-md p-3 text-sm">
      <div className="font-semibold text-gray-900">{d.state}</div>
      <div className="text-gray-600 mt-1">
        Index Rank: #{d.index_rank ?? "--"}
      </div>
      <div className="text-gray-600">GSDP Rank: #{d.gsdp_rank}</div>
      {d.rank_gap !== null && (
        <div
          className={`font-medium mt-1 ${
            d.rank_gap > 0
              ? "text-emerald-600"
              : d.rank_gap < 0
                ? "text-red-600"
                : "text-gray-500"
          }`}
        >
          Gap: {d.rank_gap > 0 ? "+" : ""}
          {d.rank_gap} ({d.gap_label})
        </div>
      )}
    </div>
  );
}

export default function GsdpScatterPlot({ data }: Props) {
  const valid = data.filter((d) => d.index_rank !== null);
  if (valid.length === 0) return null;

  const maxRank = Math.max(
    ...valid.map((d) => Math.max(d.gsdp_rank, d.index_rank!))
  );
  const axisMax = maxRank + 2;

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart margin={{ top: 20, right: 20, bottom: 30, left: 30 }}>
          <XAxis
            type="number"
            dataKey="gsdp_rank"
            domain={[1, axisMax]}
            tick={{ fontSize: 11 }}
            reversed={false}
          >
            <Label
              value="GSDP Rank (official)"
              offset={-15}
              position="insideBottom"
              style={{ fontSize: 12, fill: "#6b7280" }}
            />
          </XAxis>
          <YAxis
            type="number"
            dataKey="index_rank"
            domain={[1, axisMax]}
            tick={{ fontSize: 11 }}
            reversed
          >
            <Label
              value="Index Rank (ours)"
              angle={-90}
              position="insideLeft"
              style={{ fontSize: 12, fill: "#6b7280" }}
              offset={-10}
            />
          </YAxis>
          <Tooltip content={<ScatterTooltip />} />
          <ReferenceLine
            segment={[
              { x: 1, y: 1 },
              { x: axisMax, y: axisMax },
            ]}
            stroke="#d1d5db"
            strokeDasharray="5 5"
            strokeWidth={1.5}
          />
          <Scatter
            data={valid}
            label={<CustomDotLabel data={valid} />}
          >
            {valid.map((entry, i) => (
              <Cell
                key={entry.slug}
                fill={GAP_COLORS[entry.gap_label || "aligned"] || "#9ca3af"}
                r={5}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
      <div className="flex justify-center gap-6 mt-2 text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-600" />
          Index outperforms GDP
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-gray-400" />
          Aligned
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-red-600" />
          Index underperforms GDP
        </span>
      </div>
    </div>
  );
}
