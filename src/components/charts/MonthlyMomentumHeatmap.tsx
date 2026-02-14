"use client";

import { useState, useMemo } from "react";

interface MonthlyGrowth {
  month: string;
  yoy_pct: number;
}

interface StateProfile {
  monthly_growth: MonthlyGrowth[];
}

interface Props {
  profiles: Record<string, StateProfile>;
  stateNames: Record<string, string>;
}

/**
 * Convert "YYYY-MM" to abbreviated "MMM-YY" display format.
 */
function formatMonth(ym: string): string {
  const parts = ym.split("-");
  if (parts.length < 2) return ym;
  const year = parts[0].slice(2);
  const monthNum = parseInt(parts[1], 10);
  const names = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];
  return `${names[monthNum - 1] || parts[1]}-${year}`;
}

/**
 * Get background color for a YoY growth percentage.
 * Green for positive, red for negative, with opacity proportional to magnitude.
 */
function getCellStyle(value: number): React.CSSProperties {
  const maxMagnitude = 30; // cap at 30% for color scaling
  const magnitude = Math.min(Math.abs(value), maxMagnitude);
  const opacity = 0.15 + (magnitude / maxMagnitude) * 0.65;

  if (value >= 0) {
    return {
      backgroundColor: `rgba(34, 197, 94, ${opacity})`,
      color: opacity > 0.5 ? "#14532d" : "#166534",
    };
  }
  return {
    backgroundColor: `rgba(239, 68, 68, ${opacity})`,
    color: opacity > 0.5 ? "#450a0a" : "#991b1b",
  };
}

export default function MonthlyMomentumHeatmap({
  profiles,
  stateNames,
}: Props) {
  const [hoveredCell, setHoveredCell] = useState<{
    slug: string;
    month: string;
    value: number;
  } | null>(null);

  // Compute ranked states and month columns
  const { rankedSlugs, months } = useMemo(() => {
    const slugs = Object.keys(profiles);

    // Collect all unique months across all states, then take the last 24
    const allMonths = new Set<string>();
    for (const slug of slugs) {
      for (const mg of profiles[slug].monthly_growth) {
        allMonths.add(mg.month);
      }
    }
    const sortedMonths = Array.from(allMonths).sort();
    const last24 = sortedMonths.slice(-24);

    // Compute average growth for each state (across last 24 months)
    const avgGrowth: Record<string, number> = {};
    for (const slug of slugs) {
      const growthMap = new Map(
        profiles[slug].monthly_growth.map((mg) => [mg.month, mg.yoy_pct])
      );
      const values = last24
        .map((m) => growthMap.get(m))
        .filter((v): v is number => v !== undefined);
      avgGrowth[slug] =
        values.length > 0
          ? values.reduce((s, v) => s + v, 0) / values.length
          : -Infinity;
    }

    // Sort by average growth descending, limit to 15
    const ranked = slugs
      .sort((a, b) => (avgGrowth[b] ?? 0) - (avgGrowth[a] ?? 0))
      .slice(0, 15);

    return { rankedSlugs: ranked, months: last24 };
  }, [profiles]);

  // Build a lookup map for quick cell value access
  const valueMap = useMemo(() => {
    const map = new Map<string, number>();
    for (const slug of rankedSlugs) {
      for (const mg of profiles[slug].monthly_growth) {
        map.set(`${slug}|${mg.month}`, mg.yoy_pct);
      }
    }
    return map;
  }, [rankedSlugs, profiles]);

  if (rankedSlugs.length === 0 || months.length === 0) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-8 text-center text-gray-500">
        No monthly growth data available.
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
      <div className="overflow-x-auto">
        <table className="text-xs border-collapse">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 bg-gray-50 px-3 py-2 text-left font-medium text-gray-600 min-w-[140px]">
                State
              </th>
              {months.map((m) => (
                <th
                  key={m}
                  className="px-1 py-2 text-center font-medium text-gray-500 whitespace-nowrap min-w-[48px]"
                >
                  {formatMonth(m)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rankedSlugs.map((slug) => (
              <tr key={slug}>
                <td className="sticky left-0 z-10 bg-white px-3 py-1.5 font-medium text-gray-800 border-r border-gray-100 whitespace-nowrap">
                  {stateNames[slug] || slug}
                </td>
                {months.map((m) => {
                  const value = valueMap.get(`${slug}|${m}`);
                  const isHovered =
                    hoveredCell?.slug === slug && hoveredCell?.month === m;
                  return (
                    <td
                      key={m}
                      className="px-1 py-1.5 text-center tabular-nums relative cursor-default"
                      style={
                        value !== undefined
                          ? getCellStyle(value)
                          : { backgroundColor: "#f9fafb", color: "#d1d5db" }
                      }
                      onMouseEnter={() =>
                        value !== undefined &&
                        setHoveredCell({ slug, month: m, value })
                      }
                      onMouseLeave={() => setHoveredCell(null)}
                    >
                      {value !== undefined ? (
                        <>
                          {isHovered && (
                            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 z-20 bg-gray-900 text-white text-xs rounded px-2 py-1 whitespace-nowrap shadow-lg pointer-events-none">
                              <div className="font-medium">
                                {stateNames[slug] || slug}
                              </div>
                              <div>
                                {formatMonth(m)}: {value > 0 ? "+" : ""}
                                {value.toFixed(1)}%
                              </div>
                            </div>
                          )}
                          <span className="text-[10px]">
                            {value > 0 ? "+" : ""}
                            {Math.abs(value) >= 10
                              ? value.toFixed(0)
                              : value.toFixed(1)}
                          </span>
                        </>
                      ) : (
                        <span className="text-[10px]">{"\u00B7"}</span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="px-4 py-2 bg-gray-50 border-t border-gray-200 text-xs text-gray-500 flex items-center gap-4">
        <span>Year-over-year electricity demand growth (%)</span>
        <div className="flex items-center gap-1.5">
          <span
            className="inline-block w-3 h-3 rounded"
            style={{ backgroundColor: "rgba(239, 68, 68, 0.6)" }}
          />
          <span>Negative</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span
            className="inline-block w-3 h-3 rounded"
            style={{ backgroundColor: "rgba(34, 197, 94, 0.6)" }}
          />
          <span>Positive</span>
        </div>
      </div>
    </div>
  );
}
