"use client";

import { useState, useMemo } from "react";
import Link from "next/link";

interface IntensityRow {
  state: string;
  slug: string;
  intensity: number;
  gsdp_crore: number;
  electricity_mu: number;
}

interface ElasticityInfo {
  elasticity: number;
  label: string;
}

interface Props {
  data: IntensityRow[];
  elasticities: Record<string, { elasticity: number; label: string }>;
}

type SortKey =
  | "state"
  | "electricity_mu"
  | "gsdp_crore"
  | "intensity"
  | "elasticity"
  | "label";

const LABEL_BADGE_CLASSES: Record<string, string> = {
  industrial: "bg-red-100 text-red-700",
  "services-transitioning": "bg-blue-100 text-blue-700",
  "low-intensity": "bg-amber-100 text-amber-700",
};

function ClassificationBadge({ label }: { label: string | undefined }) {
  if (!label) {
    return <span className="text-gray-300 text-xs">--</span>;
  }
  const classes = LABEL_BADGE_CLASSES[label] || "bg-gray-100 text-gray-600";
  return (
    <span
      className={`inline-block text-xs font-medium px-2 py-0.5 rounded-full whitespace-nowrap ${classes}`}
    >
      {label}
    </span>
  );
}

/**
 * Format a number in Indian notation:
 * >= 1 lakh crore (1e7): show as "X.XX L Cr"
 * >= 1 lakh (1e5): show as "X.XX L"
 * >= 1 thousand (1e3): show as "X.XK"
 * else: raw number
 *
 * For electricity MU and GSDP Cr specifically:
 *   electricity_mu: values like 78449.88 -> "78.4K MU"
 *   gsdp_crore: values like 838636.61 -> "8.39L Cr"
 */
function formatIndianLarge(n: number, unit: string): string {
  const abs = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  if (unit === "Cr") {
    // GSDP in crore
    if (abs >= 1e5) return `${sign}${(abs / 1e5).toFixed(2)}L Cr`;
    if (abs >= 1e3) return `${sign}${(abs / 1e3).toFixed(1)}K Cr`;
    return `${sign}${abs.toFixed(0)} Cr`;
  }
  if (unit === "MU") {
    if (abs >= 1e5) return `${sign}${(abs / 1e5).toFixed(2)}L MU`;
    if (abs >= 1e3) return `${sign}${(abs / 1e3).toFixed(1)}K MU`;
    return `${sign}${abs.toFixed(0)} MU`;
  }
  return n.toFixed(4);
}

export default function IntensityTable({ data, elasticities }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("intensity");
  const [sortAsc, setSortAsc] = useState(false);

  const sorted = useMemo(() => {
    return [...data].sort((a, b) => {
      let av: number | string;
      let bv: number | string;

      switch (sortKey) {
        case "state":
          return sortAsc
            ? a.state.localeCompare(b.state)
            : b.state.localeCompare(a.state);
        case "electricity_mu":
          av = a.electricity_mu;
          bv = b.electricity_mu;
          break;
        case "gsdp_crore":
          av = a.gsdp_crore;
          bv = b.gsdp_crore;
          break;
        case "intensity":
          av = a.intensity;
          bv = b.intensity;
          break;
        case "elasticity":
          av = elasticities[a.slug]?.elasticity ?? -Infinity;
          bv = elasticities[b.slug]?.elasticity ?? -Infinity;
          break;
        case "label":
          av = elasticities[a.slug]?.label ?? "";
          bv = elasticities[b.slug]?.label ?? "";
          return sortAsc
            ? String(av).localeCompare(String(bv))
            : String(bv).localeCompare(String(av));
        default:
          return 0;
      }
      return sortAsc
        ? (av as number) - (bv as number)
        : (bv as number) - (av as number);
    });
  }, [data, sortKey, sortAsc, elasticities]);

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(key === "state");
    }
  }

  function SortHeader({
    label,
    sortKeyVal,
    align,
  }: {
    label: string;
    sortKeyVal: SortKey;
    align?: "left" | "right" | "center";
  }) {
    const active = sortKey === sortKeyVal;
    const textAlign =
      align === "right"
        ? "text-right"
        : align === "center"
          ? "text-center"
          : "text-left";
    return (
      <th
        className={`px-3 py-2 ${textAlign} text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:text-gray-900 select-none whitespace-nowrap`}
        onClick={() => handleSort(sortKeyVal)}
      >
        {label}
        {active && (
          <span className="ml-1">{sortAsc ? "\u2191" : "\u2193"}</span>
        )}
      </th>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-10">
                #
              </th>
              <SortHeader label="State" sortKeyVal="state" align="left" />
              <SortHeader
                label="Electricity (MU)"
                sortKeyVal="electricity_mu"
                align="right"
              />
              <SortHeader
                label="GSDP (Cr)"
                sortKeyVal="gsdp_crore"
                align="right"
              />
              <SortHeader
                label="Intensity (MU/Cr)"
                sortKeyVal="intensity"
                align="right"
              />
              <SortHeader
                label="Elasticity"
                sortKeyVal="elasticity"
                align="right"
              />
              <SortHeader
                label="Classification"
                sortKeyVal="label"
                align="center"
              />
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {sorted.map((row, idx) => {
              const elast: ElasticityInfo | undefined = elasticities[row.slug];
              return (
                <tr key={row.slug} className="hover:bg-gray-50">
                  <td className="px-3 py-2 text-sm text-gray-400 tabular-nums">
                    {idx + 1}
                  </td>
                  <td className="px-3 py-2 text-sm font-medium text-gray-900">
                    <Link
                      href={`/states/${row.slug}`}
                      className="hover:text-blue-600 transition-colors"
                    >
                      {row.state}
                    </Link>
                  </td>
                  <td className="px-3 py-2 text-sm text-right tabular-nums text-gray-700">
                    {formatIndianLarge(row.electricity_mu, "MU")}
                  </td>
                  <td className="px-3 py-2 text-sm text-right tabular-nums text-gray-700">
                    {formatIndianLarge(row.gsdp_crore, "Cr")}
                  </td>
                  <td className="px-3 py-2 text-sm text-right tabular-nums font-medium text-gray-800">
                    {row.intensity.toFixed(4)}
                  </td>
                  <td className="px-3 py-2 text-sm text-right tabular-nums text-gray-600">
                    {elast ? elast.elasticity.toFixed(2) : "--"}
                  </td>
                  <td className="px-3 py-2 text-center">
                    <ClassificationBadge label={elast?.label} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="px-4 py-2 bg-gray-50 border-t border-gray-200 text-xs text-gray-500">
        Intensity = Electricity (MU) / GSDP (Cr). Higher intensity suggests
        more electricity-heavy economic structure. Elasticity measures
        responsiveness of electricity demand to GDP growth.
      </div>
    </div>
  );
}
