"use client";

import { useState } from "react";
import Link from "next/link";
import type { RankingEntry } from "@/lib/types";

type SortKey =
  | "rank"
  | "state"
  | "composite_score"
  | "gst_zscore"
  | "electricity_zscore"
  | "credit_zscore"
  | "epfo_zscore";

interface Props {
  rankings: RankingEntry[];
  selectedSlug: string;
  onSelectState: (slug: string) => void;
}

function getTierColor(rank: number, total: number): string {
  const third = Math.ceil(total / 3);
  if (rank <= third) return "bg-emerald-50";
  if (rank <= third * 2) return "bg-amber-50";
  return "bg-red-50";
}

function RankChange({ change }: { change: number | null }) {
  if (change === null || change === 0) {
    return <span className="text-gray-400 text-xs">&mdash;</span>;
  }
  if (change > 0) {
    return (
      <span className="text-red-600 text-xs font-medium">
        &#9660;{Math.abs(change)}
      </span>
    );
  }
  return (
    <span className="text-emerald-600 text-xs font-medium">
      &#9650;{Math.abs(change)}
    </span>
  );
}

function ZScoreBar({ value }: { value: number | null }) {
  if (value === null) return <span className="text-gray-300 text-xs">--</span>;
  const clamped = Math.max(-3, Math.min(3, value));
  const pct = ((clamped + 3) / 6) * 100;
  const color = value >= 0 ? "bg-emerald-400" : "bg-red-400";
  return (
    <div className="flex items-center gap-1">
      <div className="w-16 h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full ${color} rounded-full`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs text-gray-500 w-8 text-right tabular-nums">
        {value.toFixed(1)}
      </span>
    </div>
  );
}

export default function RankingsTable({
  rankings,
  selectedSlug,
  onSelectState,
}: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("rank");
  const [sortAsc, setSortAsc] = useState(true);

  const sorted = [...rankings].sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    if (av === null && bv === null) return 0;
    if (av === null) return 1;
    if (bv === null) return -1;
    if (typeof av === "string" && typeof bv === "string") {
      return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
    }
    return sortAsc
      ? (av as number) - (bv as number)
      : (bv as number) - (av as number);
  });

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(key === "rank" || key === "state");
    }
  }

  function SortHeader({
    label,
    sortKeyVal,
    className,
  }: {
    label: string;
    sortKeyVal: SortKey;
    className?: string;
  }) {
    const active = sortKey === sortKeyVal;
    return (
      <th
        className={`px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:text-gray-900 select-none ${className || ""}`}
        onClick={() => handleSort(sortKeyVal)}
      >
        {label}
        {active && (
          <span className="ml-1">{sortAsc ? "\u2191" : "\u2193"}</span>
        )}
      </th>
    );
  }

  const total = rankings.length;

  return (
    <div>
      {/* Desktop table */}
      <div className="hidden md:block overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <SortHeader label="#" sortKeyVal="rank" className="w-12" />
              <SortHeader label="State" sortKeyVal="state" />
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Region
              </th>
              <SortHeader label="Score" sortKeyVal="composite_score" />
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Change
              </th>
              <SortHeader label="GST" sortKeyVal="gst_zscore" />
              <SortHeader label="Elec" sortKeyVal="electricity_zscore" />
              <SortHeader label="Credit" sortKeyVal="credit_zscore" />
              <SortHeader label="EPFO" sortKeyVal="epfo_zscore" />
              <th className="px-3 py-2 w-10" />
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {sorted.map((r) => (
              <tr
                key={r.slug}
                className={`cursor-pointer transition-colors ${
                  r.slug === selectedSlug
                    ? "ring-2 ring-inset ring-blue-400 bg-blue-50"
                    : getTierColor(r.rank, total) + " hover:bg-gray-100"
                }`}
                onClick={() => onSelectState(r.slug)}
              >
                <td className="px-3 py-2 text-sm font-semibold text-gray-700 tabular-nums">
                  {r.rank}
                </td>
                <td className="px-3 py-2 text-sm font-medium text-gray-900">
                  {r.state}
                </td>
                <td className="px-3 py-2 text-xs text-gray-500">{r.region}</td>
                <td className="px-3 py-2 text-sm font-semibold text-gray-800 tabular-nums">
                  {r.composite_score.toFixed(2)}
                </td>
                <td className="px-3 py-2">
                  <RankChange change={r.rank_change} />
                </td>
                <td className="px-3 py-2">
                  <ZScoreBar value={r.gst_zscore} />
                </td>
                <td className="px-3 py-2">
                  <ZScoreBar value={r.electricity_zscore} />
                </td>
                <td className="px-3 py-2">
                  <ZScoreBar value={r.credit_zscore} />
                </td>
                <td className="px-3 py-2">
                  <ZScoreBar value={r.epfo_zscore} />
                </td>
                <td className="px-3 py-2">
                  <Link
                    href={`/states/${r.slug}`}
                    onClick={(e) => e.stopPropagation()}
                    className="text-gray-400 hover:text-blue-600 transition-colors"
                    title={`View ${r.state} details`}
                  >
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M9 5l7 7-7 7"
                      />
                    </svg>
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Mobile cards */}
      <div className="md:hidden space-y-2">
        {sorted.map((r) => (
          <div
            key={r.slug}
            className={`p-3 rounded-lg border cursor-pointer transition-colors ${
              r.slug === selectedSlug
                ? "border-blue-400 bg-blue-50"
                : "border-gray-200 " +
                  getTierColor(r.rank, total) +
                  " hover:border-gray-300"
            }`}
            onClick={() => onSelectState(r.slug)}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-gray-200 text-xs font-bold text-gray-700">
                  {r.rank}
                </span>
                <div>
                  <div className="text-sm font-semibold text-gray-900">
                    {r.state}
                  </div>
                  <div className="text-xs text-gray-500">{r.region}</div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="text-right">
                  <div className="text-sm font-bold text-gray-800 tabular-nums">
                    {r.composite_score.toFixed(2)}
                  </div>
                  <RankChange change={r.rank_change} />
                </div>
                <Link
                  href={`/states/${r.slug}`}
                  onClick={(e) => e.stopPropagation()}
                  className="text-blue-600 text-xs font-medium hover:underline"
                >
                  View
                </Link>
              </div>
            </div>
            <div className="mt-2 grid grid-cols-4 gap-1">
              <div className="text-center">
                <div className="text-[10px] text-gray-400">GST</div>
                <ZScoreBar value={r.gst_zscore} />
              </div>
              <div className="text-center">
                <div className="text-[10px] text-gray-400">Elec</div>
                <ZScoreBar value={r.electricity_zscore} />
              </div>
              <div className="text-center">
                <div className="text-[10px] text-gray-400">Credit</div>
                <ZScoreBar value={r.credit_zscore} />
              </div>
              <div className="text-center">
                <div className="text-[10px] text-gray-400">EPFO</div>
                <ZScoreBar value={r.epfo_zscore} />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
