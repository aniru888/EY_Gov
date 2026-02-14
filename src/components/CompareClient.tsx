"use client";

import { useState, useMemo } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import type { RankingEntry, TrendChartPoint, TrendsData } from "@/lib/types";
import ComparisonTable from "@/components/comparison/ComparisonTable";

const TrendChart = dynamic(() => import("@/components/charts/TrendChart"), {
  ssr: false,
  loading: () => (
    <div className="h-80 bg-gray-50 rounded-lg animate-pulse" />
  ),
});

interface Props {
  rankings: RankingEntry[];
  trends: TrendsData;
}

function getTrendSubsetClient(
  trends: TrendsData,
  slugs: string[]
): TrendChartPoint[] {
  const fySet = new Set<string>();
  const stateData: Record<string, Record<string, number | null>> = {};

  for (const slug of slugs) {
    const entry = trends.annual[slug];
    if (!entry) continue;
    stateData[slug] = {};
    for (let i = 0; i < entry.fiscal_years.length; i++) {
      const fy = entry.fiscal_years[i];
      fySet.add(fy);
      stateData[slug][fy] = entry.composite_score[i];
    }
  }

  const fiscalYears = Array.from(fySet).sort();
  return fiscalYears.map((fy) => {
    const point: TrendChartPoint = { fiscal_year: fy };
    for (const slug of slugs) {
      point[slug] = stateData[slug]?.[fy] ?? null;
    }
    return point;
  });
}

export default function CompareClient({ rankings, trends }: Props) {
  const searchParams = useSearchParams();
  const router = useRouter();

  // Parse initial states from URL
  const initialSlugs = searchParams.get("states")?.split(",").filter(Boolean) || [];
  const [selectedSlugs, setSelectedSlugs] = useState<string[]>(initialSlugs);
  const [search, setSearch] = useState("");

  const stateNameMap: Record<string, string> = {};
  for (const r of rankings) {
    stateNameMap[r.slug] = r.state;
  }

  // Filter states for search
  const filtered = rankings.filter(
    (r) =>
      !selectedSlugs.includes(r.slug) &&
      r.state.toLowerCase().includes(search.toLowerCase())
  );

  function addState(slug: string) {
    if (selectedSlugs.length >= 5) return;
    const next = [...selectedSlugs, slug];
    setSelectedSlugs(next);
    setSearch("");
    router.replace(`/compare?states=${next.join(",")}`, { scroll: false });
  }

  function removeState(slug: string) {
    const next = selectedSlugs.filter((s) => s !== slug);
    setSelectedSlugs(next);
    router.replace(
      next.length ? `/compare?states=${next.join(",")}` : "/compare",
      { scroll: false }
    );
  }

  const selectedRankings = selectedSlugs
    .map((s) => rankings.find((r) => r.slug === s))
    .filter((r): r is RankingEntry => r !== undefined);

  const trendData = useMemo(
    () => getTrendSubsetClient(trends, selectedSlugs),
    [trends, selectedSlugs]
  );

  return (
    <div className="space-y-6">
      {/* State selector */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select states to compare (2-5)
        </label>

        {/* Selected chips */}
        <div className="flex flex-wrap gap-2 mb-3">
          {selectedSlugs.map((slug) => (
            <span
              key={slug}
              className="inline-flex items-center gap-1 px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
            >
              {stateNameMap[slug] || slug}
              <button
                onClick={() => removeState(slug)}
                className="ml-1 text-blue-600 hover:text-blue-900"
              >
                x
              </button>
            </span>
          ))}
          {selectedSlugs.length === 0 && (
            <span className="text-sm text-gray-400">
              No states selected. Search below to add.
            </span>
          )}
        </div>

        {/* Search input */}
        {selectedSlugs.length < 5 && (
          <div className="relative">
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search for a state..."
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            {search && filtered.length > 0 && (
              <div className="absolute z-10 mt-1 w-full bg-white border border-gray-200 rounded-lg shadow-lg max-h-48 overflow-y-auto">
                {filtered.slice(0, 10).map((r) => (
                  <button
                    key={r.slug}
                    onClick={() => addState(r.slug)}
                    className="w-full text-left px-3 py-2 text-sm hover:bg-gray-50 flex items-center justify-between"
                  >
                    <span>{r.state}</span>
                    <span className="text-xs text-gray-400">#{r.rank}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Results */}
      {selectedSlugs.length >= 2 && (
        <>
          {/* Trend chart overlay */}
          <div>
            <h2 className="text-lg font-bold text-gray-900 mb-3">
              Composite Score Trends
            </h2>
            <TrendChart
              data={trendData}
              slugs={selectedSlugs}
              stateNameMap={stateNameMap}
            />
          </div>

          {/* Side-by-side table */}
          <div>
            <h2 className="text-lg font-bold text-gray-900 mb-3">
              Component Comparison &mdash; FY {rankings[0] ? "2024-25" : ""}
            </h2>
            <ComparisonTable states={selectedRankings} />
          </div>
        </>
      )}

      {selectedSlugs.length === 1 && (
        <div className="text-center py-8 text-gray-500">
          Select at least one more state to see a comparison.
        </div>
      )}
    </div>
  );
}
