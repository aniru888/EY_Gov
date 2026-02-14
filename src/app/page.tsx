import Link from "next/link";
import { getRankings, getMetadata, getTrends, getTrendSubset } from "@/lib/data";
import DashboardClient from "@/components/DashboardClient";
import DataFreshness from "@/components/common/DataFreshness";

export default function HomePage() {
  const rankings = getRankings();
  const metadata = getMetadata();
  const trends = getTrends();

  // Top 5 states for trend chart
  const topSlugs = rankings.rankings.slice(0, 5).map((r) => r.slug);
  const trendData = getTrendSubset(trends, topSlugs);

  // Slug -> state name map for chart labels
  const stateNameMap: Record<string, string> = {};
  for (const r of rankings.rankings) {
    stateNameMap[r.slug] = r.state;
  }

  // Group states by region for directory
  const byRegion: Record<string, typeof rankings.rankings> = {};
  for (const r of rankings.rankings) {
    if (!byRegion[r.region]) byRegion[r.region] = [];
    byRegion[r.region].push(r);
  }
  const regionOrder = [
    "Northern",
    "Southern",
    "Western",
    "Eastern",
    "Central",
    "North-Eastern",
    "Island",
  ];

  const fyCount = metadata.data_coverage.fiscal_years.length;

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      {/* Hero */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">
          State Economic Activity Index
        </h1>
        <p className="mt-2 text-gray-600 max-w-3xl">
          A composite index tracking economic activity across Indian states
          using four hard-to-fake indicators: GST collections, electricity
          demand, bank credit growth, and formal employment (EPFO payroll).
          Inspired by the Li Keqiang Index approach.
        </p>

        {/* Key stats row */}
        <div className="mt-4 flex flex-wrap gap-6 text-sm">
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-blue-100 text-blue-700 text-xs font-bold">
              {rankings.count}
            </span>
            <span className="text-gray-600">States Ranked</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-emerald-100 text-emerald-700 text-xs font-bold">
              {fyCount}
            </span>
            <span className="text-gray-600">Fiscal Years</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center justify-center w-8 h-8 rounded-full bg-amber-100 text-amber-700 text-xs font-bold">
              4
            </span>
            <span className="text-gray-600">Components</span>
          </div>
          <Link
            href="/methodology"
            className="text-blue-600 hover:underline text-sm font-medium self-center"
          >
            How is this calculated?
          </Link>
        </div>

        <DataFreshness metadata={metadata} />
      </div>

      {/* Dashboard */}
      <DashboardClient
        rankings={rankings.rankings}
        trendData={trendData}
        trendSlugs={topSlugs}
        stateNameMap={stateNameMap}
      />

      {/* State Directory */}
      <section className="mt-12">
        <h2 className="text-xl font-bold text-gray-900 mb-4">
          Explore All States
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {regionOrder
            .filter((region) => byRegion[region])
            .map((region) => (
              <div
                key={region}
                className="bg-white border border-gray-200 rounded-lg p-4"
              >
                <h3 className="text-sm font-semibold text-gray-900 mb-2">
                  {region}
                </h3>
                <ul className="space-y-1">
                  {byRegion[region]
                    .sort((a, b) => a.rank - b.rank)
                    .map((r) => (
                      <li key={r.slug}>
                        <Link
                          href={`/states/${r.slug}`}
                          className="flex items-center justify-between text-sm hover:bg-gray-50 rounded px-1 py-0.5 -mx-1 transition-colors"
                        >
                          <span className="text-gray-700">{r.state}</span>
                          <span className="text-xs text-gray-400 tabular-nums">
                            #{r.rank}
                          </span>
                        </Link>
                      </li>
                    ))}
                </ul>
              </div>
            ))}
        </div>
      </section>
    </div>
  );
}
