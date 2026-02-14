import type { Metadata } from "next";
import Link from "next/link";
import { getStateData, getAllStateSlugs, getRankings } from "@/lib/data";
import { formatIndianNumber } from "@/lib/data";
import Breadcrumbs from "@/components/common/Breadcrumbs";
import MetricCard from "@/components/common/MetricCard";
import StateDetailClient from "@/components/StateDetailClient";
import PeerComparison from "@/components/charts/PeerComparison";
import StateDiagnosticBanner from "@/components/StateDiagnosticBanner";

export async function generateStaticParams() {
  const slugs = getAllStateSlugs();
  return slugs.map((slug) => ({ slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const data = getStateData(slug);
  return {
    title: `${data.state} | State Economic Activity Index`,
    description: `Economic activity index for ${data.state}: composite score, GST, electricity, bank credit, and EPFO payroll trends.`,
  };
}

export default async function StatePage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const stateData = getStateData(slug);
  const rankings = getRankings();

  // Find latest FY with a non-null composite score
  const ann = stateData.annual;
  let latestIdx = ann.fiscal_years.length - 1;
  while (latestIdx >= 0 && ann.composite_score[latestIdx] === null) {
    latestIdx--;
  }

  const hasScore = latestIdx >= 0;
  const latestFy = hasScore ? ann.fiscal_years[latestIdx] : "N/A";
  const score = hasScore ? ann.composite_score[latestIdx] : null;
  const rank = hasScore ? ann.rank[latestIdx] : null;
  const rankChange = hasScore ? ann.rank_change[latestIdx] : null;
  const nComponents = hasScore ? ann.n_components[latestIdx] : 0;

  // Find this state in rankings for z-scores
  const rankingEntry = rankings.rankings.find((r) => r.slug === slug);

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <Breadcrumbs
        items={[
          { label: "Home", href: "/" },
          { label: "States", href: "/#explore" },
          { label: stateData.state },
        ]}
      />

      {/* Header */}
      <div className="mb-6">
        <div className="flex flex-wrap items-baseline gap-3">
          <h1 className="text-3xl font-bold text-gray-900">
            {stateData.state}
          </h1>
          {hasScore && rank !== null && (
            <span className="text-lg font-semibold text-blue-700">
              Rank #{Math.round(rank)} / {rankings.count}
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-3 mt-1 text-sm text-gray-500">
          <span>{stateData.region} Region</span>
          <span>&middot;</span>
          <span>FY {latestFy}</span>
          {nComponents < 4 && (
            <>
              <span>&middot;</span>
              <span className="text-amber-600 font-medium">
                {nComponents}/4 components available
              </span>
            </>
          )}
        </div>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-8">
        <MetricCard
          label="Composite"
          value={score !== null ? score.toFixed(2) : "--"}
          trend={
            rankChange === null || rankChange === 0
              ? "neutral"
              : rankChange < 0
                ? "up"
                : "down"
          }
          trendLabel={
            rankChange === null
              ? "No prior year"
              : rankChange === 0
                ? "Unchanged"
                : `${Math.abs(rankChange)} ${rankChange < 0 ? "up" : "down"} in rank`
          }
        />
        <MetricCard
          label="GST"
          value={
            rankingEntry?.gst_zscore !== null && rankingEntry?.gst_zscore !== undefined
              ? (rankingEntry.gst_zscore >= 0 ? "+" : "") +
                rankingEntry.gst_zscore.toFixed(2)
              : "--"
          }
          subtext={
            hasScore
              ? formatIndianNumber(ann.gst_total[latestIdx]) + " Cr"
              : undefined
          }
        />
        <MetricCard
          label="Electricity"
          value={
            rankingEntry?.electricity_zscore !== null &&
            rankingEntry?.electricity_zscore !== undefined
              ? (rankingEntry.electricity_zscore >= 0 ? "+" : "") +
                rankingEntry.electricity_zscore.toFixed(2)
              : "--"
          }
          subtext={
            hasScore
              ? formatIndianNumber(ann.electricity_mu[latestIdx]) + " MU"
              : undefined
          }
          warning={
            rankingEntry?.electricity_zscore === null
              ? "Not available for this state"
              : undefined
          }
        />
        <MetricCard
          label="Credit"
          value={
            rankingEntry?.credit_zscore !== null &&
            rankingEntry?.credit_zscore !== undefined
              ? (rankingEntry.credit_zscore >= 0 ? "+" : "") +
                rankingEntry.credit_zscore.toFixed(2)
              : "--"
          }
          subtext={
            hasScore
              ? formatIndianNumber(ann.bank_credit_yoy[latestIdx]) + " Cr YoY"
              : undefined
          }
        />
        <MetricCard
          label="EPFO"
          value={
            rankingEntry?.epfo_zscore !== null &&
            rankingEntry?.epfo_zscore !== undefined
              ? (rankingEntry.epfo_zscore >= 0 ? "+" : "") +
                rankingEntry.epfo_zscore.toFixed(2)
              : "--"
          }
          subtext={
            hasScore
              ? formatIndianNumber(ann.epfo_payroll[latestIdx]) + " workers"
              : undefined
          }
        />
      </div>

      {/* Insights diagnostic banner */}
      {stateData.insights && (
        <StateDiagnosticBanner
          state={stateData.state}
          insights={stateData.insights}
          latestFy={latestFy}
        />
      )}

      {/* Charts */}
      <StateDetailClient stateData={stateData} />

      {/* Peer comparison */}
      {stateData.peers.length > 0 && (
        <div className="mt-6">
          <PeerComparison
            state={stateData.state}
            stateData={stateData}
            peers={stateData.peers}
          />
        </div>
      )}

      {/* Compare CTA */}
      <div className="mt-8 text-center">
        <Link
          href={`/compare?states=${slug}`}
          className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
        >
          Compare {stateData.state} with other states
        </Link>
      </div>
    </div>
  );
}
