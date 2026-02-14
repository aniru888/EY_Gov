import Link from "next/link";
import {
  getRankings,
  getMetadata,
  getEnhancedInsights,
  getEnhancedRegression,
} from "@/lib/data";

export default function HomePage() {
  const rankings = getRankings();
  const metadata = getMetadata();
  const insights = getEnhancedInsights();
  const regression = getEnhancedRegression();

  const fyCount = metadata.data_coverage.fiscal_years.length;
  const latestFY = rankings.fiscal_year;

  // Cross-sectional regression for the latest FY
  const cs = regression.cross_sectional?.[latestFY];
  const rSquared = cs ? cs.r_squared : 0.926;
  const fStat = cs ? cs.f_statistic : 59.3;
  const nStates = cs ? cs.n : 24;

  // Correlations
  const corr = insights.correlations;

  // Top 5
  const top5 = rankings.rankings.slice(0, 5);

  // PCA
  const pca = regression.pca;

  // GSDP comparison — outperformers and underperformers
  const gsdpComp = insights.gsdp_comparison || [];
  const outperformers = gsdpComp
    .filter((s) => s.rank_gap !== null && s.rank_gap > 0)
    .sort((a, b) => (b.rank_gap ?? 0) - (a.rank_gap ?? 0))
    .slice(0, 3);
  const underperformers = gsdpComp
    .filter((s) => s.rank_gap !== null && s.rank_gap < 0)
    .sort((a, b) => (a.rank_gap ?? 0) - (b.rank_gap ?? 0))
    .slice(0, 3);

  // Gap explanations for the insight text
  const gapExplanations = insights.gap_explanations?.all || {};

  // Regional analysis
  const regions = insights.regional_analysis?.regions || {};
  const regionEntries = Object.entries(regions)
    .map(([name, data]) => ({ name, ...data }))
    .sort((a, b) => b.mean_composite - a.mean_composite);

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      {/* Section 1: Hero */}
      <div className="mb-12">
        <h1 className="text-3xl font-bold text-gray-900">
          State Economic Activity Index
        </h1>
        <p className="mt-3 text-lg text-gray-600 max-w-3xl">
          India publishes state GDP figures 1-2 years late. This index tracks
          state-level economic activity in near-real-time through four
          hard-to-fake operational indicators — filling a gap no existing
          national tracker covers at the state level.
        </p>
        <div className="mt-3 flex flex-wrap items-center gap-3 text-sm text-gray-500">
          <span className="inline-flex items-center gap-1.5 bg-blue-50 text-blue-700 px-2.5 py-1 rounded-full text-xs font-medium">
            FY {latestFY}
          </span>
          <span className="text-gray-400">
            Data generated{" "}
            {new Date(metadata.generated_at).toLocaleDateString("en-IN", {
              day: "numeric",
              month: "short",
              year: "numeric",
            })}
          </span>
        </div>
      </div>

      {/* Section 2: The Big Validation Number */}
      <div className="mb-12 bg-blue-50 border border-blue-200 rounded-lg p-6">
        <div className="flex flex-col sm:flex-row sm:items-baseline gap-3">
          <span className="text-3xl font-bold text-blue-900">
            {(rSquared * 100).toFixed(0)}%
          </span>
          <span className="text-lg text-blue-800">
            of cross-state GDP variation explained by 4 operational indicators
          </span>
        </div>
        <p className="mt-2 text-sm text-blue-700">
          R&sup2;={rSquared.toFixed(3)}, F={fStat.toFixed(1)}, p&lt;0.001,
          N={nStates} states &mdash;{" "}
          <Link
            href="/insights"
            className="underline hover:text-blue-900"
          >
            Full regression analysis
          </Link>
        </p>
      </div>

      {/* Section 3: The Four Indicators */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Four Indicators
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <IndicatorCard
            name="GST Collections"
            measures="Formal transactions"
            correlation={corr.gst_gsdp.r}
          />
          <IndicatorCard
            name="Electricity Demand"
            measures="Physical output"
            correlation={corr.electricity_gsdp.r}
          />
          <IndicatorCard
            name="Bank Credit"
            measures="Investment activity"
            correlation={corr.credit_gsdp.r}
          />
          <IndicatorCard
            name="EPFO Payroll"
            measures="Formal employment"
            correlation={corr.epfo_gsdp.r}
          />
        </div>
      </div>

      {/* Section 4: Coverage Strip */}
      <div className="mb-12 flex flex-wrap items-center justify-center gap-x-8 gap-y-3 py-4 border-y border-gray-200 text-sm text-gray-600">
        <Stat value={rankings.count} label="states ranked" />
        <Stat value={fyCount} label="fiscal years (2017-25)" />
        <Stat value={4} label="components" />
        <span>
          Updated{" "}
          <span className="font-semibold text-gray-900">FY {latestFY}</span>
        </span>
      </div>

      {/* Section 5: Top 5 Leaderboard */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Top 5 States
        </h2>
        <ol className="space-y-2">
          {top5.map((r, i) => (
            <li key={r.slug}>
              <Link
                href={`/states/${r.slug}`}
                className="flex items-center gap-3 p-3 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <span className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center text-sm font-bold text-gray-700">
                  {i + 1}
                </span>
                <span className="font-medium text-gray-900">{r.state}</span>
                <span className="ml-auto text-sm text-gray-500 tabular-nums">
                  {r.composite_score.toFixed(2)}
                </span>
              </Link>
            </li>
          ))}
        </ol>
        <p className="mt-3 text-sm text-gray-500">
          Rankings reflect observable economic machinery, not GDP size.{" "}
          <Link
            href="/rankings"
            className="text-blue-600 hover:underline"
          >
            View full rankings
          </Link>
        </p>
      </div>

      {/* Section 6: What the Index Reveals */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">
          What the Index Reveals
        </h2>
        <div className="space-y-6">
          {/* Insight 1: Electricity */}
          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <h3 className="text-lg font-semibold text-gray-900">
              Electricity is the single strongest individual predictor
            </h3>
            <p className="mt-2 text-gray-700">
              r={corr.electricity_gsdp.r?.toFixed(3)} with GSDP — the highest
              of all 4 components. This mirrors the original Li Keqiang Index
              which gave electricity the joint-highest weight (40%). A
              ScienceDirect study of 18 Indian states over 1960-2015 confirmed
              long-run causal links between electricity consumption and state
              GDP.
            </p>
            <p className="mt-2 text-sm text-gray-500">
              GST r={corr.gst_gsdp.r?.toFixed(3)}, Credit r=
              {corr.credit_gsdp.r?.toFixed(3)}, EPFO r=
              {corr.epfo_gsdp.r?.toFixed(3)}. The composite (r=
              {corr.composite_gsdp.r?.toFixed(3)}) barely edges past
              electricity alone.{" "}
              <Link
                href="/electricity"
                className="text-blue-600 hover:underline"
              >
                Electricity deep-dive
              </Link>
            </p>
          </div>

          {/* Insight 2: Gap diagnostic */}
          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <h3 className="text-lg font-semibold text-gray-900">
              The gap between index and GDP rank is a structural diagnostic
            </h3>
            <p className="mt-2 text-gray-700">
              {outperformers.length > 0 && (
                <>
                  {outperformers[0].state} ranks #{outperformers[0].index_rank}{" "}
                  on our index vs #{outperformers[0].gsdp_rank} on GDP (gap ={" "}
                  +{outperformers[0].rank_gap}).{" "}
                </>
              )}
              {underperformers.length > 0 && (
                <>
                  {underperformers[0].state} ranks #
                  {underperformers[0].index_rank} vs #
                  {underperformers[0].gsdp_rank} on GDP (gap ={" "}
                  {underperformers[0].rank_gap}).{" "}
                </>
              )}
              The divergence maps onto economic structure: formalization,
              industrial base, agricultural dependence.
            </p>
            <p className="mt-2 text-sm text-gray-500">
              The 7% the index misses is as revealing as the 93% it captures.{" "}
              <Link
                href="/insights"
                className="text-blue-600 hover:underline"
              >
                Full gap analysis
              </Link>
            </p>
          </div>

          {/* Insight 3: PCA */}
          {pca && !pca.skipped && (
            <div className="bg-white border border-gray-200 rounded-lg p-5">
              <h3 className="text-lg font-semibold text-gray-900">
                Equal weights are statistically justified
              </h3>
              <p className="mt-2 text-gray-700">
                PCA-derived weights produce identical rankings (Spearman rho ={" "}
                {pca.rank_correlation_with_equal_weights.toFixed(3)}). PC1
                explains {pca.pc1_variance_explained_pct.toFixed(1)}% of
                variance. PCA-implied weights — GST{" "}
                {(pca.implied_weights.gst * 100).toFixed(1)}%, Electricity{" "}
                {(pca.implied_weights.electricity * 100).toFixed(1)}%, Credit{" "}
                {(pca.implied_weights.credit * 100).toFixed(1)}%, EPFO{" "}
                {(pca.implied_weights.epfo * 100).toFixed(1)}% — are remarkably
                close to equal 25%.
              </p>
              <p className="mt-2 text-sm text-gray-500">
                No need for complex weighting. The data confirms simplicity
                works.{" "}
                <Link
                  href="/methodology"
                  className="text-blue-600 hover:underline"
                >
                  Methodology details
                </Link>
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Section 7: States to Watch */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          States to Watch
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Outperformers */}
          <div>
            <h3 className="text-sm font-semibold text-emerald-700 uppercase tracking-wide mb-3">
              Index Outperformers (index &gt; GDP rank)
            </h3>
            <ul className="space-y-3">
              {outperformers.map((s) => {
                const explanation =
                  gapExplanations[s.slug]?.explanation || "";
                const shortExplanation = explanation
                  ? explanation.split(". ").slice(1, 2).join(". ")
                  : "";
                return (
                  <li key={s.slug}>
                    <Link
                      href={`/states/${s.slug}`}
                      className="block p-3 rounded-lg border border-emerald-200 bg-emerald-50 hover:bg-emerald-100 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-gray-900">
                          {s.state}
                        </span>
                        <span className="text-sm font-semibold text-emerald-700">
                          +{s.rank_gap}
                        </span>
                      </div>
                      {shortExplanation && (
                        <p className="mt-1 text-sm text-gray-600">
                          {shortExplanation}
                        </p>
                      )}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>

          {/* Underperformers */}
          <div>
            <h3 className="text-sm font-semibold text-amber-700 uppercase tracking-wide mb-3">
              GDP Outperformers (GDP &gt; index rank)
            </h3>
            <ul className="space-y-3">
              {underperformers.map((s) => {
                const explanation =
                  gapExplanations[s.slug]?.explanation || "";
                const shortExplanation = explanation
                  ? explanation.split(". ").slice(1, 2).join(". ")
                  : "";
                return (
                  <li key={s.slug}>
                    <Link
                      href={`/states/${s.slug}`}
                      className="block p-3 rounded-lg border border-amber-200 bg-amber-50 hover:bg-amber-100 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-gray-900">
                          {s.state}
                        </span>
                        <span className="text-sm font-semibold text-amber-700">
                          {s.rank_gap}
                        </span>
                      </div>
                      {shortExplanation && (
                        <p className="mt-1 text-sm text-gray-600">
                          {shortExplanation}
                        </p>
                      )}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        </div>
      </div>

      {/* Section 8: Regional Divergence */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Regional Divergence
        </h2>
        <div className="space-y-2">
          {regionEntries.map((region) => {
            const maxVal = Math.max(
              ...regionEntries.map((r) => Math.abs(r.mean_composite))
            );
            const barWidth =
              maxVal > 0
                ? Math.abs(region.mean_composite / maxVal) * 100
                : 0;
            const isPositive = region.mean_composite >= 0;

            return (
              <div key={region.name} className="flex items-center gap-3">
                <span className="w-28 text-sm text-gray-700 text-right shrink-0">
                  {region.name}
                </span>
                <div className="flex-1 flex items-center gap-2">
                  <div className="flex-1 h-5 bg-gray-100 rounded-full overflow-hidden relative">
                    <div
                      className={`h-full rounded-full ${
                        isPositive ? "bg-blue-500" : "bg-gray-400"
                      }`}
                      style={{ width: `${Math.max(barWidth, 2)}%` }}
                    />
                  </div>
                  <span
                    className={`text-sm tabular-nums w-14 text-right ${
                      isPositive ? "text-blue-700" : "text-gray-500"
                    }`}
                  >
                    {region.mean_composite >= 0 ? "+" : ""}
                    {region.mean_composite.toFixed(2)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
        <p className="mt-3 text-sm text-gray-500">
          Mean composite score by region.{" "}
          India&apos;s economic activity remains concentrated in western and southern
          corridors.{" "}
          <Link href="/compare" className="text-blue-600 hover:underline">
            Compare states
          </Link>
        </p>
      </div>

      {/* Section 9: Navigate Deeper */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Explore Further
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <NavCard
            href="/rankings"
            title="Rankings"
            description="Full sortable table with all 34 states, trend charts, and component breakdowns"
          />
          <NavCard
            href="/electricity"
            title="Electricity Deep-Dive"
            description="The strongest single predictor — intensity, elasticity, and state-level patterns"
          />
          <NavCard
            href="/insights"
            title="Insights"
            description="Regression analysis, GDP gaps, COVID recovery, and component diagnostics"
          />
          <NavCard
            href="/methodology"
            title="Methodology"
            description="How the index is built: data sources, normalization, weighting, and limitations"
          />
          <NavCard
            href="/compare"
            title="Compare States"
            description="Side-by-side comparison of any two states across all four components"
          />
        </div>
      </div>
    </div>
  );
}

/* ---------- Helper components ---------- */

function IndicatorCard({
  name,
  measures,
  correlation,
}: {
  name: string;
  measures: string;
  correlation: number | null;
}) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <h3 className="font-semibold text-gray-900">{name}</h3>
      <p className="text-sm text-gray-500 mt-1">{measures}</p>
      {correlation !== null && (
        <p className="text-sm text-gray-700 mt-2 tabular-nums">
          r = {correlation.toFixed(2)} with GSDP
        </p>
      )}
    </div>
  );
}

function Stat({ value, label }: { value: number; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-lg font-bold text-gray-900">{value}</span>
      <span className="text-gray-600">{label}</span>
    </div>
  );
}

function NavCard({
  href,
  title,
  description,
}: {
  href: string;
  title: string;
  description: string;
}) {
  return (
    <Link
      href={href}
      className="block p-4 rounded-lg border border-gray-200 hover:border-blue-300 hover:bg-blue-50 transition-colors"
    >
      <h3 className="font-semibold text-gray-900">{title}</h3>
      <p className="text-sm text-gray-500 mt-1">{description}</p>
    </Link>
  );
}
