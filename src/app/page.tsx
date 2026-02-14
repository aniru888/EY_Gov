import Link from "next/link";
import {
  getRankings,
  getMetadata,
  getEnhancedInsights,
  getEnhancedRegression,
  getPerformanceData,
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

  // Stepwise regression findings
  const stepwise = regression.stepwise;
  const gstOnlyR2 = stepwise?.["model_1_gst_only"]?.r_squared;
  const gstElecR2 = stepwise?.["model_2_gst_elec"]?.r_squared;

  // Without Maharashtra
  const withoutMH = cs?.without_maharashtra;

  // Electricity coefficient
  const elecCoef = cs?.coefficients?.["electricity_mu"];

  // Correlations
  const corr = insights.correlations;

  // Top 5
  const top5 = rankings.rankings.slice(0, 5);

  // Maharashtra outlier data
  const mh = rankings.rankings[0];
  const nextBest = rankings.rankings[1];

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

  // Population analysis (per-capita)
  const popAnalysis = insights.population_analysis;
  const perfRisers = popAnalysis?.biggest_perf_risers?.slice(0, 3) || [];
  const perfFallers = popAnalysis?.biggest_perf_fallers?.slice(0, 2) || [];

  // Haryana momentum data
  const haryanaGrowth = insights.growth_zscores?.rankings?.find(
    (s) => s.slug === "haryana"
  );
  const haryanaActivity = rankings.rankings.find(
    (s) => s.slug === "haryana"
  );
  const haryanaGsdp = gsdpComp.find((s) => s.slug === "haryana");
  const performance = getPerformanceData();
  const haryanaPerf = performance.rankings.find(
    (s) => s.slug === "haryana"
  );

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      {/* Section 1: Hero */}
      <div className="mb-12">
        <h1 className="text-3xl font-bold text-gray-900">
          State Economic Activity Index
        </h1>
        <p className="mt-3 text-lg text-gray-600 max-w-3xl">
          India&apos;s state GDP figures arrive 1&ndash;2 years late and get
          revised repeatedly. This index tracks state-level economic activity
          through four hard-to-fake operational signals &mdash; GST collections,
          electricity demand, bank credit, and formal employment &mdash; that
          together explain{" "}
          <span className="font-semibold text-gray-900">
            {(rSquared * 100).toFixed(0)}%
          </span>{" "}
          of cross-state GDP variation anyway.
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
          {gstOnlyR2 != null && gstElecR2 != null && (
            <>
              GST alone explains {(gstOnlyR2 * 100).toFixed(0)}%.
              Add electricity and R&sup2; jumps to{" "}
              {(gstElecR2 * 100).toFixed(0)}%.
              Bank credit and EPFO together contribute less than 0.1 percentage
              points &mdash; the two original Li Keqiang indicators do nearly
              all the work.{" "}
            </>
          )}
          R&sup2;={rSquared.toFixed(3)}, F={fStat.toFixed(1)}, p&lt;0.001,
          N={nStates} states.{" "}
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
            name="Electricity Demand"
            measures="Physical output"
            correlation={corr.electricity_gsdp.r}
            signal="primary"
          />
          <IndicatorCard
            name="GST Collections"
            measures="Formal transactions"
            correlation={corr.gst_gsdp.r}
            signal="primary"
          />
          <IndicatorCard
            name="Bank Credit"
            measures="Investment activity"
            correlation={corr.credit_gsdp.r}
            signal="complementary"
          />
          <IndicatorCard
            name="EPFO Payroll"
            measures="Formal employment"
            correlation={corr.epfo_gsdp.r}
            signal="complementary"
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
          {/* Insight 1: Electricity is the dominant predictor */}
          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <h3 className="text-lg font-semibold text-gray-900">
              Electricity is the only individually significant predictor of state
              GDP
            </h3>
            <p className="mt-2 text-gray-700">
              r={corr.electricity_gsdp.r?.toFixed(3)} with GSDP &mdash; the
              highest of all 4 components.{" "}
              {elecCoef && (
                <>
                  In the multivariate regression, electricity carries a
                  standardised coefficient of {elecCoef.beta?.toFixed(2)}{" "}
                  (p&lt;0.001) &mdash; the only individually significant
                  predictor.{" "}
                </>
              )}
              {gstOnlyR2 != null && gstElecR2 != null && (
                <>
                  Adding electricity to a GST-only model jumps R&sup2; from{" "}
                  {(gstOnlyR2 * 100).toFixed(0)}% to{" "}
                  {(gstElecR2 * 100).toFixed(0)}%.
                </>
              )}{" "}
              Tiwari, Eapen &amp; Nair (2021) found r=0.906 across 18 states
              over 55 years and confirmed that economic growth Granger-causes
              electricity demand. It is the original Li Keqiang indicator for a
              reason.
            </p>
            <p className="mt-2 text-sm text-gray-500">
              GST r={corr.gst_gsdp.r?.toFixed(3)}, Credit r=
              {corr.credit_gsdp.r?.toFixed(3)}, EPFO r=
              {corr.epfo_gsdp.r?.toFixed(3)}. High multicollinearity (VIF
              3&ndash;26) means individual coefficients are unreliable, but the
              joint F-test is overwhelming.{" "}
              <Link
                href="/electricity"
                className="text-blue-600 hover:underline"
              >
                Electricity deep-dive
              </Link>{" "}
              | Ref: Tiwari et al., <em>Energy Economics</em> 94 (2021)
            </p>
          </div>

          {/* Insight 2: Maharashtra is a 4-sigma outlier */}
          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <h3 className="text-lg font-semibold text-gray-900">
              Maharashtra is a {mh.composite_score.toFixed(0)}-sigma outlier
              &mdash; and removing it improves the model
            </h3>
            <p className="mt-2 text-gray-700">
              Maharashtra&apos;s composite score of{" "}
              {mh.composite_score.toFixed(2)} is nearly{" "}
              {(mh.composite_score / nextBest.composite_score).toFixed(0)}x the
              next-best state ({nextBest.state} at{" "}
              {nextBest.composite_score.toFixed(2)}).{" "}
              {withoutMH && (
                <>
                  Remove it and R&sup2; <em>improves</em> from{" "}
                  {(rSquared * 100).toFixed(1)}% to{" "}
                  {(withoutMH.r_squared * 100).toFixed(1)}%.
                </>
              )}{" "}
              Mumbai&apos;s outsized bank credit (credit z-score{" "}
              {mh.credit_zscore?.toFixed(1) ?? "--"} vs electricity z-score{" "}
              {mh.electricity_zscore?.toFixed(1) ?? "--"}) inflates one component well
              beyond its actual industrial footprint, adding noise rather than
              signal.
            </p>
            <p className="mt-2 text-sm text-gray-500">
              This is not a flaw in the index &mdash; it is a finding.
              Maharashtra&apos;s economy is uniquely financialised in a way that
              distorts a physical-activity-based composite.{" "}
              <Link
                href="/states/maharashtra"
                className="text-blue-600 hover:underline"
              >
                Maharashtra profile
              </Link>{" "}
              |{" "}
              <Link
                href="/insights"
                className="text-blue-600 hover:underline"
              >
                Full diagnostics
              </Link>
            </p>
          </div>

          {/* Insight 3: Per-capita divergence */}
          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <h3 className="text-lg font-semibold text-gray-900">
              Per-capita rankings invert the picture: small states lead, big
              states fall
            </h3>
            <p className="mt-2 text-gray-700">
              {perfRisers.length > 0 && (
                <>
                  Normalise by population and the rankings transform:{" "}
                  {perfRisers.map((s, i) => (
                    <span key={s.slug}>
                      {s.state} {s.activity_rank}&rarr;{s.perf_rank} (+
                      {s.gap})
                      {i < perfRisers.length - 1 ? ", " : ". "}
                    </span>
                  ))}
                </>
              )}
              {perfFallers.length > 0 && (
                <>
                  Meanwhile{" "}
                  {perfFallers.map((s, i) => (
                    <span key={s.slug}>
                      {s.state} drops {s.activity_rank}&rarr;{s.perf_rank} ({s.gap})
                      {i < perfFallers.length - 1 ? ", " : ". "}
                    </span>
                  ))}
                </>
              )}
              The Activity Index tells you <em>where</em> the economy is. The
              Performance Index tells you how much formal economic activity each
              person generates.
            </p>
            <p className="mt-2 text-sm text-gray-500">
              {popAnalysis && (
                <>
                  Activity rank correlates{" "}
                  {Math.abs(
                    popAnalysis.activity_rank_vs_population.spearman_rho * 100
                  ).toFixed(0)}
                  % with population (Spearman rho=
                  {popAnalysis.activity_rank_vs_population.spearman_rho.toFixed(
                    2
                  )}
                  ).
                  {popAnalysis.perf_rank_vs_population && (
                    <>
                      {" "}Per-capita normalization drops this to rho=
                      {popAnalysis.perf_rank_vs_population.spearman_rho.toFixed(2)}.
                    </>
                  )}{" "}
                </>
              )}
              <Link
                href="/rankings"
                className="text-blue-600 hover:underline"
              >
                View Performance rankings
              </Link>
            </p>
          </div>

          {/* Insight 4: What the index can't see */}
          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <h3 className="text-lg font-semibold text-gray-900">
              What the index cannot see &mdash; and where it partially overlaps
            </h3>
            <p className="mt-2 text-gray-700">
              <strong>Agriculture:</strong> Not directly captured, but partially
              overlaps with electricity demand in groundwater-irrigated states
              (Punjab, Haryana, UP) where electric pump-sets drive consumption.
              Rain-fed agriculture (Kerala, West Bengal) is truly invisible.{" "}
              <strong>Informal economy:</strong> Not captured by EPFO or GST
              directly, but informal businesses still consume electricity and
              some pay GST &mdash; states with larger informal sectors (Bihar,
              UP) are more undercounted.{" "}
              <strong>Government spending:</strong> Infrastructure capex shows up
              indirectly through electricity demand and bank credit, but current
              expenditure (salaries, transfers, welfare) is invisible. NE states
              and Bihar are systematically undercounted.{" "}
              <strong>Remittances:</strong> Completely invisible &mdash;
              Kerala&apos;s economy is significantly powered by Gulf remittances
              that none of our 4 indicators capture.
            </p>
            <p className="mt-2 text-sm text-gray-500">
              States where GSDP exceeds the index (Kerala, Tripura, West Bengal)
              have GDP from channels with minimal overlap to our indicators.
              States where the index exceeds GSDP (Haryana, Delhi) have
              formalization momentum the GDP figures haven&apos;t caught up with
              yet.{" "}
              <Link
                href="/methodology"
                className="text-blue-600 hover:underline"
              >
                Methodology &amp; limitations
              </Link>
            </p>
          </div>
        </div>
      </div>

      {/* Section 7: States to Watch */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          States to Watch
        </h2>

        {/* Haryana spotlight */}
        {haryanaActivity && (
          <div className="mb-6 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-5">
            <div className="flex items-center gap-2 mb-2">
              <h3 className="text-lg font-semibold text-gray-900">
                Haryana
              </h3>
              <span className="text-xs font-medium bg-blue-100 text-blue-800 px-2 py-0.5 rounded-full">
                Momentum leader
              </span>
            </div>
            <p className="text-gray-700">
              Activity rank #{haryanaActivity.rank}
              {haryanaPerf && <>, per-capita rank #{haryanaPerf.perf_rank}</>}
              {haryanaGsdp?.rank_gap != null &&
                haryanaGsdp.rank_gap > 0 && (
                  <>
                    , +{haryanaGsdp.rank_gap} gap vs GSDP
                  </>
                )}
              .{" "}
              {haryanaGrowth && (
                <>
                  GST growth{" "}
                  {haryanaGrowth.gst_yoy_pct != null
                    ? `+${haryanaGrowth.gst_yoy_pct.toFixed(0)}%`
                    : "--"}
                  , EPFO growth{" "}
                  {haryanaGrowth.epfo_yoy_pct != null
                    ? `+${haryanaGrowth.epfo_yoy_pct.toFixed(1)}%`
                    : "--"}
                  .{" "}
                </>
              )}
              Tiwari et al. (2021) found Haryana had the highest economic growth
              CAGR of all 18 states studied over 55 years. The Gurgaon/NCR belt
              drives formal employment and GST-registered activity well beyond
              what Haryana&apos;s GSDP rank would suggest.
            </p>
            <Link
              href="/states/haryana"
              className="mt-2 inline-block text-sm text-blue-600 hover:underline"
            >
              Haryana profile
            </Link>
          </div>
        )}

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
            href="/rankings"
            title="Performance Rankings"
            description="Per-capita index — which states generate the most formal activity per person?"
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
  signal,
}: {
  name: string;
  measures: string;
  correlation: number | null;
  signal?: "primary" | "complementary";
}) {
  return (
    <div
      className={`bg-white border rounded-lg p-4 ${
        signal === "primary"
          ? "border-blue-300 ring-1 ring-blue-100"
          : "border-gray-200"
      }`}
    >
      <div className="flex items-center gap-2">
        <h3 className="font-semibold text-gray-900">{name}</h3>
        {signal === "primary" && (
          <span className="text-[10px] font-semibold uppercase tracking-wider text-blue-600 bg-blue-50 px-1.5 py-0.5 rounded">
            Primary
          </span>
        )}
      </div>
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
