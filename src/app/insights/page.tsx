import type { Metadata } from "next";
import Link from "next/link";
import { getEnhancedInsights, getEnhancedRegression } from "@/lib/data";
import Breadcrumbs from "@/components/common/Breadcrumbs";
import InsightsClient from "@/components/InsightsClient";

export const metadata: Metadata = {
  title: "Insights & Analysis | State Economic Activity Index",
  description:
    "Key findings, GSDP comparison, growth dynamics, COVID recovery, panel econometrics, and statistical validation of the Li Keqiang Index for Indian States.",
};

const FINDING_COLORS: Record<string, string> = {
  fastest_rising: "border-emerald-400 bg-emerald-50",
  outperformer: "border-blue-400 bg-blue-50",
  underperformer: "border-amber-400 bg-amber-50",
  fastest_recovery: "border-teal-400 bg-teal-50",
  most_unbalanced: "border-purple-400 bg-purple-50",
  strongest_correlation: "border-indigo-400 bg-indigo-50",
};

const VAR_LABELS: Record<string, string> = {
  gst_total: "GST Collections",
  electricity_mu: "Electricity Demand",
  bank_credit_yoy: "Bank Credit (YoY)",
  epfo_payroll: "EPFO Payroll",
};

export default function InsightsPage() {
  const insights = getEnhancedInsights();
  const regression = getEnhancedRegression();

  const latestFy = insights.latest_fy;
  const cs = regression.cross_sectional?.[regression.latest_fy_with_gsdp || ""];
  const panelFe = regression.panel_fe;
  const logLog = regression.log_log;
  const lagged = regression.lagged;
  const pca = regression.pca;
  const gapExplanations = insights.gap_explanations;
  const regional = insights.regional_analysis;

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <Breadcrumbs
        items={[{ label: "Home", href: "/" }, { label: "Insights" }]}
      />

      <h1 className="text-3xl font-bold text-gray-900 mb-2">
        Insights & Analysis
      </h1>
      <p className="text-gray-600 mb-8 max-w-3xl">
        Our 4 indicators explain {cs ? `~${(cs.r_squared * 100).toFixed(0)}%` : "most"} of
        GDP variation. But the{" "}
        {cs ? `${(100 - cs.r_squared * 100).toFixed(0)}%` : "remainder"} they
        miss reveals economic structure: agriculture, informality, financial hub
        effects. FY {latestFy} data.
      </p>

      {/* Section 1: Key Findings */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Key Findings</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {insights.key_findings.map((finding) => (
            <div
              key={finding.type}
              className={`border-l-4 rounded-lg p-4 ${
                FINDING_COLORS[finding.type] || "border-gray-400 bg-gray-50"
              }`}
            >
              <h3 className="font-semibold text-gray-900 text-sm mb-1">
                {finding.title}
              </h3>
              <p className="text-sm text-gray-700">{finding.detail}</p>
              {finding.states.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {finding.states.slice(0, 3).map((slug) => (
                    <Link
                      key={slug}
                      href={`/states/${slug}`}
                      className="text-xs text-blue-600 hover:underline"
                    >
                      View details &rarr;
                    </Link>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Section 2: Statistical Validation (cross-sectional) */}
      {cs && !regression.skipped && (
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Statistical Validation
          </h2>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-5 mb-6">
            <div className="text-3xl font-bold text-blue-800">
              {(cs.r_squared * 100).toFixed(0)}%
            </div>
            <p className="text-blue-700 text-sm mt-1">
              of cross-state GDP variation explained by our 4 indicators
              (R&sup2;={cs.r_squared.toFixed(3)}, F={cs.f_statistic.toFixed(1)},
              p&lt;0.001, N={cs.n} states, FY {regression.latest_fy_with_gsdp})
            </p>
          </div>

          {/* Stepwise comparison */}
          {regression.stepwise && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-3">
                Does each component add information?
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm border border-gray-200 rounded-lg">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="text-left px-4 py-2 font-medium text-gray-700">Model</th>
                      <th className="text-right px-4 py-2 font-medium text-gray-700">R&sup2;</th>
                      <th className="text-right px-4 py-2 font-medium text-gray-700">&Delta;R&sup2;</th>
                      <th className="text-right px-4 py-2 font-medium text-gray-700">Partial F</th>
                      <th className="text-center px-4 py-2 font-medium text-gray-700">Significant?</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {Object.entries(regression.stepwise).map(([key, step]) => (
                      <tr key={key}>
                        <td className="px-4 py-2 font-medium text-gray-900">
                          {key
                            .replace("model_1_gst_only", "1. GST only")
                            .replace("model_2_gst_elec", "2. + Electricity")
                            .replace("model_3_gst_elec_credit", "3. + Bank Credit")
                            .replace("model_4_full", "4. + EPFO (full)")}
                        </td>
                        <td className="px-4 py-2 text-right tabular-nums">{step.r_squared.toFixed(3)}</td>
                        <td className="px-4 py-2 text-right tabular-nums text-gray-500">
                          {step.delta_r2 != null ? `+${step.delta_r2.toFixed(3)}` : "--"}
                        </td>
                        <td className="px-4 py-2 text-right tabular-nums text-gray-500">
                          {step.partial_f != null ? step.partial_f.toFixed(1) : "--"}
                        </td>
                        <td className="px-4 py-2 text-center">
                          {step.partial_f_p != null ? (
                            step.partial_f_p < 0.05 ? (
                              <span className="text-emerald-600 font-medium">
                                Yes (p={step.partial_f_p < 0.001 ? "<0.001" : step.partial_f_p.toFixed(3)})
                              </span>
                            ) : (
                              <span className="text-gray-400">No (p={step.partial_f_p.toFixed(3)})</span>
                            )
                          ) : (
                            <span className="text-gray-400">Baseline</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Beta weights */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">
              Standardized Coefficients (Beta Weights)
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm border border-gray-200 rounded-lg">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="text-left px-4 py-2 font-medium text-gray-700">Variable</th>
                    <th className="text-right px-4 py-2 font-medium text-gray-700">Beta</th>
                    <th className="text-right px-4 py-2 font-medium text-gray-700">p-value</th>
                    <th className="text-right px-4 py-2 font-medium text-gray-700">VIF</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {Object.entries(VAR_LABELS).map(([v, label]) => {
                    const c = cs.coefficients[v];
                    const vif = typeof cs.diagnostics.vif === "object" ? cs.diagnostics.vif[v] : null;
                    return (
                      <tr key={v}>
                        <td className="px-4 py-2 font-medium text-gray-900">{label}</td>
                        <td className="px-4 py-2 text-right tabular-nums font-medium">
                          {c?.beta != null ? c.beta.toFixed(3) : "--"}
                        </td>
                        <td className="px-4 py-2 text-right tabular-nums">
                          <span className={c && c.p < 0.05 ? "text-emerald-600 font-medium" : "text-gray-400"}>
                            {c ? (c.p < 0.001 ? "<0.001" : c.p.toFixed(3)) : "--"}
                            {c && c.p < 0.001 ? " ***" : c && c.p < 0.01 ? " **" : c && c.p < 0.05 ? " *" : ""}
                          </span>
                        </td>
                        <td className="px-4 py-2 text-right tabular-nums">
                          <span className={vif && vif > 10 ? "text-red-600 font-medium" : "text-gray-500"}>
                            {vif != null ? vif.toFixed(1) : "--"}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Warnings */}
          {cs.without_maharashtra && (
            <div className="mb-6 bg-amber-50 border border-amber-200 rounded-lg p-4">
              <h4 className="font-semibold text-amber-900 text-sm mb-1">Maharashtra Sensitivity Check</h4>
              <p className="text-sm text-amber-800">
                With Maharashtra: R&sup2; = {cs.r_squared.toFixed(3)} (N={cs.n}).
                Without: R&sup2; = {cs.without_maharashtra.r_squared.toFixed(3)} (N={cs.without_maharashtra.n}).
                {cs.without_maharashtra.note && <span> {cs.without_maharashtra.note}</span>}
              </p>
            </div>
          )}
          {cs.diagnostics.vif_warning && (
            <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
              <h4 className="font-semibold text-red-900 text-sm mb-1">Multicollinearity Warning</h4>
              <p className="text-sm text-red-800">
                {cs.diagnostics.vif_warning} Joint significance (F-test) remains valid.
              </p>
            </div>
          )}

          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-sm text-gray-600">
            <strong>Important:</strong> These are cross-sectional associations, not causal relationships.
          </div>
        </section>
      )}

      {/* Section 2b: Panel Fixed Effects */}
      {panelFe && !panelFe.skipped && (
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Within-State Dynamics (Panel Fixed Effects)
          </h2>
          <p className="text-sm text-gray-600 mb-4">
            Cross-sectional analysis shows <em>between-state</em> associations.
            Panel fixed effects answer a different question: when electricity demand
            rises <em>within a state</em> over time, does its GSDP rise too?
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
            <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-indigo-800 tabular-nums">
                {(panelFe.within_r_squared * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-indigo-600 mt-1">Within-R&sup2;</div>
              <div className="text-xs text-indigo-500">Variation within states over time</div>
            </div>
            <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-indigo-800 tabular-nums">
                {panelFe.between_r_squared != null ? (panelFe.between_r_squared * 100).toFixed(0) : "--"}%
              </div>
              <div className="text-xs text-indigo-600 mt-1">Between-R&sup2;</div>
              <div className="text-xs text-indigo-500">Variation between states</div>
            </div>
            <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-indigo-800 tabular-nums">
                {panelFe.n}
              </div>
              <div className="text-xs text-indigo-600 mt-1">Observations</div>
              <div className="text-xs text-indigo-500">{panelFe.n_states} states x {panelFe.n_years} years</div>
            </div>
          </div>

          {panelFe.coefficients && Object.keys(panelFe.coefficients).length > 0 && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-3">Panel FE Coefficients</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm border border-gray-200 rounded-lg">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="text-left px-4 py-2 font-medium text-gray-700">Variable</th>
                      <th className="text-right px-4 py-2 font-medium text-gray-700">Coefficient</th>
                      <th className="text-right px-4 py-2 font-medium text-gray-700">Std Error</th>
                      <th className="text-right px-4 py-2 font-medium text-gray-700">p-value</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {Object.entries(panelFe.coefficients).map(([v, c]) => (
                      <tr key={v}>
                        <td className="px-4 py-2 font-medium text-gray-900">{VAR_LABELS[v] || v}</td>
                        <td className="px-4 py-2 text-right tabular-nums">{c.coef.toFixed(4)}</td>
                        <td className="px-4 py-2 text-right tabular-nums text-gray-500">{c.se.toFixed(4)}</td>
                        <td className="px-4 py-2 text-right tabular-nums">
                          <span className={c.p < 0.05 ? "text-emerald-600 font-medium" : "text-gray-400"}>
                            {c.p < 0.001 ? "<0.001" : c.p.toFixed(3)}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {panelFe.hausman_test && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-sm text-gray-600">
              <strong>Hausman Test:</strong>{" "}
              {panelFe.hausman_test.stat != null
                ? `Chi-squared = ${panelFe.hausman_test.stat.toFixed(2)}, p = ${panelFe.hausman_test.p?.toFixed(3)}. Preferred model: ${panelFe.hausman_test.preferred}.`
                : panelFe.hausman_test.note}
              {" "}State and year fixed effects absorb time-invariant characteristics and national trends.
            </div>
          )}
        </section>
      )}

      {/* Section 2c: Log-Log Elasticities */}
      {logLog && !logLog.skipped && logLog.cross_sectional && (
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Log-Log Elasticities
          </h2>
          <p className="text-sm text-gray-600 mb-4">
            In economics, a log-log model gives <em>elasticities</em>: a 1% increase
            in the indicator is associated with a &beta;% increase in GSDP.
            R&sup2; = {logLog.cross_sectional.r_squared.toFixed(3)} (FY {logLog.cross_sectional.fiscal_year}).
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
            {Object.entries(logLog.cross_sectional.elasticities).map(([v, e]) => (
              <div key={v} className="bg-white border border-gray-200 rounded-lg p-3 text-center">
                <div className="text-xs text-gray-500 mb-1">{VAR_LABELS[v] || v}</div>
                <div className="text-2xl font-bold text-gray-900 tabular-nums">
                  {e.elasticity.toFixed(2)}
                </div>
                <div className="text-xs text-gray-400">
                  {e.p < 0.001 ? "p<0.001" : e.p < 0.05 ? `p=${e.p.toFixed(3)}` : `p=${e.p.toFixed(2)} (ns)`}
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-500">
            Log-log handles the Maharashtra outlier problem (log compresses large values).
            Coefficients are directly interpretable as % change in GSDP per 1% change in indicator.
          </p>
        </section>
      )}

      {/* Section 2d: PCA Robustness */}
      {pca && !pca.skipped && (
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Robustness: PCA vs Equal Weights
          </h2>
          <p className="text-sm text-gray-600 mb-4">
            Do data-driven weights produce different rankings than our equal-weight approach?
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
            <div className="bg-teal-50 border border-teal-200 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-teal-800 tabular-nums">
                {pca.pc1_variance_explained_pct}%
              </div>
              <div className="text-xs text-teal-600 mt-1">PC1 Variance Explained</div>
            </div>
            <div className="bg-teal-50 border border-teal-200 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-teal-800 tabular-nums">
                {pca.rank_correlation_with_equal_weights.toFixed(3)}
              </div>
              <div className="text-xs text-teal-600 mt-1">Spearman Rank Correlation</div>
              <div className="text-xs text-teal-500">PCA vs equal-weight rankings</div>
            </div>
            <div className="bg-teal-50 border border-teal-200 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-teal-800 tabular-nums">
                {pca.n}
              </div>
              <div className="text-xs text-teal-600 mt-1">States (FY {pca.fiscal_year})</div>
            </div>
          </div>

          {pca.implied_weights && (
            <div className="mb-4">
              <h3 className="text-sm font-semibold text-gray-900 mb-2">PCA-Implied Weights</h3>
              <div className="grid grid-cols-4 gap-2">
                {Object.entries(pca.implied_weights).map(([comp, w]) => (
                  <div key={comp} className="text-center">
                    <div className="text-xs text-gray-500 capitalize">{comp}</div>
                    <div className="text-sm font-bold tabular-nums">{(w * 100).toFixed(1)}%</div>
                    <div className="text-xs text-gray-400">vs 25.0% equal</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {pca.interpretation && (
            <div className="bg-teal-50 border border-teal-200 rounded-lg p-4 text-sm text-teal-800">
              {pca.interpretation}
            </div>
          )}
        </section>
      )}

      {/* Section 2e: Leading Indicator */}
      {lagged && !lagged.skipped && (
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Does Electricity Lead GDP Growth?
          </h2>
          <p className="text-sm text-gray-600 mb-4">
            We test whether states with faster electricity growth in year <em>t</em> had
            higher GSDP growth in year <em>t+1</em>. This uses {lagged.n} observations
            across {lagged.n_states} states.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="text-xs text-gray-500 mb-1">Simple Lagged Correlation</div>
              <div className="text-2xl font-bold text-gray-900 tabular-nums">
                r = {lagged.simple_correlation.r.toFixed(3)}
              </div>
              <div className="text-xs text-gray-400">
                p = {lagged.simple_correlation.p.toFixed(3)}
              </div>
            </div>
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="text-xs text-gray-500 mb-1">Panel FE Coefficient</div>
              <div className="text-2xl font-bold text-gray-900 tabular-nums">
                {lagged.panel_fe.electricity_growth_coef.toFixed(3)}
              </div>
              <div className="text-xs text-gray-400">
                p = {lagged.panel_fe.p < 0.001 ? "<0.001" : lagged.panel_fe.p.toFixed(3)}
              </div>
            </div>
          </div>

          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-sm text-gray-600">
            {lagged.interpretation}
          </div>
        </section>
      )}

      {/* Section 3: Component-GSDP Correlations */}
      {insights.correlations?.latest_fy && (
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Component-GSDP Correlations
          </h2>
          <p className="text-sm text-gray-600 mb-4">
            Pearson correlation between each component&apos;s raw value and state
            GSDP, across states in FY {insights.correlations.latest_fy}.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-4">
            {(
              [
                ["GST", insights.correlations.gst_gsdp],
                ["Electricity", insights.correlations.electricity_gsdp],
                ["Credit", insights.correlations.credit_gsdp],
                ["EPFO", insights.correlations.epfo_gsdp],
                ["Composite", insights.correlations.composite_gsdp],
              ] as [string, { r: number | null; p: number | null; n: number }][]
            ).map(([name, entry]) => (
              <div
                key={name}
                className="bg-white border border-gray-200 rounded-lg p-3 text-center"
              >
                <div className="text-xs text-gray-500 mb-1">{name}</div>
                <div className="text-2xl font-bold text-gray-900 tabular-nums">
                  {entry?.r != null ? entry.r.toFixed(2) : "--"}
                </div>
                <div className="text-xs text-gray-400">
                  {entry?.p != null
                    ? entry.p < 0.001 ? "p<0.001" : `p=${entry.p.toFixed(3)}`
                    : ""}
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-500">
            Correlation does not imply causation.
          </p>
        </section>
      )}

      {/* Section 4: Gap Explanations (replaces generic 2x2 grid) */}
      {gapExplanations && gapExplanations.all && Object.keys(gapExplanations.all).length > 0 && (
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Understanding the Gap: State Stories
          </h2>
          <p className="text-sm text-gray-600 mb-6">
            Why does our index sometimes diverge from official GSDP rankings?
            Here are the states with the largest gaps and what drives them.
          </p>

          {/* Top Outperformers */}
          {gapExplanations.top_outperformers.length > 0 && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-emerald-800 mb-3">
                Index Outperformers (rank higher on index than GSDP)
              </h3>
              <div className="space-y-3">
                {gapExplanations.top_outperformers.map((slug) => {
                  const entry = gapExplanations.all[slug];
                  if (!entry) return null;
                  return (
                    <div key={slug} className="border-l-4 border-emerald-400 bg-emerald-50 rounded-r-lg p-4">
                      <div className="flex items-baseline gap-3 mb-1">
                        <Link href={`/states/${slug}`} className="font-semibold text-gray-900 hover:text-blue-600">
                          {entry.state}
                        </Link>
                        <span className="text-sm text-emerald-700 font-medium">
                          Index #{entry.index_rank} vs GDP #{entry.gsdp_rank}
                          {entry.rank_gap != null && ` (+${entry.rank_gap})`}
                        </span>
                      </div>
                      <p className="text-sm text-gray-700">{entry.explanation}</p>
                      {entry.key_drivers.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {entry.key_drivers.map((d, i) => (
                            <span key={i} className="text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded">
                              {d}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Top Underperformers */}
          {gapExplanations.top_underperformers.length > 0 && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-amber-800 mb-3">
                GDP Outperformers (rank higher on GSDP than index)
              </h3>
              <div className="space-y-3">
                {gapExplanations.top_underperformers.map((slug) => {
                  const entry = gapExplanations.all[slug];
                  if (!entry) return null;
                  return (
                    <div key={slug} className="border-l-4 border-amber-400 bg-amber-50 rounded-r-lg p-4">
                      <div className="flex items-baseline gap-3 mb-1">
                        <Link href={`/states/${slug}`} className="font-semibold text-gray-900 hover:text-blue-600">
                          {entry.state}
                        </Link>
                        <span className="text-sm text-amber-700 font-medium">
                          Index #{entry.index_rank} vs GDP #{entry.gsdp_rank}
                          {entry.rank_gap != null && ` (${entry.rank_gap})`}
                        </span>
                      </div>
                      <p className="text-sm text-gray-700">{entry.explanation}</p>
                      {entry.key_drags.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {entry.key_drags.map((d, i) => (
                            <span key={i} className="text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded">
                              {d}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </section>
      )}

      {/* Section 5: Regional Analysis */}
      {regional && regional.regions && Object.keys(regional.regions).length > 0 && (
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Regional Patterns
          </h2>
          <p className="text-sm text-gray-600 mb-4">
            Average composite score by region (FY {regional.latest_fy}).
          </p>
          <div className="space-y-3">
            {Object.entries(regional.regions)
              .sort(([, a], [, b]) => b.mean_composite - a.mean_composite)
              .map(([region, data]) => {
                const maxScore = Math.max(
                  ...Object.values(regional.regions).map((r) => Math.abs(r.mean_composite))
                );
                const barWidth = maxScore > 0 ? Math.abs(data.mean_composite) / maxScore * 100 : 0;
                const isPositive = data.mean_composite >= 0;
                return (
                  <div key={region} className="flex items-center gap-3">
                    <div className="w-32 text-sm font-medium text-gray-900 text-right flex-shrink-0">
                      {region}
                    </div>
                    <div className="flex-1 flex items-center">
                      <div
                        className={`h-6 rounded ${isPositive ? "bg-blue-400" : "bg-red-300"}`}
                        style={{ width: `${Math.max(barWidth, 2)}%` }}
                      />
                      <span className="ml-2 text-sm tabular-nums text-gray-600">
                        {data.mean_composite.toFixed(3)} ({data.n_states} states)
                      </span>
                    </div>
                    {data.trend && (
                      <span className={`text-xs px-2 py-0.5 rounded-full ${
                        data.trend === "rising" ? "bg-emerald-100 text-emerald-700" :
                        data.trend === "declining" ? "bg-red-100 text-red-700" :
                        "bg-gray-100 text-gray-600"
                      }`}>
                        {data.trend}
                      </span>
                    )}
                  </div>
                );
              })}
          </div>
        </section>
      )}

      {/* Interactive charts */}
      <InsightsClient insights={insights} />

      {/* Footer links */}
      <div className="mt-8 flex justify-center gap-6">
        <Link href="/electricity" className="text-blue-600 hover:underline text-sm font-medium">
          Electricity deep-dive &rarr;
        </Link>
        <Link href="/methodology" className="text-blue-600 hover:underline text-sm font-medium">
          Full methodology &rarr;
        </Link>
      </div>
    </div>
  );
}
