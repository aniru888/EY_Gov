import type { Metadata } from "next";
import Link from "next/link";
import { getInsights, getRegressionData } from "@/lib/data";
import Breadcrumbs from "@/components/common/Breadcrumbs";
import InsightsClient from "@/components/InsightsClient";

export const metadata: Metadata = {
  title: "Insights & Analysis | State Economic Activity Index",
  description:
    "Key findings, GSDP comparison, growth dynamics, COVID recovery, and statistical validation of the Li Keqiang Index for Indian States.",
};

const FINDING_COLORS: Record<string, string> = {
  fastest_rising: "border-emerald-400 bg-emerald-50",
  outperformer: "border-blue-400 bg-blue-50",
  underperformer: "border-amber-400 bg-amber-50",
  fastest_recovery: "border-teal-400 bg-teal-50",
  most_unbalanced: "border-purple-400 bg-purple-50",
  strongest_correlation: "border-indigo-400 bg-indigo-50",
};

export default function InsightsPage() {
  const insights = getInsights();
  const regression = getRegressionData();

  const latestFy = insights.latest_fy;
  const cs = regression.cross_sectional?.[regression.latest_fy_with_gsdp || ""];

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <Breadcrumbs
        items={[{ label: "Home", href: "/" }, { label: "Insights" }]}
      />

      <h1 className="text-3xl font-bold text-gray-900 mb-2">
        Insights & Analysis
      </h1>
      <p className="text-gray-600 mb-8 max-w-3xl">
        What our four economic indicators reveal about state-level activity,
        and how they relate to official GDP. FY {latestFy} data.
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

      {/* Section 2: Statistical Validation */}
      {cs && !regression.skipped && (
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Statistical Validation
          </h2>

          {/* Headline stat */}
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
                      <th className="text-left px-4 py-2 font-medium text-gray-700">
                        Model
                      </th>
                      <th className="text-right px-4 py-2 font-medium text-gray-700">
                        R&sup2;
                      </th>
                      <th className="text-right px-4 py-2 font-medium text-gray-700">
                        &Delta;R&sup2;
                      </th>
                      <th className="text-right px-4 py-2 font-medium text-gray-700">
                        Partial F
                      </th>
                      <th className="text-center px-4 py-2 font-medium text-gray-700">
                        Significant?
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {Object.entries(regression.stepwise).map(
                      ([key, step]) => (
                        <tr key={key}>
                          <td className="px-4 py-2 font-medium text-gray-900">
                            {key
                              .replace("model_1_gst_only", "1. GST only")
                              .replace("model_2_gst_elec", "2. + Electricity")
                              .replace(
                                "model_3_gst_elec_credit",
                                "3. + Bank Credit"
                              )
                              .replace("model_4_full", "4. + EPFO (full)")}
                          </td>
                          <td className="px-4 py-2 text-right tabular-nums">
                            {step.r_squared.toFixed(3)}
                          </td>
                          <td className="px-4 py-2 text-right tabular-nums text-gray-500">
                            {step.delta_r2 != null
                              ? `+${step.delta_r2.toFixed(3)}`
                              : "--"}
                          </td>
                          <td className="px-4 py-2 text-right tabular-nums text-gray-500">
                            {step.partial_f != null
                              ? step.partial_f.toFixed(1)
                              : "--"}
                          </td>
                          <td className="px-4 py-2 text-center">
                            {step.partial_f_p != null ? (
                              step.partial_f_p < 0.05 ? (
                                <span className="text-emerald-600 font-medium">
                                  Yes (p={step.partial_f_p < 0.001
                                    ? "<0.001"
                                    : step.partial_f_p.toFixed(3)})
                                </span>
                              ) : (
                                <span className="text-gray-400">
                                  No (p={step.partial_f_p.toFixed(3)})
                                </span>
                              )
                            ) : (
                              <span className="text-gray-400">Baseline</span>
                            )}
                          </td>
                        </tr>
                      )
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Coefficient table */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">
              Standardized Coefficients (Beta Weights)
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm border border-gray-200 rounded-lg">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="text-left px-4 py-2 font-medium text-gray-700">
                      Variable
                    </th>
                    <th className="text-right px-4 py-2 font-medium text-gray-700">
                      Beta
                    </th>
                    <th className="text-right px-4 py-2 font-medium text-gray-700">
                      p-value
                    </th>
                    <th className="text-right px-4 py-2 font-medium text-gray-700">
                      VIF
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {["gst_total", "electricity_mu", "bank_credit_yoy", "epfo_payroll"].map(
                    (v) => {
                      const c = cs.coefficients[v];
                      const vif =
                        typeof cs.diagnostics.vif === "object"
                          ? cs.diagnostics.vif[v]
                          : null;
                      return (
                        <tr key={v}>
                          <td className="px-4 py-2 font-medium text-gray-900">
                            {v
                              .replace("gst_total", "GST Collections")
                              .replace("electricity_mu", "Electricity Demand")
                              .replace("bank_credit_yoy", "Bank Credit (YoY)")
                              .replace("epfo_payroll", "EPFO Payroll")}
                          </td>
                          <td className="px-4 py-2 text-right tabular-nums font-medium">
                            {c?.beta != null ? c.beta.toFixed(3) : "--"}
                          </td>
                          <td className="px-4 py-2 text-right tabular-nums">
                            <span
                              className={
                                c && c.p < 0.05
                                  ? "text-emerald-600 font-medium"
                                  : "text-gray-400"
                              }
                            >
                              {c ? (c.p < 0.001 ? "<0.001" : c.p.toFixed(3)) : "--"}
                              {c && c.p < 0.001
                                ? " ***"
                                : c && c.p < 0.01
                                  ? " **"
                                  : c && c.p < 0.05
                                    ? " *"
                                    : ""}
                            </span>
                          </td>
                          <td className="px-4 py-2 text-right tabular-nums">
                            <span
                              className={
                                vif && vif > 10
                                  ? "text-red-600 font-medium"
                                  : "text-gray-500"
                              }
                            >
                              {vif != null ? vif.toFixed(1) : "--"}
                            </span>
                          </td>
                        </tr>
                      );
                    }
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Maharashtra sensitivity */}
          {cs.without_maharashtra && (
            <div className="mb-6 bg-amber-50 border border-amber-200 rounded-lg p-4">
              <h4 className="font-semibold text-amber-900 text-sm mb-1">
                Maharashtra Sensitivity Check
              </h4>
              <p className="text-sm text-amber-800">
                With Maharashtra: R&sup2; = {cs.r_squared.toFixed(3)} (N={cs.n}).
                Without Maharashtra: R&sup2; ={" "}
                {cs.without_maharashtra.r_squared.toFixed(3)} (N=
                {cs.without_maharashtra.n}).
                {cs.without_maharashtra.note && (
                  <span> {cs.without_maharashtra.note}</span>
                )}
              </p>
            </div>
          )}

          {/* VIF warning */}
          {cs.diagnostics.vif_warning && (
            <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
              <h4 className="font-semibold text-red-900 text-sm mb-1">
                Multicollinearity Warning
              </h4>
              <p className="text-sm text-red-800">
                {cs.diagnostics.vif_warning} Joint significance (F-test) remains
                valid, but individual coefficients should be interpreted with
                caution.
              </p>
            </div>
          )}

          {/* Caveat box */}
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-sm text-gray-600">
            <strong>Important:</strong> These are cross-sectional associations.
            GST is mechanically related to GDP (it is a tax on economic
            transactions). This analysis validates that the chosen indicators
            covary with official output measures &mdash; it does not establish
            causation.
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
                    ? entry.p < 0.001
                      ? "p<0.001"
                      : `p=${entry.p.toFixed(3)}`
                    : ""}
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-500">
            Correlation does not imply causation. These associations are
            cross-sectional (across states in the same fiscal year).
          </p>
        </section>
      )}

      {/* Section 4: Understanding the Gap */}
      <section className="mb-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Understanding the Gap: Index vs Official GDP
        </h2>
        <div className="prose prose-gray max-w-none text-gray-700 space-y-3 text-sm">
          <p>
            Why does our index sometimes diverge from official GSDP rankings?
            Several structural factors explain the difference:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 text-sm mb-1">
                Agriculture (GSDP captures, index misses)
              </h4>
              <p className="text-sm text-gray-600">
                None of our 4 components directly capture agricultural output.
                Farm-heavy states like Punjab and Madhya Pradesh may rank lower on
                our index than on GSDP.
              </p>
            </div>
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 text-sm mb-1">
                Formalization (index captures, GSDP may undercount)
              </h4>
              <p className="text-sm text-gray-600">
                EPFO payroll captures formal employment creation that GSDP
                estimates may not fully reflect. States with rapid formalization
                may outperform their GDP rank.
              </p>
            </div>
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 text-sm mb-1">
                Financial hub effect
              </h4>
              <p className="text-sm text-gray-600">
                Maharashtra&apos;s bank credit is inflated by Mumbai&apos;s role as a
                financial capital. Corporate HQs book credit in Mumbai even when
                spending occurs elsewhere.
              </p>
            </div>
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 text-sm mb-1">
                Time lag
              </h4>
              <p className="text-sm text-gray-600">
                GSDP estimates lag 1-2 years and are frequently revised. Our index
                uses more recent data (GST: monthly, electricity: daily), so it
                may capture recent momentum that GSDP hasn&apos;t reflected yet.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Sections 5 & 6: Interactive charts */}
      <InsightsClient insights={insights} />

      {/* Footer link */}
      <div className="mt-8 text-center">
        <Link
          href="/methodology"
          className="text-blue-600 hover:underline text-sm font-medium"
        >
          Read the full methodology &rarr;
        </Link>
      </div>
    </div>
  );
}
