import type { Metadata } from "next";
import Breadcrumbs from "@/components/common/Breadcrumbs";
import PipelineDiagram from "@/components/methodology/PipelineDiagram";
import TableOfContents from "@/components/methodology/TableOfContents";

export const metadata: Metadata = {
  title: "Methodology | State Economic Activity Index",
  description:
    "How the Li Keqiang Index for Indian States is calculated: data sources, normalization, weighting, and known limitations.",
};

function Section({
  id,
  title,
  children,
}: {
  id: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="scroll-mt-20 mb-12">
      <h2 className="text-2xl font-bold text-gray-900 mb-4">{title}</h2>
      {children}
    </section>
  );
}

export default function MethodologyPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <Breadcrumbs
        items={[
          { label: "Home", href: "/" },
          { label: "Methodology" },
        ]}
      />

      <h1 className="text-3xl font-bold text-gray-900 mb-2">Methodology</h1>
      <p className="text-gray-600 mb-8 max-w-3xl">
        A transparent, step-by-step explanation of how the State Economic
        Activity Index is constructed, from raw government data to composite
        rankings.
      </p>

      <div className="lg:grid lg:grid-cols-[1fr_220px] lg:gap-8">
        {/* Main content */}
        <div>
          {/* Section 1: Origin */}
          <Section id="origin" title="1. The Li Keqiang Index: Origin & Adaptation">
            <div className="prose prose-gray max-w-none space-y-4 text-gray-700">
              <p>
                In 2007, Li Keqiang (then Communist Party Secretary of Liaoning
                Province, later Premier of China) told a US diplomat that China's
                official GDP figures were &ldquo;man-made&rdquo; and unreliable.
                Instead, he tracked three indicators he considered harder to fake:
                <strong> electricity consumption</strong>,{" "}
                <strong>rail freight volume</strong>, and{" "}
                <strong>bank lending</strong>.
              </p>
              <p>
                This project adapts the same philosophy for Indian states. Rather
                than relying on a single official GDP figure, we combine four
                independently-collected indicators that each capture a different
                dimension of economic activity:
              </p>
              <ul className="list-disc pl-6 space-y-1">
                <li>
                  <strong>GST Collections</strong> &mdash; economic transactions
                </li>
                <li>
                  <strong>Electricity Demand</strong> &mdash; physical economic
                  activity
                </li>
                <li>
                  <strong>Bank Credit Growth</strong> &mdash; financial depth and
                  investment
                </li>
                <li>
                  <strong>EPFO Net Payroll</strong> &mdash; formal employment
                  creation
                </li>
              </ul>
              <p>
                None of these indicators is perfect in isolation. GST misses the
                informal economy, electricity data lacks island territories, bank
                credit is inflated by Mumbai&apos;s financial hub role, and EPFO only
                covers the formal sector. But together, they triangulate to a
                more honest picture of state-level economic activity than any
                single statistic.
              </p>
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-sm">
                <strong>Important:</strong> This is a descriptive composite
                indicator, not a predictive model. It tracks patterns in
                observable economic activity — it does not predict GDP or claim
                to measure the &ldquo;true&rdquo; size of the economy.
              </div>
            </div>
          </Section>

          {/* Section 2: Components */}
          <Section id="components" title="2. The Four Components">
            <div className="space-y-6">
              {/* GST */}
              <div className="bg-white border border-gray-200 rounded-lg p-5">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  GST Collections
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm mb-3">
                  <div>
                    <span className="text-gray-500">Type:</span>{" "}
                    <span className="font-medium">Flow (monthly)</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Source:</span>{" "}
                    <span className="font-medium">gst.gov.in</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Coverage:</span>{" "}
                    <span className="font-medium">36 states, 2017+</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Unit:</span>{" "}
                    <span className="font-medium">Rs Crore</span>
                  </div>
                </div>
                <p className="text-sm text-gray-700">
                  GST is the most direct proxy for economic transactions. When
                  businesses sell goods and services, they pay GST — higher
                  collections mean more commerce. This is a flow measure
                  (collected monthly), making it directly comparable across time
                  periods. GST was introduced in July 2017, so FY 2017-18 is a
                  partial year.
                </p>
              </div>

              {/* Electricity */}
              <div className="bg-white border border-gray-200 rounded-lg p-5">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Electricity Demand
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm mb-3">
                  <div>
                    <span className="text-gray-500">Type:</span>{" "}
                    <span className="font-medium">Flow (daily)</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Source:</span>{" "}
                    <span className="font-medium">POSOCO/Grid India</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Coverage:</span>{" "}
                    <span className="font-medium">33 states, 2013+</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Unit:</span>{" "}
                    <span className="font-medium">Million Units (MU)</span>
                  </div>
                </div>
                <p className="text-sm text-gray-700">
                  One of the original Li Keqiang indicators. Factories running,
                  shops open, construction sites operating — all consume
                  electricity. Unlike self-reported figures, electricity demand
                  is hard to fake because the grid tracks exactly how much power
                  each state draws. Three island territories (Andaman & Nicobar,
                  Ladakh, Lakshadweep) are missing due to separate mini-grids.
                </p>
                <p className="text-xs text-gray-500 mt-2">
                  Data via Robbie Andrew (CC-BY-4.0). This is consumption/demand
                  data, not generation.
                </p>
              </div>

              {/* Bank Credit */}
              <div className="bg-white border border-gray-200 rounded-lg p-5">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Bank Credit (Year-over-Year Growth)
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm mb-3">
                  <div>
                    <span className="text-gray-500">Type:</span>{" "}
                    <span className="font-medium">Stock to Flow (annual)</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Source:</span>{" "}
                    <span className="font-medium">RBI Handbook T.156</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Coverage:</span>{" "}
                    <span className="font-medium">36 states, 2004+</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Unit:</span>{" "}
                    <span className="font-medium">Rs Crore (YoY delta)</span>
                  </div>
                </div>
                <p className="text-sm text-gray-700">
                  Bank credit outstanding measures how much the banking system
                  has lent to a state&apos;s economy. We convert the stock (total
                  outstanding) to a flow by computing year-over-year differences,
                  giving us &ldquo;net new credit extended&rdquo; per year. This captures
                  financial depth — more lending means more investment, business
                  expansion, and housing activity.
                </p>
                <p className="text-xs text-amber-600 mt-2">
                  Known bias: Maharashtra&apos;s numbers are inflated by Mumbai&apos;s role
                  as a financial hub — corporate HQs book credit in Mumbai even
                  if spending happens elsewhere.
                </p>
              </div>

              {/* EPFO */}
              <div className="bg-white border border-gray-200 rounded-lg p-5">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  EPFO Net Payroll (Formal Employment)
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm mb-3">
                  <div>
                    <span className="text-gray-500">Type:</span>{" "}
                    <span className="font-medium">Flow (annual)</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Source:</span>{" "}
                    <span className="font-medium">epfindia.gov.in</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Coverage:</span>{" "}
                    <span className="font-medium">32 states, 2017+</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Unit:</span>{" "}
                    <span className="font-medium">Number of workers</span>
                  </div>
                </div>
                <p className="text-sm text-gray-700">
                  EPFO net payroll measures how many workers are entering the
                  formal economy in each state each year. This captures
                  employment formalization — a dimension the other three
                  indicators miss. A state can have high GST (consumption) but
                  low EPFO additions (informal labor). However, EPFO covers only
                  the organized/semi-organized sector (~15% of India&apos;s
                  workforce), so it skews toward states with more formal
                  employment.
                </p>
              </div>
            </div>
          </Section>

          {/* Section 3: Calculation */}
          <Section id="calculation" title="3. How the Index is Calculated">
            <div className="space-y-4 text-gray-700">
              <p>
                The index is built through a 7-step pipeline. Each step reads
                from the previous step&apos;s output, ensuring reproducibility and
                transparency.
              </p>

              <PipelineDiagram />

              <div className="bg-white border border-gray-200 rounded-lg p-5">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">
                  Worked Example: Gujarat, FY 2024-25
                </h3>
                <p className="text-sm text-gray-600 mb-3">
                  Here&apos;s how Gujarat&apos;s composite score and rank are computed for a
                  single fiscal year:
                </p>
                <div className="overflow-x-auto">
                  <table className="text-sm w-full">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-2 pr-4 font-medium text-gray-700">
                          Step
                        </th>
                        <th className="text-left py-2 pr-4 font-medium text-gray-700">
                          GST
                        </th>
                        <th className="text-left py-2 pr-4 font-medium text-gray-700">
                          Electricity
                        </th>
                        <th className="text-left py-2 pr-4 font-medium text-gray-700">
                          Credit
                        </th>
                        <th className="text-left py-2 font-medium text-gray-700">
                          EPFO
                        </th>
                      </tr>
                    </thead>
                    <tbody className="text-gray-600">
                      <tr className="border-b border-gray-100">
                        <td className="py-2 pr-4 font-medium">Raw value</td>
                        <td className="py-2 pr-4 tabular-nums">Rs 2.25L Cr</td>
                        <td className="py-2 pr-4 tabular-nums">146,600 MU</td>
                        <td className="py-2 pr-4 tabular-nums">Rs 1.05L Cr YoY</td>
                        <td className="py-2 tabular-nums">1.18M workers</td>
                      </tr>
                      <tr className="border-b border-gray-100">
                        <td className="py-2 pr-4 font-medium">Z-score</td>
                        <td className="py-2 pr-4 tabular-nums">+1.29</td>
                        <td className="py-2 pr-4 tabular-nums">+1.90</td>
                        <td className="py-2 pr-4 tabular-nums">+0.92</td>
                        <td className="py-2 tabular-nums">+1.18</td>
                      </tr>
                      <tr>
                        <td className="py-2 pr-4 font-medium">Weight</td>
                        <td className="py-2 pr-4">0.25</td>
                        <td className="py-2 pr-4">0.25</td>
                        <td className="py-2 pr-4">0.25</td>
                        <td className="py-2">0.25</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <div className="mt-3 p-3 bg-blue-50 rounded text-sm">
                  <strong>Composite score</strong> = (1.29 + 1.90 + 0.92 +
                  1.18) / 4 ={" "}
                  <strong className="text-blue-700">1.32</strong>
                  <br />
                  <strong>Rank</strong>: #3 out of 34 states
                </div>
              </div>
            </div>
          </Section>

          {/* Section 4: Z-Score */}
          <Section id="zscore" title="4. Z-Score Normalization Explained">
            <div className="space-y-4 text-gray-700">
              <p>
                Raw values aren&apos;t comparable across components — GST is in
                crores of rupees, electricity in million units, credit in crores
                of YoY delta, EPFO in number of people. A z-score puts all four
                on the same scale.
              </p>

              <div className="bg-gray-50 border border-gray-200 rounded-lg p-5">
                <h4 className="font-semibold text-gray-900 mb-2">
                  What is a z-score?
                </h4>
                <p className="text-sm">
                  A z-score tells you how many standard deviations a state is
                  from the average. A score of <strong>+1.0</strong> means the
                  state is one standard deviation above the national average for
                  that indicator. A score of <strong>-1.0</strong> means one
                  standard deviation below.
                </p>
                <div className="mt-3 font-mono text-sm bg-white border rounded p-3">
                  z = (state_value - mean_of_all_states) / std_dev_of_all_states
                </div>
              </div>

              <p>
                We compute <strong>cross-sectional</strong> z-scores — comparing
                states to each other within the same fiscal year, not to
                themselves across years. This makes the index a{" "}
                <em>relative ranking tool</em>: &ldquo;How does Gujarat compare to
                other states in FY 2024-25?&rdquo; rather than &ldquo;How does Gujarat
                compare to itself in FY 2020-21?&rdquo;
              </p>

              <p>
                With 32-36 states per year, the z-scores are statistically
                stable. The cross-sectional approach prevents older periods with
                lower absolute values from dominating the index.
              </p>
            </div>
          </Section>

          {/* Section 5: Data Sources */}
          <Section id="sources" title="5. Data Sources & Collection">
            <div className="overflow-x-auto">
              <table className="w-full text-sm border border-gray-200 rounded-lg overflow-hidden">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="text-left px-4 py-3 font-medium text-gray-700">
                      Component
                    </th>
                    <th className="text-left px-4 py-3 font-medium text-gray-700">
                      Source
                    </th>
                    <th className="text-left px-4 py-3 font-medium text-gray-700">
                      Frequency
                    </th>
                    <th className="text-left px-4 py-3 font-medium text-gray-700">
                      Coverage
                    </th>
                    <th className="text-left px-4 py-3 font-medium text-gray-700">
                      License
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  <tr>
                    <td className="px-4 py-3 font-medium">GST</td>
                    <td className="px-4 py-3">gst.gov.in (Excel)</td>
                    <td className="px-4 py-3">Monthly</td>
                    <td className="px-4 py-3">36 states, FY 2017-18+</td>
                    <td className="px-4 py-3 text-gray-500">Govt open data</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-3 font-medium">Electricity</td>
                    <td className="px-4 py-3">
                      POSOCO/Grid India via Robbie Andrew
                    </td>
                    <td className="px-4 py-3">Daily</td>
                    <td className="px-4 py-3">33 states, 2013+</td>
                    <td className="px-4 py-3 text-blue-600">CC-BY-4.0</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-3 font-medium">Bank Credit</td>
                    <td className="px-4 py-3">RBI Handbook Table 156</td>
                    <td className="px-4 py-3">Annual</td>
                    <td className="px-4 py-3">36 states, 2004+</td>
                    <td className="px-4 py-3 text-gray-500">Govt open data</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-3 font-medium">EPFO Payroll</td>
                    <td className="px-4 py-3">epfindia.gov.in (Excel)</td>
                    <td className="px-4 py-3">Annual</td>
                    <td className="px-4 py-3">32 states, FY 2017-18+</td>
                    <td className="px-4 py-3 text-gray-500">Govt open data</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="text-sm text-gray-600 mt-3">
              All data is sourced directly from government publications. The
              composite index uses annual granularity (matching the lowest
              common denominator — RBI and EPFO are annual). Monthly trend
              charts show GST + electricity only, clearly labeled as partial
              views.
            </p>
          </Section>

          {/* Section 6: Limitations */}
          <Section id="limitations" title="6. Known Limitations">
            <div className="space-y-3">
              {[
                {
                  title: "GST starts mid-2017",
                  detail:
                    "GST was introduced July 2017. No state-level GST data exists before that. FY 2017-18 is a partial year (9 months).",
                },
                {
                  title: "3 island territories missing electricity data",
                  detail:
                    "Andaman & Nicobar, Ladakh, and Lakshadweep have separate mini-grids not captured in the national grid data. Their composite index uses 3 of 4 components.",
                },
                {
                  title: "Mumbai credit inflation",
                  detail:
                    "Maharashtra's bank credit numbers are inflated by corporate headquarters booking credit in Mumbai, even when the economic activity occurs elsewhere.",
                },
                {
                  title: "EPFO covers only the formal sector",
                  detail:
                    "EPFO net payroll captures roughly 15% of India's workforce. It systematically underrepresents states with larger informal economies.",
                },
                {
                  title: "Equal weights are an assumption",
                  detail:
                    "v1 uses equal weights (0.25 each) because we have no prior reason to weight one component more than another. PCA-derived weights are a future enhancement.",
                },
                {
                  title: "Annual granularity for composite",
                  detail:
                    "The composite index is annual because RBI and EPFO publish annually. Monthly indices are not possible without interpolating (fabricating) data.",
                },
                {
                  title: "State reorganization discontinuities",
                  detail:
                    "Telangana (separated from Andhra Pradesh in 2014) and Ladakh (from J&K in 2019) create data discontinuities. Earlier data is attributed to the pre-split entity.",
                },
              ].map((item) => (
                <div
                  key={item.title}
                  className="bg-white border border-gray-200 rounded-lg p-4"
                >
                  <h4 className="font-medium text-gray-900 text-sm">
                    {item.title}
                  </h4>
                  <p className="text-sm text-gray-600 mt-1">{item.detail}</p>
                </div>
              ))}
            </div>
          </Section>

          {/* Section 7: Validation */}
          <Section id="validation" title="7. Validation & Spot-Checks">
            <div className="space-y-4 text-gray-700">
              <p>
                To validate the index, we checked whether results match
                known economic realities:
              </p>

              <div className="space-y-3">
                <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-4">
                  <h4 className="font-medium text-emerald-900 text-sm">
                    Maharashtra is consistently #1
                  </h4>
                  <p className="text-sm text-emerald-800 mt-1">
                    Maharashtra ranks #1 in every fiscal year from 2017-18 to
                    2024-25. This matches its known position as India&apos;s largest
                    state economy, top GST contributor, and financial capital.
                    Composite score: 4.01 (FY 2024-25).
                  </p>
                </div>

                <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-4">
                  <h4 className="font-medium text-emerald-900 text-sm">
                    COVID dip is clearly visible
                  </h4>
                  <p className="text-sm text-emerald-800 mt-1">
                    FY 2020-21 shows a visible dip across most states in all
                    components, with strong recovery in 2021-22. The monthly
                    GST + electricity charts show the sharp April 2020 drop from
                    the national lockdown.
                  </p>
                </div>

                <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-4">
                  <h4 className="font-medium text-emerald-900 text-sm">
                    Tamil Nadu leads EPFO
                  </h4>
                  <p className="text-sm text-emerald-800 mt-1">
                    Tamil Nadu has the highest EPFO z-score, matching its known
                    strength in formal manufacturing employment (auto,
                    electronics, textiles). 2.87 million net new enrollments in
                    FY 2024-25.
                  </p>
                </div>

                <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-4">
                  <h4 className="font-medium text-emerald-900 text-sm">
                    Top 5 matches economic intuition
                  </h4>
                  <p className="text-sm text-emerald-800 mt-1">
                    The top 5 states (Maharashtra, Tamil Nadu, Gujarat,
                    Karnataka, Uttar Pradesh) are all recognized as major
                    economic powerhouses. No surprising entries, no missing
                    obvious ones.
                  </p>
                </div>
              </div>

              <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-sm">
                <strong>Version:</strong> v1 (Equal Weights) &mdash;{" "}
                <strong>34 states</strong> ranked &mdash;{" "}
                <strong>FY 2017-18 to 2024-25</strong> &mdash; Pipeline runs in
                ~3.5 seconds
              </div>
            </div>
          </Section>
          {/* Section 8: Analytical Metrics */}
          <Section id="analytical-metrics" title="8. Analytical Metrics">
            <div className="space-y-4 text-gray-700">
              <p>
                Beyond static rankings, the index computes several derived metrics
                to surface dynamics and trends.
              </p>

              <div className="space-y-3">
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-900 text-sm mb-1">
                    Growth Rates (YoY %)
                  </h4>
                  <p className="text-sm text-gray-600">
                    Year-over-year percentage change in each component&apos;s{" "}
                    <strong>raw value</strong> (not z-scores). Z-scores are
                    cross-sectional and change when peers change &mdash; raw values
                    reflect actual growth in that state&apos;s economy. Formula:{" "}
                    <code className="text-xs bg-gray-100 px-1 rounded">
                      (value[t] / value[t-1]) - 1
                    </code>
                  </p>
                </div>

                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-900 text-sm mb-1">
                    Rank Momentum (3-Year)
                  </h4>
                  <p className="text-sm text-gray-600">
                    How many rank positions a state has moved over 3 fiscal years:{" "}
                    <code className="text-xs bg-gray-100 px-1 rounded">
                      rank[t-3] - rank[t]
                    </code>
                    . Positive = improved. Thresholds: &ge;+3 = &ldquo;rising&rdquo;,
                    &le;-3 = &ldquo;declining&rdquo;, within &plusmn;2 = &ldquo;stable&rdquo;.
                    A &plusmn;1 or &plusmn;2 shift is within noise for 34 states.
                  </p>
                </div>

                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-900 text-sm mb-1">
                    COVID Recovery Speed
                  </h4>
                  <p className="text-sm text-gray-600">
                    The COVID dip is measured as composite_score[FY 2020-21] minus
                    composite_score[FY 2019-20]. Recovery speed is the number of
                    fiscal years after 2020-21 until composite_score returns to the
                    pre-COVID level. States that never recovered are flagged as
                    &ldquo;Not yet recovered&rdquo;. States already declining before COVID
                    are noted separately.
                  </p>
                </div>

                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-900 text-sm mb-1">
                    Component Diagnostics
                  </h4>
                  <p className="text-sm text-gray-600">
                    For each state, the strongest and weakest z-score components are
                    identified. The divergence score (max z minus min z) flags states
                    with unbalanced economic profiles. A state with high GST but low
                    EPFO may have a large informal workforce.
                  </p>
                </div>
              </div>
            </div>
          </Section>

          {/* Section 9: Statistical Validation */}
          <Section id="statistical-validation" title="9. Statistical Validation">
            <div className="space-y-4 text-gray-700">
              <p>
                To validate whether our four indicators have explanatory power for
                official economic output, we run cross-sectional OLS regressions of
                state GSDP on the four activity indicators. This is a validation
                exercise, <strong>not a predictive model</strong>.
              </p>

              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 text-sm mb-2">
                  Cross-Sectional OLS
                </h4>
                <p className="text-sm text-gray-600">
                  For each fiscal year with both GSDP and index data:
                </p>
                <div className="mt-2 font-mono text-xs bg-gray-50 border rounded p-3">
                  GSDP ~ beta_1*GST + beta_2*Electricity + beta_3*Credit +
                  beta_4*EPFO + constant
                </div>
                <p className="text-sm text-gray-600 mt-2">
                  Using <code className="text-xs bg-gray-100 px-1 rounded">statsmodels.OLS</code>{" "}
                  with full diagnostics. Standardized coefficients (beta weights)
                  allow direct comparison of each component&apos;s marginal association.
                </p>
              </div>

              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 text-sm mb-2">
                  Stepwise Model Comparison
                </h4>
                <p className="text-sm text-gray-600">
                  Models built incrementally (GST only &rarr; +Electricity &rarr;
                  +Credit &rarr; +EPFO) with partial F-tests at each step. This tests
                  whether each additional component adds significant explanatory power
                  beyond what&apos;s already captured.
                </p>
              </div>

              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 text-sm mb-2">
                  Diagnostic Tests
                </h4>
                <div className="overflow-x-auto">
                  <table className="text-sm w-full mt-2">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-2 pr-4 font-medium text-gray-700">
                          Test
                        </th>
                        <th className="text-left py-2 pr-4 font-medium text-gray-700">
                          Purpose
                        </th>
                        <th className="text-left py-2 font-medium text-gray-700">
                          Concern If Fails
                        </th>
                      </tr>
                    </thead>
                    <tbody className="text-gray-600 divide-y divide-gray-100">
                      <tr>
                        <td className="py-2 pr-4 font-medium">VIF</td>
                        <td className="py-2 pr-4">Multicollinearity</td>
                        <td className="py-2">
                          VIF &gt; 10 means components are too correlated; individual
                          coefficients unreliable
                        </td>
                      </tr>
                      <tr>
                        <td className="py-2 pr-4 font-medium">Breusch-Pagan</td>
                        <td className="py-2 pr-4">Heteroscedasticity</td>
                        <td className="py-2">
                          If significant, use robust (HC3) standard errors
                        </td>
                      </tr>
                      <tr>
                        <td className="py-2 pr-4 font-medium">Shapiro-Wilk</td>
                        <td className="py-2 pr-4">Normality of residuals</td>
                        <td className="py-2">
                          If non-normal, t-tests and F-tests are approximate
                        </td>
                      </tr>
                      <tr>
                        <td className="py-2 pr-4 font-medium">Cook&apos;s Distance</td>
                        <td className="py-2 pr-4">Influential observations</td>
                        <td className="py-2">
                          Flag states with D &gt; 4/N (Maharashtra expected)
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-sm">
                <strong>Critical caveats:</strong> GST is mechanically related to
                GDP (it is a tax on economic transactions that also constitute GDP).
                A high R&sup2; is partly tautological. Results are sensitive to
                Maharashtra&apos;s inclusion. With N~34, degrees of freedom are limited.
                These are cross-sectional associations &mdash; not causal
                relationships. See the{" "}
                <a href="/insights" className="text-blue-600 hover:underline">
                  Insights page
                </a>{" "}
                for full regression results.
              </div>
            </div>
          </Section>

          {/* Section 10: Relationship with GSDP */}
          <Section id="gsdp-relationship" title="10. Relationship with Official GSDP">
            <div className="space-y-4 text-gray-700">
              <p>
                We compare our index rankings to official GSDP rankings (RBI
                Handbook Tables 21 &amp; 22) to understand where and why they
                diverge.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-900 text-sm mb-1">
                    Agriculture gap
                  </h4>
                  <p className="text-sm text-gray-600">
                    None of our 4 components capture agricultural output. Farm-heavy
                    states like Punjab and Madhya Pradesh may rank lower on our index
                    than on GSDP.
                  </p>
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-900 text-sm mb-1">
                    Formalization signal
                  </h4>
                  <p className="text-sm text-gray-600">
                    EPFO payroll captures formal employment that GSDP may not fully
                    reflect. States with rapid formalization may outperform their GDP
                    rank on our index.
                  </p>
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h4 className="font-semibold text-gray-900 text-sm mb-1">
                    Financial hub bias
                  </h4>
                  <p className="text-sm text-gray-600">
                    Maharashtra&apos;s bank credit is inflated by Mumbai&apos;s role as a
                    financial capital. Corporate HQs book credit there even when
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
                    may capture momentum that GSDP hasn&apos;t reflected yet.
                  </p>
                </div>
              </div>

              <p className="text-sm">
                Pearson correlations between each component and GSDP, plus the
                full scatter plot comparison, are available on the{" "}
                <a href="/insights" className="text-blue-600 hover:underline">
                  Insights page
                </a>
                .
              </p>
            </div>
          </Section>

          {/* Section 11: Additional Caveats */}
          <Section id="updated-limitations" title="11. Additional Caveats">
            <div className="space-y-3">
              {[
                {
                  title: "GSDP data lags 1-2 years",
                  detail:
                    "RBI publishes GSDP with a 1-2 year lag. Our cross-sectional regressions use the latest available GSDP year, which may be 1-2 years behind our most recent index year.",
                },
                {
                  title: "Correlation is not causation",
                  detail:
                    "A high R-squared between our index and GSDP validates that the chosen indicators covary with official output — it does not establish that one causes the other.",
                },
                {
                  title: "Mechanical GST-GDP relationship",
                  detail:
                    "GST is a tax on the same economic transactions that constitute GDP. The GST coefficient in regression partially reflects this mechanical, not causal, relationship.",
                },
                {
                  title: "Small sample size (N~34)",
                  detail:
                    "With ~34 states and 4 regressors plus a constant, we have ~29 degrees of freedom. Marginal but usable. Pooled panel regression across years provides more observations.",
                },
                {
                  title: "Maharashtra influence",
                  detail:
                    "Maharashtra is an extreme outlier (composite score 4.01 when the next is ~1.4). Regression results with and without Maharashtra are both reported for transparency.",
                },
                {
                  title: "BRAP methodology changes",
                  detail:
                    "BRAP (Business Reform Action Plan) scoring methodology changed across editions (2015-2022). We show only the latest edition (2020) as a categorical label, not a time series.",
                },
              ].map((item) => (
                <div
                  key={item.title}
                  className="bg-white border border-gray-200 rounded-lg p-4"
                >
                  <h4 className="font-medium text-gray-900 text-sm">
                    {item.title}
                  </h4>
                  <p className="text-sm text-gray-600 mt-1">{item.detail}</p>
                </div>
              ))}
            </div>
          </Section>
        </div>

        {/* Sidebar TOC — desktop only */}
        <aside className="hidden lg:block">
          <div className="sticky top-20">
            <TableOfContents />
          </div>
        </aside>
      </div>
    </div>
  );
}
