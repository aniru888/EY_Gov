import type { Metadata } from "next";
import Link from "next/link";
import { getElectricityData, getEnhancedRegression } from "@/lib/data";
import Breadcrumbs from "@/components/common/Breadcrumbs";
import ElectricityClient from "@/components/ElectricityClient";

export const metadata: Metadata = {
  title: "Electricity Deep-Dive | State Economic Activity Index",
  description:
    "Electricity demand as a real-time economic proxy: intensity, elasticity, seasonality, and momentum analysis across Indian states.",
};

export default function ElectricityPage() {
  const data = getElectricityData();
  const regression = getEnhancedRegression();

  const ns = data.national_summary;
  const stepwise = regression.stepwise;

  // Extract the delta_r2 from adding electricity to GST
  const elecDeltaR2 = stepwise?.model_2_gst_elec?.delta_r2;

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <Breadcrumbs
        items={[{ label: "Home", href: "/" }, { label: "Electricity" }]}
      />

      {/* Hero */}
      <div className="mb-10">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Electricity Demand as a Real-Time Economic Proxy
        </h1>
        <p className="text-gray-600 max-w-3xl mb-6">
          Of all economic signals, electricity demand is the hardest to fake and
          the fastest to arrive. It correlates r={ns.gsdp_correlation_r?.toFixed(2)} with
          state GDP and reveals whether a state is industrial, services-driven,
          or transitioning.
        </p>

        {/* Metric cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-5">
            <div className="text-3xl font-bold text-blue-800 tabular-nums">
              {ns.total_mu_latest_fy
                ? `${(ns.total_mu_latest_fy / 1e6).toFixed(2)}M`
                : "--"}{" "}
              MU
            </div>
            <p className="text-blue-700 text-sm mt-1">
              National electricity demand ({ns.latest_fy})
            </p>
          </div>
          <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-5">
            <div className="text-3xl font-bold text-emerald-800 tabular-nums">
              r = {ns.gsdp_correlation_r?.toFixed(3) ?? "--"}
            </div>
            <p className="text-emerald-700 text-sm mt-1">
              Correlation with state GSDP (strongest single predictor)
            </p>
          </div>
          <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-5">
            <div className="text-3xl font-bold text-indigo-800 tabular-nums">
              +{elecDeltaR2 ? (elecDeltaR2 * 100).toFixed(1) : "--"}pp
            </div>
            <p className="text-indigo-700 text-sm mt-1">
              R&sup2; improvement when added to GST-only model
            </p>
          </div>
        </div>
      </div>

      {/* Why Electricity Works */}
      <section className="mb-10">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          Why Electricity Works
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <h3 className="font-semibold text-gray-900 text-sm mb-1">
              Hard to fake
            </h3>
            <p className="text-sm text-gray-600">
              Unlike GDP estimates subject to revision, grid electricity demand
              is physically measured by POSOCO/Grid India. You cannot
              overstate it.
            </p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <h3 className="font-semibold text-gray-900 text-sm mb-1">
              Available daily
            </h3>
            <p className="text-sm text-gray-600">
              Electricity data arrives daily vs GSDP which lags 1&ndash;2 years.
              It is the most timely macro signal available at the state level.
            </p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <h3 className="font-semibold text-gray-900 text-sm mb-1">
              Captures physical activity
            </h3>
            <p className="text-sm text-gray-600">
              Factory output, construction, commercial operations, cold chains
              &mdash; industrial and commercial activity requires power.
              Electricity is a direct proxy for physical economic output.
            </p>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <h3 className="font-semibold text-gray-900 text-sm mb-1">
              Independent signal
            </h3>
            <p className="text-sm text-gray-600">
              Low VIF (3.25) with other index components. Electricity adds
              genuine information beyond what GST, credit, and EPFO provide.
            </p>
          </div>
        </div>
      </section>

      {/* Interactive Charts */}
      <ElectricityClient data={data} />

      {/* What Electricity Misses */}
      <section className="mt-10 mb-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          What Electricity Misses
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
            <h3 className="font-semibold text-amber-900 text-sm mb-1">
              Services economy
            </h3>
            <p className="text-sm text-amber-800">
              IT exports, financial services, tourism &mdash; these are
              low-electricity-intensity sectors. States like Kerala and
              Karnataka&apos;s IT hubs are underrepresented.
            </p>
          </div>
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
            <h3 className="font-semibold text-amber-900 text-sm mb-1">
              Agriculture
            </h3>
            <p className="text-sm text-amber-800">
              Farm power is subsidized and often unmeasured. States with large
              agricultural economies (Punjab, MP) show lower electricity
              intensity relative to their GDP.
            </p>
          </div>
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
            <h3 className="font-semibold text-amber-900 text-sm mb-1">
              Informal sector
            </h3>
            <p className="text-sm text-amber-800">
              Household consumption and small enterprises use electricity but
              their economic output isn&apos;t captured in formal GDP
              or our index.
            </p>
          </div>
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
            <h3 className="font-semibold text-amber-900 text-sm mb-1">
              Renewable energy shifts
            </h3>
            <p className="text-sm text-amber-800">
              States with growing rooftop solar may show declining grid demand
              even as total energy use grows. This will become more relevant
              over time.
            </p>
          </div>
        </div>
      </section>

      {/* Footer links */}
      <div className="mt-8 flex justify-center gap-6">
        <Link
          href="/insights"
          className="text-blue-600 hover:underline text-sm font-medium"
        >
          Statistical analysis &rarr;
        </Link>
        <Link
          href="/methodology"
          className="text-blue-600 hover:underline text-sm font-medium"
        >
          Full methodology &rarr;
        </Link>
      </div>
    </div>
  );
}
