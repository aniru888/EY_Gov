import type { StateInsights } from "@/lib/types";

interface Props {
  state: string;
  insights: StateInsights;
  latestFy: string;
}

function MomentumBadge({ tier }: { tier: string | null }) {
  if (!tier) return null;
  const colors: Record<string, string> = {
    rising: "bg-emerald-100 text-emerald-700",
    declining: "bg-red-100 text-red-700",
    stable: "bg-gray-100 text-gray-600",
  };
  return (
    <span
      className={`inline-block text-xs font-medium px-2 py-0.5 rounded-full ${
        colors[tier] || colors.stable
      }`}
    >
      {tier}
    </span>
  );
}

function BrapBadge({ category }: { category: string | null }) {
  if (!category) return null;
  const colors: Record<string, string> = {
    "Top Achiever": "bg-indigo-100 text-indigo-700",
    Achiever: "bg-blue-100 text-blue-700",
    Aspirer: "bg-amber-100 text-amber-700",
    Emerging: "bg-gray-100 text-gray-600",
  };
  return (
    <span
      className={`inline-block text-xs font-medium px-2 py-0.5 rounded-full ${
        colors[category] || "bg-gray-100 text-gray-600"
      }`}
    >
      BRAP: {category}
    </span>
  );
}

function pctFmt(val: number | null): string {
  if (val === null || val === undefined) return "--";
  const sign = val >= 0 ? "+" : "";
  return `${sign}${val.toFixed(1)}%`;
}

export default function StateDiagnosticBanner({
  state,
  insights,
  latestFy,
}: Props) {
  const hasGsdpGap =
    insights.rank_gap !== null && Math.abs(insights.rank_gap) >= 3;

  return (
    <div className="space-y-4 mb-8">
      {/* Diagnostic text + badges */}
      {insights.diagnostic_text && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex flex-wrap items-center gap-2 mb-2">
            <MomentumBadge tier={insights.momentum_tier} />
            <BrapBadge category={insights.brap_category} />
          </div>
          <p className="text-sm text-blue-900">{insights.diagnostic_text}</p>
        </div>
      )}

      {/* GSDP gap callout */}
      {hasGsdpGap && (
        <div
          className={`border rounded-lg p-4 ${
            insights.rank_gap! > 0
              ? "border-emerald-200 bg-emerald-50"
              : "border-amber-200 bg-amber-50"
          }`}
        >
          <p
            className={`text-sm font-medium ${
              insights.rank_gap! > 0 ? "text-emerald-900" : "text-amber-900"
            }`}
          >
            Ranks #{insights.gsdp_rank! - insights.rank_gap!} on the activity
            index vs #{insights.gsdp_rank} on official GSDP
            <span className="font-normal">
              {" "}
              &mdash; {insights.gap_label}
              {insights.rank_gap! > 0
                ? " (index captures more activity than GDP reflects)"
                : " (GDP captures more activity than our index reflects)"}
            </span>
          </p>
        </div>
      )}

      {/* Growth metrics row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-white border border-gray-200 rounded-lg p-3">
          <div className="text-xs text-gray-500 font-medium">GST YoY</div>
          <div className="text-lg font-bold tabular-nums text-gray-900">
            {pctFmt(insights.gst_yoy_pct)}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3">
          <div className="text-xs text-gray-500 font-medium">Elec YoY</div>
          <div className="text-lg font-bold tabular-nums text-gray-900">
            {pctFmt(insights.elec_yoy_pct)}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3">
          <div className="text-xs text-gray-500 font-medium">3yr Momentum</div>
          <div className="text-lg font-bold tabular-nums text-gray-900">
            {insights.rank_momentum_3yr !== null ? (
              <span
                className={
                  insights.rank_momentum_3yr >= 3
                    ? "text-emerald-600"
                    : insights.rank_momentum_3yr <= -3
                      ? "text-red-600"
                      : ""
                }
              >
                {insights.rank_momentum_3yr > 0 ? "+" : ""}
                {insights.rank_momentum_3yr} positions
              </span>
            ) : (
              "--"
            )}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-3">
          <div className="text-xs text-gray-500 font-medium">EPFO YoY</div>
          <div className="text-lg font-bold tabular-nums text-gray-900">
            {pctFmt(insights.epfo_yoy_pct)}
          </div>
        </div>
      </div>
    </div>
  );
}
