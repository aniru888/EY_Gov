import Link from "next/link";
import type { PeerState, StateDetail } from "@/lib/types";

interface Props {
  state: string;
  stateData: StateDetail;
  peers: PeerState[];
}

function ZValue({ value }: { value: number | null }) {
  if (value === null) return <span className="text-gray-300">--</span>;
  const color =
    value >= 1
      ? "text-emerald-700"
      : value >= 0
        ? "text-emerald-600"
        : value >= -1
          ? "text-red-500"
          : "text-red-700";
  return (
    <span className={`tabular-nums font-medium ${color}`}>
      {value >= 0 ? "+" : ""}
      {value.toFixed(2)}
    </span>
  );
}

export default function PeerComparison({ state, stateData, peers }: Props) {
  // Find latest FY with scored data
  const ann = stateData.annual;
  let latestIdx = ann.fiscal_years.length - 1;
  while (latestIdx >= 0 && ann.composite_score[latestIdx] === null) {
    latestIdx--;
  }
  if (latestIdx < 0) return null;

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-gray-900 mb-3">
        Peer Comparison &mdash; FY {ann.fiscal_years[latestIdx]}
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="text-left py-2 pr-3 font-medium text-gray-700">
                State
              </th>
              <th className="text-right py-2 px-2 font-medium text-gray-700">
                Rank
              </th>
              <th className="text-right py-2 px-2 font-medium text-gray-700">
                Score
              </th>
              <th className="text-right py-2 px-2 font-medium text-gray-700">
                GST
              </th>
              <th className="text-right py-2 px-2 font-medium text-gray-700">
                Elec
              </th>
              <th className="text-right py-2 px-2 font-medium text-gray-700">
                Credit
              </th>
              <th className="text-right py-2 pl-2 font-medium text-gray-700">
                EPFO
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {/* Current state */}
            <tr className="bg-blue-50">
              <td className="py-2 pr-3 font-semibold text-gray-900">
                {state}
              </td>
              <td className="py-2 px-2 text-right tabular-nums">
                #{ann.rank[latestIdx] !== null ? Math.round(ann.rank[latestIdx]!) : "--"}
              </td>
              <td className="py-2 px-2 text-right tabular-nums font-semibold">
                {ann.composite_score[latestIdx]?.toFixed(2) ?? "--"}
              </td>
              <td className="py-2 px-2 text-right">
                <ZValue value={ann.gst_zscore[latestIdx]} />
              </td>
              <td className="py-2 px-2 text-right">
                <ZValue value={ann.electricity_zscore[latestIdx]} />
              </td>
              <td className="py-2 px-2 text-right">
                <ZValue value={ann.credit_zscore[latestIdx]} />
              </td>
              <td className="py-2 pl-2 text-right">
                <ZValue value={ann.epfo_zscore[latestIdx]} />
              </td>
            </tr>
            {/* Peers */}
            {peers.map((p) => (
              <tr key={p.slug}>
                <td className="py-2 pr-3">
                  <Link
                    href={`/states/${p.slug}`}
                    className="text-blue-600 hover:underline"
                  >
                    {p.state}
                  </Link>
                </td>
                <td className="py-2 px-2 text-right tabular-nums text-gray-600">
                  #{p.rank ?? "--"}
                </td>
                <td className="py-2 px-2 text-right tabular-nums text-gray-600">
                  {p.composite_score.toFixed(2)}
                </td>
                <td className="py-2 px-2 text-right text-gray-400" colSpan={4}>
                  <span className="text-xs">View state for details</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
