import type { RankingEntry } from "@/lib/types";

interface Props {
  states: RankingEntry[];
}

function CellValue({
  value,
  isMax,
}: {
  value: number | null;
  isMax: boolean;
}) {
  if (value === null) return <span className="text-gray-300">--</span>;
  return (
    <span
      className={`tabular-nums ${isMax ? "font-bold text-emerald-700" : "text-gray-700"}`}
    >
      {value >= 0 ? "+" : ""}
      {value.toFixed(2)}
    </span>
  );
}

const ROWS: {
  label: string;
  key: keyof RankingEntry;
}[] = [
  { label: "Composite Score", key: "composite_score" },
  { label: "GST Z-Score", key: "gst_zscore" },
  { label: "Electricity Z-Score", key: "electricity_zscore" },
  { label: "Bank Credit Z-Score", key: "credit_zscore" },
  { label: "EPFO Z-Score", key: "epfo_zscore" },
];

export default function ComparisonTable({ states }: Props) {
  if (states.length === 0) return null;

  return (
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="text-left px-4 py-3 font-medium text-gray-700">
                Metric
              </th>
              {states.map((s) => (
                <th
                  key={s.slug}
                  className="text-right px-4 py-3 font-medium text-gray-700"
                >
                  {s.state}
                  <div className="text-xs text-gray-400 font-normal">
                    #{s.rank}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {ROWS.map((row) => {
              const values = states.map(
                (s) => s[row.key] as number | null
              );
              const maxVal = Math.max(
                ...values.filter((v): v is number => v !== null)
              );
              return (
                <tr key={row.key}>
                  <td className="px-4 py-3 font-medium text-gray-700">
                    {row.label}
                  </td>
                  {values.map((v, i) => (
                    <td key={states[i].slug} className="px-4 py-3 text-right">
                      <CellValue value={v} isMax={v === maxVal && v !== null} />
                    </td>
                  ))}
                </tr>
              );
            })}
            <tr>
              <td className="px-4 py-3 font-medium text-gray-700">
                Components
              </td>
              {states.map((s) => (
                <td key={s.slug} className="px-4 py-3 text-right text-gray-600">
                  {s.n_components}/4
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
