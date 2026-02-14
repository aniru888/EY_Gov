interface Props {
  label: string;
  value: string;
  subtext?: string;
  trend?: "up" | "down" | "neutral";
  trendLabel?: string;
  warning?: string;
}

export default function MetricCard({
  label,
  value,
  subtext,
  trend,
  trendLabel,
  warning,
}: Props) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="text-xs font-medium text-gray-500 uppercase tracking-wider">
        {label}
      </div>
      <div className="mt-1 text-2xl font-bold text-gray-900 tabular-nums">
        {value}
      </div>
      {subtext && <div className="text-sm text-gray-500 mt-0.5">{subtext}</div>}
      {trend && trendLabel && (
        <div
          className={`text-xs font-medium mt-1 ${
            trend === "up"
              ? "text-emerald-600"
              : trend === "down"
                ? "text-red-600"
                : "text-gray-400"
          }`}
        >
          {trend === "up" ? "\u25B2" : trend === "down" ? "\u25BC" : "\u2014"}{" "}
          {trendLabel}
        </div>
      )}
      {warning && (
        <div className="text-xs text-amber-600 mt-1">{warning}</div>
      )}
    </div>
  );
}
