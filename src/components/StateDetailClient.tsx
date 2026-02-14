"use client";

import dynamic from "next/dynamic";
import type { StateDetail } from "@/lib/types";

const StateScoreTrend = dynamic(
  () => import("@/components/charts/StateScoreTrend"),
  {
    ssr: false,
    loading: () => (
      <div className="h-72 bg-gray-50 rounded-lg animate-pulse" />
    ),
  }
);

const ComponentTrends = dynamic(
  () => import("@/components/charts/ComponentTrends"),
  {
    ssr: false,
    loading: () => (
      <div className="h-80 bg-gray-50 rounded-lg animate-pulse" />
    ),
  }
);

const MonthlySubIndex = dynamic(
  () => import("@/components/charts/MonthlySubIndex"),
  {
    ssr: false,
    loading: () => (
      <div className="h-72 bg-gray-50 rounded-lg animate-pulse" />
    ),
  }
);

interface Props {
  stateData: StateDetail;
}

export default function StateDetailClient({ stateData }: Props) {
  const ann = stateData.annual;
  const mon = stateData.monthly;

  return (
    <div className="space-y-6">
      {/* Composite score trend */}
      <StateScoreTrend
        fiscalYears={ann.fiscal_years}
        compositeScore={ann.composite_score}
      />

      {/* Component z-score trends */}
      <ComponentTrends
        fiscalYears={ann.fiscal_years}
        gstZscore={ann.gst_zscore}
        electricityZscore={ann.electricity_zscore}
        creditZscore={ann.credit_zscore}
        epfoZscore={ann.epfo_zscore}
      />

      {/* Monthly sub-index */}
      {mon.months.length > 0 && (
        <MonthlySubIndex
          months={mon.months}
          gstTotal={mon.gst_total}
          electricityMu={mon.electricity_mu}
        />
      )}
    </div>
  );
}
