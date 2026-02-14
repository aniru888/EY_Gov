import type { Metadata } from "next";
import { Suspense } from "react";
import { getRankings, getTrends } from "@/lib/data";
import Breadcrumbs from "@/components/common/Breadcrumbs";
import CompareClient from "@/components/CompareClient";

export const metadata: Metadata = {
  title: "Compare States | State Economic Activity Index",
  description:
    "Compare economic activity across Indian states: overlay trends, side-by-side component analysis.",
};

export default function ComparePage() {
  const rankings = getRankings();
  const trends = getTrends();

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
      <Breadcrumbs
        items={[
          { label: "Home", href: "/" },
          { label: "Compare States" },
        ]}
      />

      <h1 className="text-3xl font-bold text-gray-900 mb-2">
        Compare States
      </h1>
      <p className="text-gray-600 mb-6">
        Select 2-5 states to overlay their composite score trends and compare
        individual components side by side.
      </p>

      <Suspense
        fallback={
          <div className="h-96 bg-gray-50 rounded-lg animate-pulse" />
        }
      >
        <CompareClient rankings={rankings.rankings} trends={trends} />
      </Suspense>
    </div>
  );
}
