import fs from "fs";
import path from "path";
import type {
  RankingsData,
  MetadataData,
  TrendsData,
  StateDetail,
  TrendChartPoint,
  InsightsData,
  RegressionData,
} from "./types";

const DATA_DIR = path.join(process.cwd(), "public", "data");

export function getRankings(): RankingsData {
  const raw = fs.readFileSync(path.join(DATA_DIR, "rankings.json"), "utf-8");
  return JSON.parse(raw);
}

export function getMetadata(): MetadataData {
  const raw = fs.readFileSync(path.join(DATA_DIR, "metadata.json"), "utf-8");
  return JSON.parse(raw);
}

export function getTrends(): TrendsData {
  const raw = fs.readFileSync(path.join(DATA_DIR, "trends.json"), "utf-8");
  return JSON.parse(raw);
}

export function getStateData(slug: string): StateDetail {
  const raw = fs.readFileSync(
    path.join(DATA_DIR, "states", `${slug}.json`),
    "utf-8"
  );
  return JSON.parse(raw);
}

/** Returns all state slugs by reading the states directory */
export function getAllStateSlugs(): string[] {
  const statesDir = path.join(DATA_DIR, "states");
  if (!fs.existsSync(statesDir)) return [];
  return fs
    .readdirSync(statesDir)
    .filter((f) => f.endsWith(".json"))
    .map((f) => f.replace(".json", ""));
}

/**
 * Extracts composite scores for selected states into Recharts-friendly format.
 * Pivots from per-state arrays into per-FY rows with slug columns.
 */
export function getTrendSubset(
  trends: TrendsData,
  slugs: string[]
): TrendChartPoint[] {
  const fySet = new Set<string>();
  const stateData: Record<string, Record<string, number | null>> = {};

  for (const slug of slugs) {
    const entry = trends.annual[slug];
    if (!entry) continue;
    stateData[slug] = {};
    for (let i = 0; i < entry.fiscal_years.length; i++) {
      const fy = entry.fiscal_years[i];
      fySet.add(fy);
      stateData[slug][fy] = entry.composite_score[i];
    }
  }

  const fiscalYears = Array.from(fySet).sort();
  return fiscalYears.map((fy) => {
    const point: TrendChartPoint = { fiscal_year: fy };
    for (const slug of slugs) {
      point[slug] = stateData[slug]?.[fy] ?? null;
    }
    return point;
  });
}

export function getInsights(): InsightsData {
  const raw = fs.readFileSync(path.join(DATA_DIR, "insights.json"), "utf-8");
  return JSON.parse(raw);
}

export function getRegressionData(): RegressionData {
  const raw = fs.readFileSync(
    path.join(DATA_DIR, "regression.json"),
    "utf-8"
  );
  return JSON.parse(raw);
}

/** Format number in Indian lakh/crore notation */
export function formatIndianNumber(n: number | null): string {
  if (n === null) return "--";
  const abs = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  if (abs >= 1e7) return `${sign}${(abs / 1e7).toFixed(2)} Cr`;
  if (abs >= 1e5) return `${sign}${(abs / 1e5).toFixed(2)} L`;
  if (abs >= 1e3) return `${sign}${(abs / 1e3).toFixed(1)}K`;
  return `${sign}${abs.toFixed(0)}`;
}
