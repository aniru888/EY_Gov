"use client";

import { useEffect, useState } from "react";

const sections = [
  { id: "origin", label: "Origin & Adaptation" },
  { id: "components", label: "The Four Components" },
  { id: "calculation", label: "How It's Calculated" },
  { id: "zscore", label: "Z-Score Normalization" },
  { id: "sources", label: "Data Sources" },
  { id: "limitations", label: "Known Limitations" },
  { id: "validation", label: "Validation" },
  { id: "analytical-metrics", label: "Analytical Metrics" },
  { id: "statistical-validation", label: "Statistical Validation" },
  { id: "panel-fe", label: "Panel Fixed Effects" },
  { id: "loglog-pca", label: "Log-Log & PCA" },
  { id: "gsdp-relationship", label: "Relationship with GSDP" },
  { id: "percapita", label: "Per-Capita Performance" },
  { id: "size-dependency", label: "Size Dependency" },
  { id: "updated-limitations", label: "Additional Caveats" },
];

export default function TableOfContents() {
  const [activeId, setActiveId] = useState("");

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id);
          }
        }
      },
      { rootMargin: "-80px 0px -60% 0px", threshold: 0.1 }
    );

    for (const section of sections) {
      const el = document.getElementById(section.id);
      if (el) observer.observe(el);
    }

    return () => observer.disconnect();
  }, []);

  return (
    <nav className="space-y-1">
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
        Contents
      </div>
      {sections.map((s) => (
        <a
          key={s.id}
          href={`#${s.id}`}
          className={`block text-sm py-1 px-2 rounded transition-colors ${
            activeId === s.id
              ? "bg-blue-50 text-blue-700 font-medium"
              : "text-gray-600 hover:text-gray-900 hover:bg-gray-50"
          }`}
        >
          {s.label}
        </a>
      ))}
    </nav>
  );
}
