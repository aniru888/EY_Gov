import type { MetadataData } from "@/lib/types";

interface Props {
  metadata: MetadataData;
}

export default function DataFreshness({ metadata }: Props) {
  const generatedDate = new Date(metadata.generated_at).toLocaleDateString(
    "en-IN",
    { day: "numeric", month: "short", year: "numeric" }
  );

  return (
    <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-500">
      <span>
        Data generated: <span className="font-medium">{generatedDate}</span>
      </span>
      {metadata.methodology.components.map((c) => (
        <span key={c.column} className="flex items-center gap-1">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-gray-400" />
          {c.name}: {c.frequency}
        </span>
      ))}
    </div>
  );
}
