const steps = [
  {
    num: "01",
    label: "GST Collections",
    detail: "9 Excel files from gst.gov.in",
    color: "bg-blue-100 text-blue-800 border-blue-300",
  },
  {
    num: "02",
    label: "Electricity Demand",
    detail: "Daily CSV from POSOCO/Grid India",
    color: "bg-yellow-100 text-yellow-800 border-yellow-300",
  },
  {
    num: "03",
    label: "Bank Credit",
    detail: "RBI Handbook Table 156",
    color: "bg-green-100 text-green-800 border-green-300",
  },
  {
    num: "04",
    label: "EPFO Payroll",
    detail: "Monthly Excel from epfindia.gov.in",
    color: "bg-purple-100 text-purple-800 border-purple-300",
  },
  {
    num: "05",
    label: "Clean & Merge",
    detail: "Standardize names, align time, join on state x FY",
    color: "bg-gray-100 text-gray-800 border-gray-300",
  },
  {
    num: "06",
    label: "Compute Index",
    detail: "Z-score normalize, equal-weight average, rank",
    color: "bg-orange-100 text-orange-800 border-orange-300",
  },
  {
    num: "07",
    label: "Generate JSON",
    detail: "Rankings, trends, state detail, metadata",
    color: "bg-indigo-100 text-indigo-800 border-indigo-300",
  },
];

export default function PipelineDiagram() {
  return (
    <div className="my-6">
      <div className="flex flex-col gap-2">
        {steps.map((step, i) => (
          <div key={step.num} className="flex items-start gap-3">
            {/* Connector line */}
            <div className="flex flex-col items-center w-8 shrink-0">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold border ${step.color}`}
              >
                {step.num}
              </div>
              {i < steps.length - 1 && (
                <div className="w-0.5 h-4 bg-gray-300" />
              )}
            </div>
            {/* Content */}
            <div className={`flex-1 rounded-lg border p-3 ${step.color}`}>
              <div className="font-semibold text-sm">{step.label}</div>
              <div className="text-xs opacity-75 mt-0.5">{step.detail}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
