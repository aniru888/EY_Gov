import Link from "next/link";

export default function Footer() {
  return (
    <footer className="border-t border-gray-200 bg-gray-50 mt-12">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 text-sm text-gray-600">
          <div>
            <h3 className="font-semibold text-gray-900 mb-2">About</h3>
            <p>
              Composite State Economic Activity Index using GST collections,
              electricity demand, bank credit, and EPFO payroll data.
            </p>
            <p className="mt-2 text-gray-500">EY Government Advisory</p>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 mb-2">Data Sources</h3>
            <ul className="space-y-1">
              <li>
                GST Collections &mdash;{" "}
                <a
                  href="https://gst.gov.in"
                  className="text-blue-600 hover:underline"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  gst.gov.in
                </a>
              </li>
              <li>
                Electricity Demand &mdash; POSOCO/Grid India via Robbie Andrew
                (CC-BY-4.0)
              </li>
              <li>
                Bank Credit &mdash;{" "}
                <a
                  href="https://rbi.org.in"
                  className="text-blue-600 hover:underline"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  RBI Handbook Table 156
                </a>
              </li>
              <li>
                EPFO Payroll &mdash;{" "}
                <a
                  href="https://epfindia.gov.in"
                  className="text-blue-600 hover:underline"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  epfindia.gov.in
                </a>
              </li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 mb-2">Pages</h3>
            <ul className="space-y-1">
              <li>
                <Link href="/" className="text-blue-600 hover:underline">
                  Rankings
                </Link>
              </li>
              <li>
                <Link
                  href="/methodology"
                  className="text-blue-600 hover:underline"
                >
                  Methodology
                </Link>
              </li>
              <li>
                <Link href="/compare" className="text-blue-600 hover:underline">
                  Compare States
                </Link>
              </li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 mb-2">Top States</h3>
            <ul className="space-y-1">
              <li>
                <Link
                  href="/states/maharashtra"
                  className="text-blue-600 hover:underline"
                >
                  Maharashtra (#1)
                </Link>
              </li>
              <li>
                <Link
                  href="/states/tamil-nadu"
                  className="text-blue-600 hover:underline"
                >
                  Tamil Nadu (#2)
                </Link>
              </li>
              <li>
                <Link
                  href="/states/gujarat"
                  className="text-blue-600 hover:underline"
                >
                  Gujarat (#3)
                </Link>
              </li>
              <li>
                <Link
                  href="/states/karnataka"
                  className="text-blue-600 hover:underline"
                >
                  Karnataka (#4)
                </Link>
              </li>
              <li>
                <Link
                  href="/states/uttar-pradesh"
                  className="text-blue-600 hover:underline"
                >
                  Uttar Pradesh (#5)
                </Link>
              </li>
            </ul>
          </div>
        </div>
        <div className="mt-6 pt-4 border-t border-gray-200 text-xs text-gray-400 text-center">
          Electricity data licensed under CC-BY-4.0. Not for commercial
          redistribution.
        </div>
      </div>
    </footer>
  );
}
