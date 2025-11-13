import Image from "next/image";
import Link from "next/link";

export default function Home() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-zinc-950">
      <main className="flex min-h-screen w-full max-w-3xl flex-col items-center justify-between py-16 px-6 bg-white dark:bg-zinc-900 sm:items-start">
        <div className="w-full">
          <h1 className="text-4xl font-bold text-zinc-900 dark:text-zinc-50 mb-2">
            KILOS Bayanihan
          </h1>
          <p className="text-lg text-zinc-600 dark:text-zinc-400 mb-2">
            Crowdsourced Electrical Infrastructure Hazard Reporting
          </p>
          <p className="text-sm text-zinc-500 dark:text-zinc-500 mb-8">
            Powered by ML • Built for Meralco Engineers
          </p>
        </div>

        <div className="flex flex-col items-center gap-8 text-center sm:items-start sm:text-left w-full">
          <div className="bg-zinc-50 dark:bg-zinc-800 rounded-lg p-6 w-full">
            <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-50 mb-4">
              Community-Powered Infrastructure Monitoring
            </h2>
            <p className="text-zinc-700 dark:text-zinc-300 mb-4">
              Citizens snap geotagged photos of electrical infrastructure hazards
              (leaning poles, sparking transformers, vegetation interference). 
              Reports populate a live <strong>Hazard Heatmap</strong> for Meralco engineers.
              Multiple user verification auto-escalates priority.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
              <div className="bg-white dark:bg-zinc-900 p-4 rounded border border-zinc-200 dark:border-zinc-700">
                <div className="text-2xl mb-2">📸</div>
                <h3 className="font-semibold text-sm text-zinc-900 dark:text-zinc-50 mb-1">
                  Geotagged Photos
                </h3>
                <p className="text-xs text-zinc-600 dark:text-zinc-400">
                  GPS-validated reports (&lt;50m accuracy)
                </p>
              </div>
              
              <div className="bg-white dark:bg-zinc-900 p-4 rounded border border-zinc-200 dark:border-zinc-700">
                <div className="text-2xl mb-2">🤖</div>
                <h3 className="font-semibold text-sm text-zinc-900 dark:text-zinc-50 mb-1">
                  ML Classification
                </h3>
                <p className="text-xs text-zinc-600 dark:text-zinc-400">
                  Urgent/Moderate/Normal detection
                </p>
              </div>
              
              <div className="bg-white dark:bg-zinc-900 p-4 rounded border border-zinc-200 dark:border-zinc-700">
                <div className="text-2xl mb-2">🎯</div>
                <h3 className="font-semibold text-sm text-zinc-900 dark:text-zinc-50 mb-1">
                  Auto-Prioritization
                </h3>
                <p className="text-xs text-zinc-600 dark:text-zinc-400">
                  Smart triage with verification boost
                </p>
              </div>
            </div>
          </div>

          <div className="bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800 rounded-lg p-4 w-full">
            <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
              ⚡ Current Implementation Status
            </h3>
            <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
              <li>✅ Mobile-first report form with camera capture</li>
              <li>✅ GPS validation with manual coordinate fallback</li>
              <li>✅ TensorFlow.js ML classifier (client-side)</li>
              <li>✅ Priority calculation with verification boost</li>
              <li>✅ Live hazard heatmap with Leaflet</li>
              <li>✅ Real-time updates from report store</li>
              <li>✅ Interactive markers with popups</li>
              <li>⏳ Engineer dashboard with filters (coming next)</li>
            </ul>
          </div>
        </div>

        <div className="flex flex-col gap-4 text-base font-medium sm:flex-row w-full">
          <Link
            className="flex h-12 w-full items-center justify-center gap-2 rounded-full bg-blue-600 px-5 text-white transition-colors hover:bg-blue-700 md:w-auto md:flex-1"
            href="/report"
          >
            📱 Report a Hazard
          </Link>
          <Link
            className="flex h-12 w-full items-center justify-center rounded-full border border-solid border-zinc-300 dark:border-zinc-700 px-5 transition-colors hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-900 dark:text-zinc-50 md:w-auto md:flex-1"
            href="/map"
          >
            🗺️ View Live Heatmap
          </Link>
          <Link
            className="flex h-12 w-full items-center justify-center rounded-full border border-solid border-green-300 dark:border-green-700 px-5 transition-colors hover:bg-green-50 dark:hover:bg-green-950 text-green-900 dark:text-green-50 md:w-auto md:flex-1"
            href="/test-model"
          >
            🤖 Test AI Model
          </Link>
        </div>
      </main>
    </div>
  );
}
