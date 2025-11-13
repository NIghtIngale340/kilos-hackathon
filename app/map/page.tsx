'use client';

import Link from 'next/link';
import HazardMap from '@/components/HazardMap';
import { generateSampleReports } from '@/lib/sampleData';
import { reportStore } from '@/lib/reportStore';

export default function MapPage() {
  const loadSampleData = () => {
    const samples = generateSampleReports(15);
    samples.forEach(report => reportStore.addReport(report));
  };

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 py-4 px-4">
      <div className="max-w-7xl mx-auto h-[calc(100vh-2rem)]">
        <div className="bg-white dark:bg-zinc-900 rounded-lg shadow-lg p-4 h-full flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
                Live Hazard Heatmap
              </h1>
              <p className="text-sm text-zinc-600 dark:text-zinc-400">
                Real-time infrastructure hazard reports
              </p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={loadSampleData}
                className="px-4 py-2 bg-zinc-600 text-white rounded-lg hover:bg-zinc-700 transition-colors text-sm font-semibold"
              >
                🧪 Load Sample Data
              </button>
              <Link
                href="/report"
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-semibold"
              >
                📱 Report Hazard
              </Link>
            </div>
          </div>

          <div className="flex-1 min-h-0">
            <HazardMap />
          </div>

          <div className="mt-4 flex items-center justify-between text-xs text-zinc-600 dark:text-zinc-400">
            <div className="flex gap-4">
              <span>🔄 Live Updates</span>
              <span>📍 Metro Manila</span>
              <span>🌙 Dark Mode Optimized</span>
            </div>
            <Link href="/" className="hover:underline">
              ← Back to Home
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
