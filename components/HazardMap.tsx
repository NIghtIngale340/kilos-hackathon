'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { HazardReport } from '@/lib/types';
import { reportStore } from '@/lib/reportStore';

// Dynamically import map component (client-side only)
const MapComponent = dynamic<{ reports: HazardReport[] }>(
  () => import('./MapComponent'),
  {
    ssr: false,
    loading: () => (
      <div className="h-full w-full flex items-center justify-center bg-zinc-100 dark:bg-zinc-800">
        <div className="text-center">
          <div className="text-4xl mb-2">🗺️</div>
          <p className="text-sm text-zinc-600 dark:text-zinc-400">Loading map...</p>
        </div>
      </div>
    )
  }
);

export default function HazardMap() {
  const [reports, setReports] = useState<HazardReport[]>([]);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    // Load existing reports
    setReports(reportStore.getReports());
    setIsLoaded(true);

    // Subscribe to new reports
    const unsubscribe = reportStore.subscribe((newReport) => {
      setReports(prev => [...prev, newReport]);
    });

    return () => unsubscribe();
  }, []);

  if (!isLoaded) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-zinc-100 dark:bg-zinc-800">
        <div className="text-center">
          <div className="text-4xl mb-2">🗺️</div>
          <p className="text-sm text-zinc-600 dark:text-zinc-400">Initializing map...</p>
        </div>
      </div>
    );
  }

  return <MapComponent reports={reports} />;
}
