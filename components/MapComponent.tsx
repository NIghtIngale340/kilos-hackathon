'use client';

import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { HazardReport, HazardType } from '@/lib/types';

// Fix Leaflet default marker icons
const fixLeafletIcons = () => {
  delete (L.Icon.Default.prototype as any)._getIconUrl;
  L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
    iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
    shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  });
};

interface MapComponentProps {
  reports: HazardReport[];
}

export default function MapComponent({ reports }: MapComponentProps) {
  const mapRef = useRef<L.Map | null>(null);
  const markersRef = useRef<L.LayerGroup | null>(null);
  const [mapReady, setMapReady] = useState(false);

  // Initialize map
  useEffect(() => {
    if (typeof window === 'undefined' || mapRef.current) return;

    fixLeafletIcons();

    // Default to Metro Manila center
    const map = L.map('map', {
      center: [14.5995, 120.9842],
      zoom: 12,
      zoomControl: true,
    });

    // Add tile layer (dark mode compatible)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: 'abcd',
      maxZoom: 20
    }).addTo(map);

    // Create layer group for markers
    const markersLayer = L.layerGroup().addTo(map);
    markersRef.current = markersLayer;

    mapRef.current = map;
    setMapReady(true);

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  // Update markers when reports change
  useEffect(() => {
    if (!mapReady || !markersRef.current || !mapRef.current) return;

    // Clear existing markers
    markersRef.current.clearLayers();

    // Create custom icons for different hazard types
    const getMarkerIcon = (hazardType: HazardType, priority: number) => {
      // Meralco hazard colors
      const colors: Record<HazardType, string> = {
        urgent: '#ef4444',      // red - critical
        moderate: '#f59e0b',    // orange - needs attention
        normal: '#22c55e'       // green - routine
      };

      // Meralco-specific emoji icons
      const emojis: Record<HazardType, string> = {
        urgent: '⚡',     // Lightning for urgent electrical hazards
        moderate: '⚠️',   // Warning for moderate issues
        normal: '✓'      // Check for normal/safe
      };

      const color = colors[hazardType];
      const emoji = emojis[hazardType];
      const size = priority > 0.8 ? 30 : priority > 0.6 ? 25 : 20;

      return L.divIcon({
        className: 'custom-marker',
        html: `
          <div style="
            width: ${size}px;
            height: ${size}px;
            background-color: ${color};
            border: 3px solid white;
            border-radius: 50%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: ${size * 0.5}px;
          ">
            ${emoji}
          </div>
        `,
        iconSize: [size, size],
        iconAnchor: [size / 2, size / 2],
      });
    };

    // Add markers for all reports
    const markers: L.Marker[] = [];
    reports.forEach((report) => {
      const marker = L.marker(
        [report.location.latitude, report.location.longitude],
        { icon: getMarkerIcon(report.hazardType, report.priority) }
      );

      // Create popup content
      const popupContent = `
        <div style="min-width: 200px; color: #333;">
          <h3 style="font-weight: bold; margin-bottom: 8px; text-transform: capitalize; color: #000;">
            ${report.hazardType} Hazard
          </h3>
          <div style="margin-bottom: 8px;">
            ${report.imageDataUrl ? `<img src="${report.imageDataUrl}" style="width: 100%; height: 120px; object-fit: cover; border-radius: 4px; margin-bottom: 8px;" />` : ''}
          </div>
          <div style="font-size: 12px; color: #666;">
            <p style="margin: 4px 0;"><strong>Priority:</strong> ${(report.priority * 100).toFixed(0)}%</p>
            <p style="margin: 4px 0;"><strong>Confidence:</strong> ${(report.mlConfidence * 100).toFixed(1)}%</p>
            <p style="margin: 4px 0;"><strong>Verifications:</strong> ${report.verificationCount}</p>
            <p style="margin: 4px 0;"><strong>Location:</strong> ${report.location.latitude.toFixed(4)}, ${report.location.longitude.toFixed(4)}</p>
            <p style="margin: 4px 0;"><strong>Reported:</strong> ${new Date(report.timestamp).toLocaleString()}</p>
            ${report.description ? `<p style="margin: 4px 0;"><strong>Notes:</strong> ${report.description}</p>` : ''}
          </div>
        </div>
      `;

      marker.bindPopup(popupContent, { maxWidth: 250 });
      marker.addTo(markersRef.current!);
      markers.push(marker);
    });

    // Auto-fit bounds if we have markers
    if (markers.length > 0) {
      const group = L.featureGroup(markers);
      mapRef.current.fitBounds(group.getBounds(), { padding: [50, 50], maxZoom: 15 });
    }

  }, [reports, mapReady]);

  return (
    <div className="relative w-full h-full">
      <div id="map" className="w-full h-full rounded-lg" />
      
      {/* Legend */}
      <div className="absolute bottom-4 right-4 bg-white dark:bg-zinc-900 rounded-lg shadow-lg p-3 text-xs z-[1000]">
        <h4 className="font-semibold mb-2 text-zinc-900 dark:text-zinc-50">Legend</h4>
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 bg-red-500 rounded-full flex items-center justify-center">🔥</div>
            <span className="text-zinc-700 dark:text-zinc-300">Fire</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 bg-orange-500 rounded-full flex items-center justify-center">⚠️</div>
            <span className="text-zinc-700 dark:text-zinc-300">Structural</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 bg-green-500 rounded-full flex items-center justify-center">🌳</div>
            <span className="text-zinc-700 dark:text-zinc-300">Vegetation</span>
          </div>
        </div>
        <div className="mt-2 pt-2 border-t border-zinc-200 dark:border-zinc-700">
          <p className="text-zinc-600 dark:text-zinc-400">Size = Priority</p>
        </div>
      </div>

      {/* Stats overlay - Meralco categories */}
      <div className="absolute top-4 left-4 bg-white dark:bg-zinc-900 rounded-lg shadow-lg p-3 z-[1000]">
        <div className="flex items-center gap-3">
          <div className="text-center">
            <div className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">{reports.length}</div>
            <div className="text-xs text-zinc-600 dark:text-zinc-400">Total Reports</div>
          </div>
          <div className="h-10 w-px bg-zinc-200 dark:bg-zinc-700" />
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600 dark:text-red-400">
              {reports.filter(r => r.hazardType === 'urgent').length}
            </div>
            <div className="text-xs text-zinc-600 dark:text-zinc-400">Urgent</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">
              {reports.filter(r => r.hazardType === 'moderate').length}
            </div>
            <div className="text-xs text-zinc-600 dark:text-zinc-400">Moderate</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
              {reports.filter(r => r.hazardType === 'normal').length}
            </div>
            <div className="text-xs text-zinc-600 dark:text-zinc-400">Normal</div>
          </div>
        </div>
      </div>
    </div>
  );
}
