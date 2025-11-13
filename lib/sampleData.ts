// Sample Meralco hazard data generator for testing the map

import { HazardReport, HazardType } from './types';

export function generateSampleReports(count: number = 10): HazardReport[] {
  const reports: HazardReport[] = [];
  
  // Metro Manila coordinates (approximate bounds)
  const center = { lat: 14.5995, lon: 120.9842 };
  const radius = 0.15; // ~15km radius
  
  const hazardTypes: HazardType[] = ['urgent', 'moderate', 'normal'];
  
  for (let i = 0; i < count; i++) {
    // Random position within radius
    const angle = Math.random() * 2 * Math.PI;
    const dist = Math.random() * radius;
    const lat = center.lat + dist * Math.cos(angle);
    const lon = center.lon + dist * Math.sin(angle);
    
    const hazardType = hazardTypes[Math.floor(Math.random() * hazardTypes.length)];
    const mlConfidence = 0.5 + Math.random() * 0.4; // 0.5 to 0.9
    const verificationCount = Math.floor(Math.random() * 3);
    
    // Calculate priority (Meralco formula)
    const base = hazardType === 'urgent' ? 0.9 : hazardType === 'moderate' ? 0.6 : 0.3;
    const priority = Math.min(1.0, base + mlConfidence * 0.3 + verificationCount * 0.1);
    
    const report: HazardReport = {
      id: `sample-${i}-${Date.now()}`,
      location: {
        latitude: lat,
        longitude: lon,
        accuracy: 10 + Math.random() * 30,
        timestamp: Date.now() - Math.random() * 86400000 // Random within last 24h
      },
      hazardType,
      imageDataUrl: '', // No images for sample data
      mlConfidence,
      priority,
      verificationCount,
      timestamp: Date.now() - Math.random() * 86400000,
      description: getSampleDescription(hazardType)
    };
    
    reports.push(report);
  }
  
  return reports;
}

function getSampleDescription(type: HazardType): string {
  const descriptions: Record<HazardType, string[]> = {
    urgent: [
      'Sparking transformer observed - immediate danger',
      'Severely leaning power pole (>20° tilt)',
      'Exposed live wires touching tree branches',
      'Smoke coming from electrical box',
      'Broken insulator with visible damage',
      'Flooded electrical equipment submerged in water'
    ],
    moderate: [
      'Tree branches near power lines - needs trimming',
      'Rust visible on utility pole',
      'Slightly tilted pole (<15° angle)',
      'Minor corrosion on equipment housing',
      'Vegetation growing close to transformer',
      'Loose cable ties on wiring'
    ],
    normal: [
      'Well-maintained power pole',
      'Clean transformer with proper clearance',
      'Recently painted utility pole',
      'Properly organized cables',
      'New installation, no issues observed',
      'Routine inspection - infrastructure in good condition'
    ]
  };
  
  const options = descriptions[type];
  return options[Math.floor(Math.random() * options.length)];
}
