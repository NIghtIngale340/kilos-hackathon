// In-memory event store for hazard reports (mock WebSocket replacement)

import { HazardReport } from './types';

type ReportListener = (report: HazardReport) => void;

class ReportStore {
  private reports: HazardReport[] = [];
  private listeners: ReportListener[] = [];

  addReport(report: HazardReport) {
    this.reports.push(report);
    // Notify all listeners
    this.listeners.forEach(listener => listener(report));
  }

  getReports(): HazardReport[] {
    return [...this.reports];
  }

  subscribe(listener: ReportListener): () => void {
    this.listeners.push(listener);
    // Return unsubscribe function
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  // Find nearby reports within radius (meters)
  findNearbyReports(lat: number, lon: number, radiusMeters: number): HazardReport[] {
    return this.reports.filter(report => {
      const distance = this.calculateDistance(
        lat,
        lon,
        report.location.latitude,
        report.location.longitude
      );
      return distance <= radiusMeters;
    });
  }

  // Haversine formula for distance in meters
  private calculateDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
    const R = 6371e3; // Earth radius in meters
    const φ1 = (lat1 * Math.PI) / 180;
    const φ2 = (lat2 * Math.PI) / 180;
    const Δφ = ((lat2 - lat1) * Math.PI) / 180;
    const Δλ = ((lon2 - lon1) * Math.PI) / 180;

    const a =
      Math.sin(Δφ / 2) * Math.sin(Δφ / 2) +
      Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

    return R * c;
  }
}

// Singleton instance
export const reportStore = new ReportStore();
