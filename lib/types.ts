// Core types for KILOS Bayanihan - Meralco Infrastructure Hazards

export type HazardType = 'urgent' | 'moderate' | 'normal';

export interface GeoLocation {
  latitude: number;
  longitude: number;
  accuracy: number; // meters
  timestamp: number;
}

export interface HazardReport {
  id: string;
  location: GeoLocation;
  hazardType: HazardType;
  imageDataUrl: string; // base64 preview (max 500KB)
  mlConfidence: number;
  priority: number;
  verificationCount: number;
  timestamp: number;
  description?: string;
}

export interface MLClassificationResult {
  hazardType: HazardType;
  confidence: number;
  rawScores: number[]; // [urgent, moderate, normal]
}
