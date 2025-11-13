/*
Priority calculation for Meralco infrastructure hazards.
Formula:
  - base = 0.9 for 'urgent', 0.6 for 'moderate', 0.3 for 'normal'
  - priority = min(1.0, base + (mlConfidence * 0.3) + (verificationCount * 0.1))

Also exposes a helper for applying verification boosts for nearby reports.
*/

export type HazardType = 'urgent' | 'moderate' | 'normal';

export function calculatePriority(
  mlConfidence: number,
  verificationCount: number,
  hazardType: HazardType
) {
  // Meralco-specific base priorities
  const base = hazardType === 'urgent' ? 0.9 : hazardType === 'moderate' ? 0.6 : 0.3;
  const priority = Math.min(1.0, base + mlConfidence * 0.3 + verificationCount * 0.1);
  return priority;
}

export function verificationBoostByNearbyReports(originalPriority: number, nearbyCount: number) {
  // Each additional report within 50m increases priority by 15% (per prompt)
  const boosted = originalPriority * (1 + 0.15 * nearbyCount);
  return Math.min(1.0, boosted);
}
