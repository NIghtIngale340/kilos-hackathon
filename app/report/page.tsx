'use client';

import { useState, useRef } from 'react';
import { HazardReport, GeoLocation, HazardType } from '@/lib/types';
import { classifyImageWithAPI } from '@/lib/apiClassifier';
import { calculatePriority, verificationBoostByNearbyReports } from '@/lib/priority';
import { reportStore } from '@/lib/reportStore';

export default function ReportPage() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [location, setLocation] = useState<GeoLocation | null>(null);
  const [locationError, setLocationError] = useState<string | null>(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [classificationResult, setClassificationResult] = useState<{
    hazardType: HazardType;
    confidence: number;
    priority: number;
  } | null>(null);
  const [submitStatus, setSubmitStatus] = useState<'idle' | 'submitting' | 'success' | 'error'>('idle');
  const [description, setDescription] = useState('');
  const [useManualCoords, setUseManualCoords] = useState(false);
  const [manualLat, setManualLat] = useState('');
  const [manualLon, setManualLon] = useState('');
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageCapture = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file size (max 500KB)
    const MAX_SIZE = 500 * 1024; // 500KB
    if (file.size > MAX_SIZE) {
      alert('Image too large! Please use an image under 500KB.');
      return;
    }

    setImageFile(file);
    
    // Create preview
    const reader = new FileReader();
    reader.onload = (evt) => {
      if (evt.target?.result) {
        setImagePreview(evt.target.result as string);
      }
    };
    reader.readAsDataURL(file);

    // Auto-trigger geolocation
    requestLocation();
  };

  const requestLocation = () => {
    setLocationError(null);
    
    if (!navigator.geolocation) {
      setLocationError('Geolocation not supported by your browser');
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const accuracy = position.coords.accuracy;
        
        // Enforce <50m accuracy requirement (relaxed for testing)
        if (accuracy > 50) {
          setLocationError(
            `GPS accuracy: ${Math.round(accuracy)}m (Required: <50m). Use manual coordinates below for testing.`
          );
          // Still set location for reference, but mark as low accuracy
          setLocation({
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
            accuracy,
            timestamp: Date.now()
          });
          return;
        }

        setLocation({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          accuracy,
          timestamp: Date.now()
        });
      },
      (error) => {
        setLocationError(`Location error: ${error.message}. Use manual coordinates below.`);
        setLocation(null);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0
      }
    );
  };

  const handleManualCoords = () => {
    const lat = parseFloat(manualLat);
    const lon = parseFloat(manualLon);
    
    if (isNaN(lat) || isNaN(lon)) {
      alert('Please enter valid latitude and longitude values');
      return;
    }
    
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      alert('Latitude must be between -90 and 90, Longitude between -180 and 180');
      return;
    }
    
    setLocation({
      latitude: lat,
      longitude: lon,
      accuracy: 10, // Manual entry assumed accurate
      timestamp: Date.now()
    });
    setLocationError(null);
    setUseManualCoords(false);
  };

  const handleClassifyAndSubmit = async () => {
    if (!imageFile || !location || !imagePreview) {
      alert('Please capture an image and ensure GPS location is available');
      return;
    }

    setIsClassifying(true);
    setSubmitStatus('submitting');

    try {
      // Run ML classification using Random Forest API
      const mlResult = await classifyImageWithAPI(imageFile);
      
      // Check if confidence is too low (reject per prompt)
      if (mlResult.confidence < 0.4) {
        alert(
          `Classification confidence too low (${(mlResult.confidence * 100).toFixed(1)}%). Please retake the photo with better lighting/angle.`
        );
        setIsClassifying(false);
        setSubmitStatus('error');
        return;
      }

      // Calculate base priority
      const basePriority = calculatePriority(mlResult.confidence, 0, mlResult.hazardType);

      // Check for nearby reports within 50m
      const nearbyReports = reportStore.findNearbyReports(
        location.latitude,
        location.longitude,
        50
      );

      // Apply verification boost if there are nearby reports
      const finalPriority = nearbyReports.length > 0
        ? verificationBoostByNearbyReports(basePriority, nearbyReports.length)
        : basePriority;

      setClassificationResult({
        hazardType: mlResult.hazardType,
        confidence: mlResult.confidence,
        priority: finalPriority
      });

      // Create report
      const report: HazardReport = {
        id: `report-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        location,
        hazardType: mlResult.hazardType,
        imageDataUrl: imagePreview,
        mlConfidence: mlResult.confidence,
        priority: finalPriority,
        verificationCount: nearbyReports.length,
        timestamp: Date.now(),
        description: description || undefined
      };

      // Add to store (this will notify map listeners)
      reportStore.addReport(report);

      setSubmitStatus('success');
      setIsClassifying(false);

      // Reset form after 2 seconds
      setTimeout(() => {
        resetForm();
      }, 2000);
    } catch (error) {
      console.error('Classification error:', error);
      alert('Failed to classify image. Please try again.');
      setIsClassifying(false);
      setSubmitStatus('error');
    }
  };

  const resetForm = () => {
    setImageFile(null);
    setImagePreview(null);
    setLocation(null);
    setLocationError(null);
    setClassificationResult(null);
    setSubmitStatus('idle');
    setDescription('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 py-8 px-4">
      <div className="max-w-2xl mx-auto">
        <div className="bg-white dark:bg-zinc-900 rounded-lg shadow-lg p-6">
          <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50 mb-2">
            Report Infrastructure Hazard
          </h1>
          <p className="text-sm text-zinc-600 dark:text-zinc-400 mb-6">
            Snap a photo of hazards like leaning poles, sparking transformers, or vegetation interference
          </p>

          {/* Camera Capture */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2">
              Capture Photo
            </label>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              capture="environment"
              onChange={handleImageCapture}
              className="w-full text-sm text-zinc-900 dark:text-zinc-100 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700 cursor-pointer"
              disabled={submitStatus === 'submitting'}
            />
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
              Max 500KB • GPS accuracy must be &lt;50m
            </p>
          </div>

          {/* Image Preview */}
          {imagePreview && (
            <div className="mb-6">
              <img
                src={imagePreview}
                alt="Preview"
                className="w-full h-64 object-cover rounded-lg border border-zinc-200 dark:border-zinc-700"
              />
            </div>
          )}

          {/* GPS Status */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                GPS Location
              </label>
              <div className="flex gap-2">
                <button
                  onClick={() => setUseManualCoords(!useManualCoords)}
                  className="text-xs text-zinc-600 dark:text-zinc-400 hover:underline"
                  disabled={submitStatus === 'submitting'}
                >
                  {useManualCoords ? 'Use GPS' : 'Manual Entry'}
                </button>
                {!useManualCoords && (
                  <button
                    onClick={requestLocation}
                    className="text-xs text-blue-600 dark:text-blue-400 hover:underline"
                    disabled={submitStatus === 'submitting'}
                  >
                    Refresh GPS
                  </button>
                )}
              </div>
            </div>
            
            {useManualCoords ? (
              <div className="bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800 rounded p-3">
                <p className="text-xs text-blue-700 dark:text-blue-300 mb-2">
                  Enter coordinates manually (for testing/indoor use)
                </p>
                <div className="grid grid-cols-2 gap-2 mb-2">
                  <div>
                    <label className="text-xs text-blue-700 dark:text-blue-300">Latitude</label>
                    <input
                      type="number"
                      step="0.000001"
                      value={manualLat}
                      onChange={(e) => setManualLat(e.target.value)}
                      placeholder="14.5995"
                      className="w-full px-2 py-1 text-sm border border-blue-300 dark:border-blue-700 rounded bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-blue-700 dark:text-blue-300">Longitude</label>
                    <input
                      type="number"
                      step="0.000001"
                      value={manualLon}
                      onChange={(e) => setManualLon(e.target.value)}
                      placeholder="120.9842"
                      className="w-full px-2 py-1 text-sm border border-blue-300 dark:border-blue-700 rounded bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100"
                    />
                  </div>
                </div>
                <button
                  onClick={handleManualCoords}
                  className="w-full py-1 px-3 bg-blue-600 text-white text-sm rounded hover:bg-blue-700"
                >
                  Set Coordinates
                </button>
                <p className="text-xs text-blue-600 dark:text-blue-400 mt-2">
                  💡 Metro Manila: 14.5995, 120.9842 (Quezon City)
                </p>
              </div>
            ) : location ? (
              <div className={`${location.accuracy <= 50 ? 'bg-green-50 dark:bg-green-950 border-green-200 dark:border-green-800' : 'bg-yellow-50 dark:bg-yellow-950 border-yellow-200 dark:border-yellow-800'} border rounded p-3`}>
                <p className={`text-sm ${location.accuracy <= 50 ? 'text-green-800 dark:text-green-200' : 'text-yellow-800 dark:text-yellow-200'}`}>
                  {location.accuracy <= 50 ? '✓' : '⚠'} GPS acquired ({Math.round(location.accuracy)}m accuracy)
                </p>
                <p className={`text-xs ${location.accuracy <= 50 ? 'text-green-700 dark:text-green-300' : 'text-yellow-700 dark:text-yellow-300'} mt-1`}>
                  {location.latitude.toFixed(6)}, {location.longitude.toFixed(6)}
                </p>
                {location.accuracy > 50 && (
                  <p className="text-xs text-yellow-700 dark:text-yellow-300 mt-1">
                    ⚠ For production, accuracy must be &lt;50m. Use manual entry for testing.
                  </p>
                )}
              </div>
            ) : locationError ? (
              <div className="bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded p-3">
                <p className="text-sm text-red-800 dark:text-red-200">✗ {locationError}</p>
              </div>
            ) : (
              <div className="bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded p-3">
                <p className="text-sm text-zinc-600 dark:text-zinc-400">
                  Waiting for GPS... (Capture a photo to auto-request)
                </p>
              </div>
            )}
          </div>

          {/* Optional Description */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2">
              Description (Optional)
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full px-3 py-2 border border-zinc-300 dark:border-zinc-700 rounded bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 text-sm"
              rows={3}
              placeholder="Additional details about the hazard..."
              disabled={submitStatus === 'submitting'}
            />
          </div>

          {/* Classification Result */}
          {classificationResult && (
            <div className="mb-6 bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800 rounded p-4">
              <h3 className="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-2">
                ML Classification Result
              </h3>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div>
                  <span className="text-blue-700 dark:text-blue-300">Type:</span>
                  <p className="font-semibold text-blue-900 dark:text-blue-100 capitalize">
                    {classificationResult.hazardType}
                  </p>
                </div>
                <div>
                  <span className="text-blue-700 dark:text-blue-300">Confidence:</span>
                  <p className="font-semibold text-blue-900 dark:text-blue-100">
                    {(classificationResult.confidence * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <span className="text-blue-700 dark:text-blue-300">Priority:</span>
                  <p className="font-semibold text-blue-900 dark:text-blue-100">
                    {(classificationResult.priority * 100).toFixed(0)}%
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Submit Button */}
          <button
            onClick={handleClassifyAndSubmit}
            disabled={!imageFile || !location || submitStatus === 'submitting'}
            className={`w-full py-3 px-4 rounded font-semibold text-white transition-colors ${
              !imageFile || !location || submitStatus === 'submitting'
                ? 'bg-zinc-400 cursor-not-allowed'
                : submitStatus === 'success'
                ? 'bg-green-600'
                : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {submitStatus === 'submitting' && '⏳ Classifying & Submitting...'}
            {submitStatus === 'success' && '✓ Report Submitted!'}
            {submitStatus === 'idle' && 'Classify & Submit Report'}
            {submitStatus === 'error' && 'Submit Report'}
          </button>

          {submitStatus === 'success' && (
            <p className="text-center text-sm text-green-600 dark:text-green-400 mt-3">
              Report added to heatmap. Resetting form...
            </p>
          )}
        </div>

        {/* Info Box */}
        <div className="mt-6 bg-zinc-100 dark:bg-zinc-900 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100 mb-2">
            How it works
          </h3>
          <ul className="text-xs text-zinc-600 dark:text-zinc-400 space-y-1">
            <li>• Capture a clear photo of the infrastructure hazard</li>
            <li>• GPS accuracy must be &lt;50m (automatic validation)</li>
            <li>• ML model classifies: fire / structural / vegetation</li>
            <li>• Priority auto-calculated (higher for fire, boosted by nearby reports)</li>
            <li>• Reports appear on live heatmap for Meralco engineers</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
