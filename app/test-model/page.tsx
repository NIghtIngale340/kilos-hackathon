'use client';

import { useState, useRef, useEffect } from 'react';
import { classifyImageWithAPI, checkAPIHealth } from '@/lib/apiClassifier';

export default function ModelTestPage() {
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [apiHealthy, setApiHealthy] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Check API health on mount
    checkAPIHealth().then(setApiHealthy);
  }, []);

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Create preview
    const reader = new FileReader();
    reader.onload = (evt) => {
      if (evt.target?.result) {
        setImagePreview(evt.target.result as string);
      }
    };
    reader.readAsDataURL(file);

    // Classify using API
    setIsClassifying(true);
    setResult(null);

    try {
      const classification = await classifyImageWithAPI(file);
      setResult(classification);
      
      // Set model info from API
      setModelInfo({
        architecture: 'Random Forest',
        accuracy: 0.857,
        version: '1.0-traditional-ml',
        trainingSamples: 112,
        classes: ['Moderate', 'Normal', 'Spagetti', 'Urgent']
      });
    } catch (error) {
      console.error('Classification error:', error);
      setResult({ error: String(error) });
    } finally {
      setIsClassifying(false);
    }
  };

  const getPriorityColor = (hazardType: string) => {
    switch (hazardType) {
      case 'urgent':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'moderate':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'normal':
        return 'text-green-600 bg-green-50 border-green-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.7) return 'text-green-600';
    if (confidence >= 0.5) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            🔌 Pole Hazard Model Test
          </h1>
          <p className="text-gray-600">
            Random Forest Classifier (85.7% accuracy) - Traditional ML approach
          </p>
          {apiHealthy ? (
            <p className="text-green-600 font-semibold mt-2">✅ API Connected</p>
          ) : (
            <p className="text-red-600 font-semibold mt-2">⚠️ API Offline - Make sure Flask server is running</p>
          )}
        </div>

        {/* Model Info Card */}
        {modelInfo && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6 border-2 border-blue-200">
            <h2 className="text-xl font-bold text-gray-900 mb-4">📊 Model Information</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600">Architecture</p>
                <p className="font-semibold">{modelInfo.architecture}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Accuracy</p>
                <p className="font-semibold text-green-600">{(modelInfo.accuracy * 100).toFixed(0)}%</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Version</p>
                <p className="font-semibold">{modelInfo.version}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Training Samples</p>
                <p className="font-semibold">{modelInfo.trainingSamples}</p>
              </div>
              <div className="col-span-2">
                <p className="text-sm text-gray-600">Classes</p>
                <p className="font-semibold">{modelInfo.classes.join(', ')}</p>
              </div>
            </div>
          </div>
        )}

        {/* Upload Area */}
        <div className="bg-white rounded-lg shadow-md p-8 mb-6">
          <div className="text-center">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded-lg transition-colors"
            >
              📸 Upload Pole Image
            </button>
            <p className="text-sm text-gray-500 mt-2">
              Upload an image of an electric pole to classify
            </p>
          </div>
        </div>

        {/* Image Preview */}
        {imagePreview && (
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4">Uploaded Image</h3>
            <img
              src={imagePreview}
              alt="Preview"
              className="w-full max-w-md mx-auto rounded-lg border-2 border-gray-200"
            />
          </div>
        )}

        {/* Loading State */}
        {isClassifying && (
          <div className="bg-white rounded-lg shadow-md p-8 text-center">
            <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Analyzing pole condition...</p>
          </div>
        )}

        {/* Results */}
        {result && !isClassifying && (
          <div className="bg-white rounded-lg shadow-md p-6">
            {result.error ? (
              <div className="text-center text-red-600">
                <p className="font-bold">Error:</p>
                <p>{result.error}</p>
              </div>
            ) : (
              <>
                <h3 className="text-2xl font-bold text-gray-900 mb-6 text-center">
                  Classification Results
                </h3>

                {/* Main Result */}
                <div className={`border-2 rounded-lg p-6 mb-6 ${getPriorityColor(result.hazardType)}`}>
                  <div className="text-center">
                    <p className="text-sm font-semibold uppercase mb-2">Detected Hazard Level</p>
                    <p className="text-5xl font-bold mb-4">
                      {result.hazardType.toUpperCase()}
                    </p>
                    <div className="flex items-center justify-center gap-2">
                      <p className="text-lg">Confidence:</p>
                      <p className={`text-2xl font-bold ${getConfidenceColor(result.confidence)}`}>
                        {(result.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>

                {/* All Probabilities */}
                <div className="space-y-3">
                  <h4 className="font-bold text-gray-700">All Class Probabilities:</h4>
                  
                  {/* Urgent */}
                  <div className="flex items-center gap-3">
                    <span className="w-24 font-semibold text-red-600">🔴 Urgent</span>
                    <div className="flex-1 bg-gray-200 rounded-full h-6 overflow-hidden">
                      <div
                        className="bg-red-500 h-full transition-all duration-500"
                        style={{ width: `${result.rawScores[0] * 100}%` }}
                      ></div>
                    </div>
                    <span className="w-16 text-right font-semibold">
                      {(result.rawScores[0] * 100).toFixed(1)}%
                    </span>
                  </div>

                  {/* Moderate */}
                  <div className="flex items-center gap-3">
                    <span className="w-24 font-semibold text-yellow-600">🟡 Moderate</span>
                    <div className="flex-1 bg-gray-200 rounded-full h-6 overflow-hidden">
                      <div
                        className="bg-yellow-500 h-full transition-all duration-500"
                        style={{ width: `${result.rawScores[1] * 100}%` }}
                      ></div>
                    </div>
                    <span className="w-16 text-right font-semibold">
                      {(result.rawScores[1] * 100).toFixed(1)}%
                    </span>
                  </div>

                  {/* Normal */}
                  <div className="flex items-center gap-3">
                    <span className="w-24 font-semibold text-green-600">🟢 Normal</span>
                    <div className="flex-1 bg-gray-200 rounded-full h-6 overflow-hidden">
                      <div
                        className="bg-green-500 h-full transition-all duration-500"
                        style={{ width: `${result.rawScores[2] * 100}%` }}
                      ></div>
                    </div>
                    <span className="w-16 text-right font-semibold">
                      {(result.rawScores[2] * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Interpretation */}
                <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
                  <p className="text-sm text-gray-700">
                    <strong>Interpretation:</strong>{' '}
                    {result.confidence >= 0.7
                      ? '✅ High confidence - Result is reliable'
                      : result.confidence >= 0.5
                      ? '⚠️ Moderate confidence - Consider manual verification'
                      : '❌ Low confidence - Manual verification required'}
                  </p>
                </div>
              </>
            )}
          </div>
        )}

        {/* Instructions */}
        {!result && !isClassifying && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mt-6">
            <h3 className="font-bold text-blue-900 mb-2">📝 How to Use:</h3>
            <ol className="list-decimal list-inside space-y-2 text-blue-800">
              <li>Click "Upload Pole Image" to select an image</li>
              <li>The model will automatically analyze it</li>
              <li>View the classification results with confidence scores</li>
              <li>Check all class probabilities to see how the model decided</li>
            </ol>
          </div>
        )}

        {/* Back Button */}
        <div className="text-center mt-8">
          <a
            href="/"
            className="inline-block bg-gray-600 hover:bg-gray-700 text-white font-bold py-2 px-6 rounded-lg transition-colors"
          >
            ← Back to Home
          </a>
        </div>
      </div>
    </div>
  );
}
