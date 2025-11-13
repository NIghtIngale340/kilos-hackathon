'use client';

import { useState } from 'react';
import * as tf from '@tensorflow/tfjs';

export default function DebugModelPage() {
  const [logs, setLogs] = useState<string[]>([]);
  const [image, setImage] = useState<string | null>(null);

  const addLog = (msg: string) => {
    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
  };

  const testPreprocessing = async (file: File) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    
    img.onload = async () => {
      setImage(url);
      addLog('🖼️ Image loaded');
      
      // Method 1: Current preprocessing (0 to 1)
      const tensor1 = tf.browser
        .fromPixels(img)
        .resizeBilinear([224, 224])
        .expandDims(0)
        .toFloat()
        .div(tf.scalar(255));
      
      const stats1 = {
        min: (await tensor1.min().data())[0],
        max: (await tensor1.max().data())[0],
        mean: (await tensor1.mean().data())[0]
      };
      addLog(`Method 1 (÷255): min=${stats1.min.toFixed(3)}, max=${stats1.max.toFixed(3)}, mean=${stats1.mean.toFixed(3)}`);
      
      // Method 2: MobileNetV3 preprocessing (-1 to 1)
      const tensor2 = tf.browser
        .fromPixels(img)
        .resizeBilinear([224, 224])
        .expandDims(0)
        .toFloat()
        .div(tf.scalar(127.5))
        .sub(tf.scalar(1));
      
      const stats2 = {
        min: (await tensor2.min().data())[0],
        max: (await tensor2.max().data())[0],
        mean: (await tensor2.mean().data())[0]
      };
      addLog(`Method 2 (MobileNetV3): min=${stats2.min.toFixed(3)}, max=${stats2.max.toFixed(3)}, mean=${stats2.mean.toFixed(3)}`);
      
      // Load model and test both
      addLog('📦 Loading model...');
      const model = await tf.loadGraphModel('/models/pole-hazard/model.json');
      addLog('✅ Model loaded');
      
      // Predict with method 1
      addLog('🔮 Predicting with Method 1 (÷255)...');
      const pred1 = model.predict(tensor1) as tf.Tensor;
      const scores1 = await pred1.data();
      addLog(`Method 1 results: [${Array.from(scores1).map(s => s.toFixed(4)).join(', ')}]`);
      addLog(`Method 1 class order: moderate=${scores1[0].toFixed(4)}, normal=${scores1[1].toFixed(4)}, urgent=${scores1[2].toFixed(4)}`);
      
      // Predict with method 2
      addLog('🔮 Predicting with Method 2 (MobileNetV3)...');
      const pred2 = model.predict(tensor2) as tf.Tensor;
      const scores2 = await pred2.data();
      addLog(`Method 2 results: [${Array.from(scores2).map(s => s.toFixed(4)).join(', ')}]`);
      addLog(`Method 2 class order: moderate=${scores2[0].toFixed(4)}, normal=${scores2[1].toFixed(4)}, urgent=${scores2[2].toFixed(4)}`);
      
      // Find predicted classes
      const idx1 = Array.from(scores1).indexOf(Math.max(...scores1));
      const idx2 = Array.from(scores2).indexOf(Math.max(...scores2));
      const classes = ['moderate', 'normal', 'urgent'];
      
      addLog(`⭐ Method 1 predicts: ${classes[idx1]} (${(scores1[idx1] * 100).toFixed(1)}%)`);
      addLog(`⭐ Method 2 predicts: ${classes[idx2]} (${(scores2[idx2] * 100).toFixed(1)}%)`);
      
      // Cleanup
      tensor1.dispose();
      tensor2.dispose();
      pred1.dispose();
      pred2.dispose();
    };
    
    img.src = url;
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">🔍 Model Preprocessing Debug</h1>
        
        <div className="bg-gray-800 rounded-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Upload Test Image</h2>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) {
                setLogs([]);
                testPreprocessing(file);
              }
            }}
            className="block w-full text-sm text-gray-300
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-600 file:text-white
              hover:file:bg-blue-700 cursor-pointer"
          />
        </div>

        {image && (
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Test Image</h2>
            <img src={image} alt="Test" className="max-w-md mx-auto rounded" />
          </div>
        )}

        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Debug Logs</h2>
          <div className="font-mono text-xs bg-black p-4 rounded h-96 overflow-y-auto">
            {logs.length === 0 ? (
              <p className="text-gray-500">Upload an image to see debug logs...</p>
            ) : (
              logs.map((log, idx) => (
                <div key={idx} className="mb-1">
                  {log}
                </div>
              ))
            )}
          </div>
        </div>

        <div className="mt-6 bg-blue-900 border border-blue-700 rounded-lg p-4">
          <h3 className="font-semibold mb-2">📝 Expected Behavior:</h3>
          <ul className="list-disc list-inside text-sm space-y-1">
            <li>Method 1 (÷255) normalizes to [0, 1] range</li>
            <li>Method 2 (MobileNetV3) normalizes to [-1, 1] range</li>
            <li>MobileNetV3 preprocessing: (pixel / 127.5) - 1</li>
            <li>The model expects MobileNetV3 preprocessing based on metadata</li>
            <li>If Method 1 always predicts "moderate", that's the bug!</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
