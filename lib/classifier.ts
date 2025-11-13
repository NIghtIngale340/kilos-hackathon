// ML classifier helpers with fallback logic
// Updated to use the newly trained EfficientNetB0 model (80% accuracy!)

import { MLClassificationResult, HazardType } from './types';

let tf: typeof import('@tensorflow/tfjs') | null = null;
let model: import('@tensorflow/tfjs').GraphModel | import('@tensorflow/tfjs').LayersModel | null = null;

export async function ensureTf() {
  if (tf) return tf;
  const mod = await import('@tensorflow/tfjs');
  tf = mod;
  return tf;
}

export async function loadModel(modelPath = '/models/pole-hazard/model.json') {
  if (model) return model;
  const _tf = await ensureTf();
  
  try {
    // Try loading as GraphModel first (TensorFlow.js format - preferred)
    console.log('🔄 Loading pole hazard detection model...');
    model = await _tf.loadGraphModel(modelPath);
    console.log('✅ Pole hazard detection model loaded successfully (GraphModel)');
    console.log('📊 Model: EfficientNetB0 | Accuracy: 80% | Classes: urgent, moderate, normal');
    return model;
  } catch (err) {
    console.warn('⚠️ GraphModel failed, trying LayersModel...', err);
    
    try {
      // Fallback to LayersModel (Keras format)
      model = await _tf.loadLayersModel(modelPath);
      console.log('✅ Pole hazard detection model loaded successfully (LayersModel)');
      return model;
    } catch (err2) {
      console.error('❌ Failed to load pole hazard model in any format', err2);
      console.warn('⚠️ Using fallback classification');
      return null;
    }
  }
}

function preprocessImage(_tf: typeof import('@tensorflow/tfjs'), image: HTMLImageElement) {
  // MobileNetV2 preprocessing: resize to 224x224 and normalize to [-1, 1]
  // This matches the training preprocessing: keras.applications.mobilenet_v2.preprocess_input
  // Formula: (pixel / 127.5) - 1.0
  return _tf.browser
    .fromPixels(image)
    .resizeBilinear([224, 224])
    .expandDims(0)
    .toFloat()
    .div(_tf.scalar(127.5))
    .sub(_tf.scalar(1.0));
}

function interpretMLOutput(rawScores: number[]): MLClassificationResult {
  // Model outputs 4 classes in alphabetical order: [moderate, normal, spagetti, urgent]
  // Map to our HazardType enum: ['urgent', 'moderate', 'normal']
  // Note: "spagetti" (spaghetti wire) maps to "urgent" for now
  
  const types: HazardType[] = ['urgent', 'moderate', 'normal'];
  
  if (rawScores.length < 4) {
    // Fallback if model output is unexpected
    console.warn('⚠️ Unexpected model output length:', rawScores.length, 'expected 4');
    return {
      hazardType: 'moderate',
      confidence: 0.5,
      rawScores: [0.2, 0.5, 0.3]
    };
  }

  // Model outputs: [moderate, normal, spagetti, urgent] (alphabetical)
  // Combine spagetti + urgent into urgent category
  // Reorder to: [urgent, moderate, normal] for our app
  const urgentScore = Math.max(rawScores[2], rawScores[3]); // spagetti or urgent
  const reorderedScores = [
    urgentScore,   // urgent (includes spagetti wire hazards)
    rawScores[0],  // moderate
    rawScores[1]   // normal
  ];
  
  // Find the class with highest probability
  let maxIdx = 0;
  let maxVal = reorderedScores[0];
  for (let i = 1; i < 3; i++) {
    if (reorderedScores[i] > maxVal) {
      maxVal = reorderedScores[i];
      maxIdx = i;
    }
  }

  const result = {
    hazardType: types[maxIdx],
    confidence: maxVal,
    rawScores: reorderedScores
  };

  console.log('🎯 Classification result:', {
    predicted: result.hazardType,
    confidence: `${(result.confidence * 100).toFixed(1)}%`,
    modelOutput: {
      moderate: `${(rawScores[0] * 100).toFixed(1)}%`,
      normal: `${(rawScores[1] * 100).toFixed(1)}%`,
      spagetti: `${(rawScores[2] * 100).toFixed(1)}%`,
      urgent: `${(rawScores[3] * 100).toFixed(1)}%`
    },
    combinedProbabilities: {
      urgent: `${(reorderedScores[0] * 100).toFixed(1)}%`,
      moderate: `${(reorderedScores[1] * 100).toFixed(1)}%`,
      normal: `${(reorderedScores[2] * 100).toFixed(1)}%`
    }
  });

  return result;
}


export async function classifyImageElement(image: HTMLImageElement): Promise<MLClassificationResult> {
  try {
    const _tf = await ensureTf();
    const mdl = await loadModel();
    
    if (!mdl) {
      // Model failed to load, use fallback
      console.warn('⚠️ Model not available, using moderate priority fallback');
      return {
        hazardType: 'moderate',
        confidence: 0.4, // Below 0.7 threshold, will require verification
        rawScores: [0.2, 0.4, 0.4]
      };
    }

    const tensor = preprocessImage(_tf, image);
    const out = mdl.predict(tensor) as import('@tensorflow/tfjs').Tensor;
    const arr = Array.from(await out.data());
    
    // Clean up tensors to prevent memory leaks
    tensor.dispose();
    out.dispose();
    
    const result = interpretMLOutput(arr);
    
    return result;
  } catch (err) {
    console.error('❌ Classification error:', err);
    // Fallback on error
    return {
      hazardType: 'moderate',
      confidence: 0.4,
      rawScores: [0.2, 0.4, 0.4]
    };
  }
}

export async function classifyFile(file: File): Promise<MLClassificationResult> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = (e) => reject(e);
    reader.onload = () => {
      const img = new Image();
      img.onload = async () => {
        try {
          const result = await classifyImageElement(img);
          resolve(result);
        } catch (err) {
          reject(err);
        }
      };
      img.onerror = (e) => reject(e);
      if (typeof reader.result === 'string') {
        img.src = reader.result;
      } else {
        reject(new Error('Failed to read file as data URL'));
      }
    };
    reader.readAsDataURL(file);
  });
}

export function isModelLoaded() {
  return !!model;
}

export async function getModelInfo() {
  const _tf = await ensureTf();
  const mdl = await loadModel();
  
  if (!mdl) {
    return null;
  }
  
  return {
    modelName: 'Pole Hazard Detector',
    architecture: 'EfficientNetB0 (Custom Trained)',
    accuracy: 0.80, // 80% validation accuracy
    classes: ['urgent', 'moderate', 'normal'],
    classDetails: {
      urgent: 'Broken/Old poles requiring immediate attention',
      moderate: 'Poles with vegetation requiring maintenance',
      normal: 'Poles in good condition'
    },
    inputSize: [224, 224, 3],
    version: '2.0',
    trainingDate: '2025-11-13',
    trainingSamples: 48,
    modelFormat: 'TensorFlow.js GraphModel'
  };
}
