/*
Client-side hazard classifier using TensorFlow.js.
This module dynamically imports `@tensorflow/tfjs` at runtime to avoid bundling TF into the initial app shell.

Usage:
  import { loadModel, classifyImageElement, classifyFile } from '@/lib/HazardClassifier';
  await loadModel();
  const result = await classifyImageElement(myImgEl);

Model file expected at: /public/models/hazard-tiny/model.json
*/

let tf: typeof import('@tensorflow/tfjs') | null = null;
let model: import('@tensorflow/tfjs').GraphModel | null = null;

export async function ensureTf() {
  if (tf) return tf;
  // dynamic import keeps TF out of the initial bundle
  const mod = await import('@tensorflow/tfjs');
  tf = mod;
  return tf;
}

export async function loadModel(modelPath = '/models/hazard-tiny/model.json') {
  if (model) return model;
  const _tf = await ensureTf();
  model = await _tf.loadGraphModel(modelPath);
  return model;
}

function preprocessImage(_tf: typeof import('@tensorflow/tfjs'), image: HTMLImageElement) {
  // Convert image to tensor, resize to 224x224, normalize to [0,1]
  return _tf.browser
    .fromPixels(image)
    .resizeBilinear([224, 224])
    .expandDims(0)
    .toFloat()
    .div(_tf.scalar(255));
}

export async function classifyImageElement(image: HTMLImageElement) {
  const _tf = await ensureTf();
  const mdl = await loadModel();
  try {
    const tensor = preprocessImage(_tf, image);
    const out = mdl.predict(tensor) as import('@tensorflow/tfjs').Tensor;
    // Return raw array (assumes small output like [fire, structural, vegetation])
    const arr = Array.from(await out.data());
    return arr;
  } catch (err) {
    console.error('HazardClassifier classify error', err);
    throw err;
  }
}

export async function classifyFile(file: File) {
  // Create image element from file and classify
  return new Promise<number[]>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = (e) => reject(e);
    reader.onload = () => {
      const img = new Image();
      img.onload = async () => {
        try {
          const r = await classifyImageElement(img);
          resolve(r);
        } catch (err) {
          reject(err);
        }
      };
      img.onerror = (e) => reject(e);
      if (typeof reader.result === 'string') img.src = reader.result;
      else reject(new Error('Failed to read file as data URL'));
    };
    reader.readAsDataURL(file);
  });
}

export function isModelLoaded() {
  return !!model;
}
