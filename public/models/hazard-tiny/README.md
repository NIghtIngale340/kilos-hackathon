# TensorFlow.js Model Placeholder

This directory should contain your pre-trained TensorFlow.js hazard detection model.

## Required Files

Place the following files from your provided TF.js model in this directory:

1. `model.json` - Model manifest file
2. `group1-shard1of1.bin` (or similar weight files referenced in model.json)

## Model Requirements

- **Input:** 224x224 RGB image tensor (normalized to [0,1])
- **Output:** Array of 3 values representing confidence scores:
  - `[fire_risk, structural_risk, vegetation_risk]`

## Fallback Behavior

If the model files are not found, the classifier will use a fallback with:
- Default hazard type: `structural`
- Default confidence: `0.4` (below 0.7 threshold)
- This will require manual verification (2+ reports)

## Testing Without Model

You can test the app without a model - it will:
1. Show a warning in the browser console
2. Use the fallback classification
3. Still calculate priorities and accept reports
4. Reports with confidence < 0.4 will be rejected (as per spec)

## Model Training Notes

For reference, the model should be trained on:
- Meralco infrastructure hazard images
- 3-class classification (fire, structural, vegetation)
- MobileNetV3 or similar lightweight architecture
- Converted to TensorFlow.js format using `tensorflowjs_converter`
