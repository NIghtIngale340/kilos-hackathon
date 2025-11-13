import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs
import os
import shutil

# Load the H5 model
print("Loading model...")
model = keras.models.load_model('pole_hazard_detector.h5')
model.summary()

# Create output directory
output_dir = 'pole_tfjs_final'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Convert using tfjs directly
print("\nConverting to TensorFlow.js...")
tfjs.converters.save_keras_model(model, output_dir)

print(f"\n✅ Conversion complete! Files in {output_dir}/")
for file in os.listdir(output_dir):
    print(f"  - {file}")
