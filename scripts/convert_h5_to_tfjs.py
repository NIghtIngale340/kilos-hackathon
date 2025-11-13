import tensorflow as tf
from tensorflow import keras
import json
import os
import shutil

print("Loading model...")
model = keras.models.load_model('pole_hazard_detector.h5')
model.summary()

# Step 1: Save as SavedModel format
saved_model_dir = 'temp_model'
if os.path.exists(saved_model_dir):
    shutil.rmtree(saved_model_dir)

print("\nSaving as SavedModel format...")
model.save(saved_model_dir)

# Step 2: Use tensorflowjs_converter command line tool instead
output_dir = 'pole_tfjs_final'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

print("\n✅ Model saved in SavedModel format!")
print(f"Now run this command to convert to TensorFlow.js:")
print(f"\ntensorflowjs_converter --input_format=keras 'pole_hazard_detector.h5' '{output_dir}'")
print("\nOr run:")
print(f"tensorflowjs_converter --input_format=tf_saved_model '{saved_model_dir}' '{output_dir}'")
