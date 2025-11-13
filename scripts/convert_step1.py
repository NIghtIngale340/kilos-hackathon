import tensorflow as tf
from tensorflow import keras
import json
import os
import shutil
import numpy as np

print("Loading model...")
model = keras.models.load_model('pole_hazard_detector.h5')
model.summary()

# Save as SavedModel  
savedmodel_dir = 'temp_savedmodel'
if os.path.exists(savedmodel_dir):
    shutil.rmtree(savedmodel_dir)
    
print("\nSaving as SavedModel format...")
model.save(savedmodel_dir, save_format='tf')

print(f"\n✅ SavedModel created at {savedmodel_dir}/")
print("\nNow run this command:")
print(f"tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model {savedmodel_dir} pole_tfjs_final")
