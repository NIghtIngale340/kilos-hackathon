#!/usr/bin/env python
"""
Direct conversion script without importing problematic modules
"""
import subprocess
import sys
import os

h5_model_path = 'pole_hazard_detector.h5'
output_dir = 'pole_tfjs_final'

print(f"Converting {h5_model_path} to TensorFlow.js format...")
print(f"Output directory: {output_dir}")
print()

# Use subprocess to call tensorflowjs_converter command
cmd = [
    sys.executable, '-m', 'tensorflowjs.converters.converter',
    '--input_format', 'keras',
    '--output_format', 'tfjs_graph_model',
    h5_model_path,
    output_dir
]

print(f"Running command: {' '.join(cmd)}")
print()

try:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Warnings/Info:")
        print(result.stderr)
    
    print(f"\n✅ Conversion complete! Files in {output_dir}/")
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / 1024  # Size in KB
                print(f"  - {file} ({size:.2f} KB)")
except subprocess.CalledProcessError as e:
    print("❌ Conversion failed!")
    print(f"Error: {e}")
    print(f"Output: {e.stdout}")
    print(f"Error output: {e.stderr}")
    sys.exit(1)
