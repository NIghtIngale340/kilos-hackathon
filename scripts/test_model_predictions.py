import tensorflow as tf
import numpy as np
import os
from pathlib import Path

# Load the model
model_path = Path(__file__).parent / 'pole_saved_model'
print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)

print(f"\nModel input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# Test on a few images from each class
test_dir = Path(__file__).parent.parent / 'pole-mock-dataset' / 'test'
classes = ['Moderate', 'Normal', 'Spagetti', 'Urgent']

print("\n" + "="*60)
print("TESTING MODEL PREDICTIONS")
print("="*60)

for class_name in classes:
    class_dir = test_dir / class_name
    if not class_dir.exists():
        print(f"\n⚠️ Directory not found: {class_dir}")
        continue
        
    images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
    
    if not images:
        print(f"\n⚠️ No images found in {class_name}")
        continue
    
    print(f"\n{'='*60}")
    print(f"CLASS: {class_name} ({len(images)} images)")
    print(f"{'='*60}")
    
    correct = 0
    
    for i, img_path in enumerate(images[:5]):  # Test first 5 images
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            img_path, 
            target_size=(224, 224)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        # MobileNetV2 preprocessing
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get predicted class
        predicted_idx = np.argmax(predictions)
        predicted_class = classes[predicted_idx]
        confidence = predictions[predicted_idx]
        
        is_correct = predicted_class == class_name
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        
        print(f"\n{status} Image {i+1}: {img_path.name}")
        print(f"  Predicted: {predicted_class} ({confidence*100:.1f}%)")
        print(f"  Raw scores:")
        for j, (cls, score) in enumerate(zip(classes, predictions)):
            marker = "👉" if j == predicted_idx else "  "
            print(f"    {marker} {cls}: {score*100:.1f}%")
    
    accuracy = (correct / min(5, len(images))) * 100
    print(f"\n📊 Accuracy for {class_name}: {accuracy:.1f}% ({correct}/{min(5, len(images))})")

print("\n" + "="*60)
print("TESTING COMPLETE")
print("="*60)
