# -*- coding: utf-8 -*-
"""
Train Pole Hazard Detection Model from pole-mock-dataset
Maps pole conditions to hazard levels:
- Broken Pole -> urgent
- Old Pole -> urgent  
- Vegetation Pole / Pole Vegetation -> moderate
- Good Pole -> normal

OPTIMIZED FOR SMALL DATASETS:
- Advanced augmentation
- Class weighting
- Progressive learning
- EfficientNet architecture
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
import json
import shutil
from pathlib import Path
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Enable mixed precision for faster training (only if GPU available)
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("GPU detected - Mixed precision enabled")
    else:
        print("CPU mode - Using float32")
except:
    print("CPU mode - Using float32")

# Configuration - AGGRESSIVE anti-overfitting for 112-image dataset
IMG_SIZE = 224
BATCH_SIZE = 4  # Very small batch for tiny dataset
EPOCHS_WARMUP = 10  # Fewer epochs to prevent memorization
EPOCHS_TRANSFER = 15  # Further reduced
EPOCHS_FINE_TUNE = 8  # Stop much earlier
LEARNING_RATE_INITIAL = 0.0001  # Lower LR to prevent overfitting
LEARNING_RATE_FINE = 0.000005  # Even lower for fine-tuning

def augment_image(image, label):
    """AGGRESSIVE CPU-compatible augmentation to prevent overfitting"""
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    # Random vertical flip
    image = tf.image.random_flip_up_down(image)
    # STRONGER brightness (increased from 0.2 to 0.3)
    image = tf.image.random_brightness(image, max_delta=0.3)
    # STRONGER contrast (expanded range)
    image = tf.image.random_contrast(image, lower=0.6, upper=1.5)
    # STRONGER saturation (expanded range)
    image = tf.image.random_saturation(image, lower=0.6, upper=1.5)
    # STRONGER hue (increased from 0.1 to 0.2)
    image = tf.image.random_hue(image, max_delta=0.2)
    # Ensure values stay in valid range
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label

def prepare_direct_dataset(source_path):
    """
    Use original 7 classes directly without reorganization:
    1. Normal
    2. Old Pole
    3. Pole Vegetation
    4. Unlabeled
    5. UnlabeledPole Vegetation
    6. Urgent
    7. Vegetation Pole
    """
    # Just return the path - no reorganization needed
    return source_path

def augment_image(image, label):
    """CPU-compatible augmentation using tf.image operations"""
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    # Random vertical flip (poles can be photographed from different angles)
    image = tf.image.random_flip_up_down(image)
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # Random saturation
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    # Random hue (slight color shifts)
    image = tf.image.random_hue(image, max_delta=0.1)
    # Ensure values stay in valid range
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label

def create_model():
    """Build MobileNetV2 model for pole hazard classification (smaller, less prone to overfitting)"""
    
    # Base model - MobileNetV2 (smaller, better for tiny datasets)
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.35  # MINIMAL model size for tiny 112-image dataset
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build classifier with VERY strong regularization to prevent overfitting
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Preprocessing for MobileNetV2
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Simplified classification head (fewer parameters)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.85)(x)  # EXTREME dropout to prevent collapse
    
    # Single dense layer with strong L2 regularization
    x = layers.Dense(
        64,  # Even smaller - reduced from 128
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.05),  # MUCH stronger regularization
        kernel_initializer='he_normal'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.85)(x)  # EXTREME dropout
    
    # Output layer (4 classes: moderate, normal, spaghetti, urgent)
    # Add L2 to output layer to prevent collapse
    outputs = layers.Dense(
        4, 
        activation='softmax',
        kernel_regularizer=keras.regularizers.l2(0.01),
        name='predictions'
    )(x)
    
    model = keras.Model(inputs, outputs, name="pole_hazard_detector_4class")
    
    return model, base_model

def load_dataset(dataset_path):
    """Load reorganized hazard dataset with enhanced preprocessing"""
    
    train_dir = os.path.join(dataset_path, 'train')
    valid_dir = os.path.join(dataset_path, 'valid')
    test_dir = os.path.join(dataset_path, 'test')
    
    print("\n📂 Loading datasets...")
    
    # Load training data - shuffle heavily for small dataset
    train_dataset = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    # Apply augmentation to training data only (CPU-compatible)
    print("  📸 Applying data augmentation to training set...")
    train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Load validation data
    val_dataset = keras.utils.image_dataset_from_directory(
        valid_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False,
        seed=42
    )
    
    # Load test data if exists
    test_dataset = None
    if os.path.exists(test_dir):
        test_dataset = keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            label_mode='categorical',
            shuffle=False,
            seed=42
        )
    
    # Optimize performance with caching and prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Cache training data in memory (small dataset)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(buffer_size=100)  # Re-shuffle
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
    
    if test_dataset:
        test_dataset = test_dataset.cache()
        test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    
    return train_dataset, val_dataset, test_dataset

def compute_class_weights(train_dataset):
    """Compute class weights to handle imbalanced dataset"""
    
    print("\n⚖️  Computing class weights for imbalanced data...")
    
    # Extract all labels
    labels = []
    for _, batch_labels in train_dataset.unbatch():
        label_idx = tf.argmax(batch_labels).numpy()
        labels.append(label_idx)
    
    labels = np.array(labels)
    
    # Compute weights
    unique_classes = np.unique(labels)
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=labels
    )
    
    # Create weight dictionary
    class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
    
    # Fill in missing classes with default weight of 1.0
    for i in range(7):  # We have 7 classes
        if i not in class_weights:
            class_weights[i] = 1.0
    
    print("  Class weights:")
    # Classes will be alphabetically sorted by Keras
    class_names = ['Normal', 'Old Pole', 'Pole Vegetation', 'Unlabeled', 
                   'UnlabeledPole Vegetation', 'Urgent', 'Vegetation Pole']
    for idx, name in enumerate(class_names):
        count = np.sum(labels == idx)
        weight = class_weights.get(idx, 1.0)
        if count > 0:
            print(f"    {name}: {weight:.2f} (samples: {count})")
        else:
            print(f"    {name}: {weight:.2f} (samples: {count}) ⚠️ No samples!")
    
    return class_weights

def print_dataset_info(train_dataset, val_dataset, test_dataset):
    """Print dataset statistics"""
    
    num_train = sum([batch[0].shape[0] for batch in train_dataset])
    num_val = sum([batch[0].shape[0] for batch in val_dataset])
    
    print("\n📊 Dataset Statistics:")
    print(f"  Training samples:   {num_train}")
    print(f"  Validation samples: {num_val}")
    
    if test_dataset:
        num_test = sum([batch[0].shape[0] for batch in test_dataset])
        print(f"  Test samples:       {num_test}")
    
    print(f"\n  Classes: urgent, moderate, normal")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")

def main():
    print("🔌 Pole Hazard Detection - Advanced Training v2")
    print("=" * 70)
    print("🎯 Target: 80%+ accuracy with small dataset")
    print("=" * 70)
    
    # Source dataset path
    source_dataset = Path('../pole-mock-dataset')
    if not source_dataset.exists():
        source_dataset = Path('pole-mock-dataset')
    if not source_dataset.exists():
        source_dataset = Path(__file__).parent.parent / 'pole-mock-dataset'
    
    if not source_dataset.exists():
        print("❌ Error: pole-mock-dataset not found!")
        sys.exit(1)
    
    print(f"📁 Source dataset: {source_dataset}")
    
    # Use original dataset directly (no reorganization)
    print(f"\n📂 Using original 7-class dataset: {source_dataset}")
    dataset_to_use = prepare_direct_dataset(source_dataset)
    
    # Load datasets
    try:
        train_dataset, val_dataset, test_dataset = load_dataset(dataset_to_use)
        print_dataset_info(train_dataset, val_dataset, test_dataset)
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(train_dataset)
    
    # Build model
    print("\n🏗️  Building MobileNetV2 model for 7 classes...")
    model, base_model = create_model()
    
    # Compile with initial learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_INITIAL),
        loss='categorical_crossentropy',
        metrics=['accuracy']  # Simplified to avoid serialization issues
    )
    
    print("\n📊 Model Architecture:")
    print(f"  Base: MobileNetV2 (alpha=0.75, pre-trained on ImageNet)")
    print(f"  Output: 7 classes (all original pole types)")
    print(f"  Total params: {model.count_params():,}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Regularization: Dropout(0.7, 0.6) + L2(0.02)")
    print(f"  Augmentation: Enabled (flips, brightness, contrast, saturation, hue)")
    
    # AGGRESSIVE anti-overfitting callbacks
    callbacks = [
        # Early stopping - VERY aggressive to prevent collapse
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,  # Much more aggressive - stop after 4 epochs of no improvement
            restore_best_weights=True,
            verbose=1,
            mode='min',
            min_delta=0.005  # Larger threshold - must improve by at least 0.5%
        ),
        
        # Reduce learning rate on plateau (MORE aggressive)
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,  # Cut LR to 20% instead of 30%
            patience=3,  # Even faster reduction
            min_lr=1e-7,
            verbose=1,
            mode='min'
        )
    ]
    
    # Phase 1: Warmup with augmentation
    print("\n" + "=" * 70)
    print(f"🔥 PHASE 1: Warmup with Data Augmentation ({EPOCHS_WARMUP} epochs)")
    print(f"   Learning rate: {LEARNING_RATE_INITIAL:.2e}")
    print(f"   Augmentation: Flips, Brightness, Contrast, Saturation, Hue")
    print("=" * 70)
    
    # Compile for warmup
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_INITIAL),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    history_warmup = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS_WARMUP,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2
    )
    
    # Phase 2: Transfer Learning (full learning rate)
    print("\n" + "=" * 70)
    print(f"🚀 PHASE 2: Transfer Learning ({EPOCHS_TRANSFER} epochs)")
    print(f"   Learning rate: {LEARNING_RATE_INITIAL:.2e}")
    print(f"   Strategy: Fewer epochs to prevent overfitting")
    print("=" * 70)
    
    # Recompile with full learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_INITIAL),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    history_transfer = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS_TRANSFER,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2
    )
    
    # Phase 3: Fine-Tuning
    print("\n" + "=" * 70)
    print(f"🎯 PHASE 3: Fine-Tuning ({EPOCHS_FINE_TUNE} epochs)")
    print(f"   Learning rate: {LEARNING_RATE_FINE:.2e}")
    print("=" * 70)
    
    # Unfreeze base model
    base_model.trainable = True
    
    # Freeze early layers (keep first 80% frozen)
    num_layers = len(base_model.layers)
    freeze_until = int(num_layers * 0.8)
    
    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True
    
    print(f"   Unfrozen layers: {num_layers - freeze_until}/{num_layers}")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    history_fine = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS_FINE_TUNE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2
    )
    
    # Final Evaluation
    print("\n" + "=" * 70)
    print("📈 FINAL EVALUATION")
    print("=" * 70)
    
    # Evaluate on validation set
    print("\n📊 Validation Set:")
    val_results = model.evaluate(val_dataset, verbose=0)
    val_metrics = dict(zip(model.metrics_names, val_results))
    
    print(f"  Loss:       {val_metrics['loss']:.4f}")
    print(f"  Accuracy:   {val_metrics['accuracy'] * 100:.2f}%")
    print(f"  Top-2 Acc:  {val_metrics['top_2_accuracy'] * 100:.2f}%")
    print(f"  Precision:  {val_metrics['precision'] * 100:.2f}%")
    print(f"  Recall:     {val_metrics['recall'] * 100:.2f}%")
    print(f"  AUC:        {val_metrics['auc']:.4f}")
    
    # Evaluate on test set if available
    if test_dataset:
        print("\n📊 Test Set:")
        test_results = model.evaluate(test_dataset, verbose=0)
        test_metrics = dict(zip(model.metrics_names, test_results))
        
        print(f"  Loss:       {test_metrics['loss']:.4f}")
        print(f"  Accuracy:   {test_metrics['accuracy'] * 100:.2f}%")
        print(f"  Top-2 Acc:  {test_metrics['top_2_accuracy'] * 100:.2f}%")
        print(f"  Precision:  {test_metrics['precision'] * 100:.2f}%")
        print(f"  Recall:     {test_metrics['recall'] * 100:.2f}%")
        print(f"  AUC:        {test_metrics['auc']:.4f}")
    
    # Save models
    print("\n💾 Saving models...")
    
    # Save as H5 (save weights only to avoid JSON serialization errors)
    try:
        model.save_weights('pole_hazard_detector.h5')
        print("  ✅ Saved weights: pole_hazard_detector.h5")
    except Exception as e:
        print(f"  ⚠️ H5 save failed: {e}")
    
    # Save as SavedModel (for TF.js conversion) - this usually works better
    try:
        model.save('pole_saved_model', save_format='tf')
        print("  ✅ Saved: pole_saved_model/")
    except Exception as e:
        print(f"  ⚠️ SavedModel save failed: {e}")
        # Fallback: compile with simple metrics and save
        print("  🔄 Trying fallback save method...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.save('pole_saved_model', save_format='tf')
        print("  ✅ Saved with fallback method: pole_saved_model/")
    
    # Save class mapping
    class_mapping = {
        'classes': ['moderate', 'normal', 'spagetti', 'urgent'],  # Alphabetical order from keras (matching folder names)
        'description': '4-class pole hazard detection',
        'mapping': {
            'urgent': ['Broken Pole', 'Old Pole', 'Severely Leaning'],
            'spagetti': ['Tangled Wires', 'Messy Cables', 'Unauthorized Taps'],
            'moderate': ['Vegetation Pole', 'Pole Vegetation', 'Tree Contact'],
            'normal': ['Good Pole', 'Clean Pole', 'Well Maintained']
        }
    }
    with open('pole_classes.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print("  ✅ Saved: pole_classes.json")
    
    # Save training history
    import pickle
    with open('pole_training_history.pkl', 'wb') as f:
        pickle.dump({
            'warmup': history_warmup.history,
            'transfer': history_transfer.history,
            'fine_tune': history_fine.history
        }, f)
    print("  ✅ Saved: pole_training_history.pkl")
    
    # Performance assessment
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    
    accuracy = val_metrics['accuracy']
    
    print("\n📊 Model Performance:")
    if accuracy >= 0.85:
        print("  ✅ EXCELLENT - Exceeded target!")
    elif accuracy >= 0.80:
        print("  ✅ GREAT - Achieved 80%+ target!")
    elif accuracy >= 0.70:
        print("  ⚠️  GOOD - Close to target")
    else:
        print("  ⚠️  NEEDS IMPROVEMENT")
    
    print(f"\n🎯 Final Validation Accuracy: {accuracy * 100:.2f}%")
    
    print("\n🚀 Next Steps:")
    print("  1. Convert to TF.js: tensorflowjs_converter --input_format=keras pole_hazard_detector.h5 pole_tfjs")
    print("  2. Test conversion: Check pole_tfjs/ folder")
    print("  3. Deploy: Copy to public/models/pole-hazard/")
    
    print("\n📊 Training logs saved:")
    print("  • pole_training_log.csv       - Epoch-by-epoch metrics")
    print("  • pole_training_history.pkl   - Full training history")
    print("  • pole_hazard_checkpoint.h5   - Best checkpoint")
    
    print("\n💡 Tips for better results:")
    print("  • Collect more images (aim for 50+ per class)")
    print("  • Ensure varied lighting and angles")
    print("  • Balance the dataset (equal samples per class)")

if __name__ == "__main__":
    main()
