"""
Traditional ML Classifier for Pole Hazard Detection
Uses Random Forest with hand-crafted features - MUCH better for small datasets!
"""

import numpy as np
import cv2
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
from tqdm import tqdm

def extract_features(image_path):
    """Extract ENHANCED hand-crafted features from pole image"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Resize to standard size
    img = cv2.resize(img, (224, 224))
    
    # Convert to different color spaces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    features = []
    
    # 1. ENHANCED COLOR FEATURES (42 features)
    # RGB mean, std, and percentiles
    for channel in range(3):
        features.append(np.mean(img[:,:,channel]))
        features.append(np.std(img[:,:,channel]))
        features.append(np.percentile(img[:,:,channel], 25))
        features.append(np.percentile(img[:,:,channel], 75))
    
    # HSV mean, std, and percentiles
    for channel in range(3):
        features.append(np.mean(hsv[:,:,channel]))
        features.append(np.std(hsv[:,:,channel]))
        features.append(np.percentile(hsv[:,:,channel], 25))
        features.append(np.percentile(hsv[:,:,channel], 75))
    
    # LAB color space (better for perceptual differences)
    for channel in range(3):
        features.append(np.mean(lab[:,:,channel]))
        features.append(np.std(lab[:,:,channel]))
    
    # Enhanced color histograms (12 features)
    for channel in range(3):
        hist = cv2.calcHist([img], [channel], None, [4], [0, 256])
        features.extend(hist.flatten())
    
    # 2. TEXTURE FEATURES (9 features)
    # Grayscale statistics
    features.append(np.mean(gray))
    features.append(np.std(gray))
    features.append(np.min(gray))
    features.append(np.max(gray))
    features.append(np.median(gray))
    
    # Edge density (important for detecting tangled wires!)
    edges = cv2.Canny(gray, 50, 150)
    features.append(np.sum(edges) / edges.size)  # Edge ratio
    
    # Laplacian variance (blur detection - important for pole condition)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    features.append(laplacian_var)
    
    # Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    features.append(np.mean(np.abs(sobelx)))
    features.append(np.mean(np.abs(sobely)))
    
    # 3. SHAPE/STRUCTURE FEATURES (12 features - ENHANCED)
    # Contour analysis
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features.append(len(contours))  # Number of contours (high for spaghetti wires)
    
    if contours:
        # Largest contour area
        max_area = max([cv2.contourArea(c) for c in contours])
        features.append(max_area)
        
        # Average contour area
        avg_area = np.mean([cv2.contourArea(c) for c in contours])
        features.append(avg_area)
        
        # Contour aspect ratio (pole should be tall and narrow)
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        aspect_ratio = float(h) / (w + 1e-6)  # height / width
        features.append(aspect_ratio)
        
        # Vertical elongation (poles are vertical)
        features.append(1.0 if aspect_ratio > 2.0 else 0.0)
    else:
        features.extend([0, 0, 0, 0, 0])
    
    # Line detection (Hough transform) - important for pole alignment
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    if lines is not None:
        features.append(len(lines))  # Number of lines
        # Average line length
        lengths = [np.sqrt((x2-x1)**2 + (y2-y1)**2) for x1,y1,x2,y2 in lines[:,0]]
        features.append(np.mean(lengths))
        features.append(np.std(lengths))
        
        # Vertical line ratio (poles have many vertical lines)
        vertical_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            if angle > 75 or angle < 15:  # Near vertical
                vertical_lines += 1
        features.append(vertical_lines / (len(lines) + 1e-6))
    else:
        features.extend([0, 0, 0, 0])
    
    # Corner detection (important for broken/damaged poles)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    features.append(len(corners) if corners is not None else 0)
    
    # 4. ENHANCED BRIGHTNESS & COLOR REGIONS (12 features)
    # Bright spots (potential issues with transformer/equipment)
    bright_mask = gray > 200
    features.append(np.sum(bright_mask) / gray.size)
    
    # Dark spots
    dark_mask = gray < 50
    features.append(np.sum(dark_mask) / gray.size)
    
    # Mid-tone ratio
    mid_mask = (gray >= 100) & (gray <= 150)
    features.append(np.sum(mid_mask) / gray.size)
    
    # Vegetation detection (green dominance in HSV) - CRITICAL for moderate hazards
    green_mask = (hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 85) & (hsv[:,:,1] > 40)
    features.append(np.sum(green_mask) / gray.size)
    
    # Brown/rust detection (important for old poles) - CRITICAL for urgent hazards
    brown_mask = (hsv[:,:,0] >= 10) & (hsv[:,:,0] <= 20) & (hsv[:,:,1] > 30)
    features.append(np.sum(brown_mask) / gray.size)
    
    # Wire density (high saturation, low value regions) - CRITICAL for spaghetti wires
    wire_mask = (hsv[:,:,1] < 30) & (hsv[:,:,2] < 100)
    features.append(np.sum(wire_mask) / gray.size)
    
    # Spatial brightness distribution (6 more features for pole structure)
    h, w = gray.shape
    top_bright = np.mean(gray[:h//3, :])
    mid_bright = np.mean(gray[h//3:2*h//3, :])
    bot_bright = np.mean(gray[2*h//3:, :])
    features.extend([top_bright, mid_bright, bot_bright])
    
    left_bright = np.mean(gray[:, :w//2])
    right_bright = np.mean(gray[:, w//2:])
    center_bright = np.mean(gray[h//4:3*h//4, w//4:3*w//4])
    features.extend([left_bright, right_bright, center_bright])
    
    # 5. ADVANCED TEXTURE FEATURES (16 features - added ASM)
    # Local Binary Pattern (LBP) for texture
    radius = 3
    n_points = 8 * radius
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
    features.extend(lbp_hist[:5])  # Top 5 LBP bins
    
    # Gabor filters (detect oriented patterns - good for wire detection)
    ksize = 31
    sigma = 4.0
    theta_values = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    for theta in theta_values:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, 10.0, 0.5, 0)
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        features.append(np.mean(np.abs(filtered)))
    
    # GLCM texture features (co-occurrence matrix) - ENHANCED
    from skimage.feature import graycomatrix, graycoprops
    # Normalize to 0-255 range
    gray_norm = (gray / gray.max() * 255).astype(np.uint8) if gray.max() > 0 else gray
    glcm = graycomatrix(gray_norm, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    features.append(graycoprops(glcm, 'contrast')[0, 0])
    features.append(graycoprops(glcm, 'dissimilarity')[0, 0])
    features.append(graycoprops(glcm, 'homogeneity')[0, 0])
    features.append(graycoprops(glcm, 'energy')[0, 0])
    features.append(graycoprops(glcm, 'correlation')[0, 0])
    features.append(graycoprops(glcm, 'ASM')[0, 0])  # Angular Second Moment
    
    return np.array(features)


def load_dataset(dataset_path, split='train'):
    """Load images and extract features"""
    base_path = Path(dataset_path) / split
    
    classes = ['Moderate', 'Normal', 'Spagetti', 'Urgent']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    X = []
    y = []
    failed = 0
    
    print(f"\n📂 Loading {split} set...")
    for class_name in classes:
        class_path = base_path / class_name
        if not class_path.exists():
            print(f"⚠️  Warning: {class_path} not found")
            continue
        
        images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg')) + list(class_path.glob('*.png'))
        print(f"  {class_name}: {len(images)} images")
        
        for img_path in tqdm(images, desc=f"  Extracting features from {class_name}", leave=False):
            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(class_to_idx[class_name])
            else:
                failed += 1
    
    if failed > 0:
        print(f"  ⚠️  Failed to process {failed} images")
    
    return np.array(X), np.array(y), classes


def main():
    print("="*70)
    print("🌲 Traditional ML Classifier - Random Forest")
    print("="*70)
    print("✨ Perfect for small datasets!")
    print("✨ Uses hand-crafted features instead of deep learning")
    print("="*70)
    
    dataset_path = Path(__file__).parent.parent / 'pole-mock-dataset'
    
    # Load datasets
    X_train, y_train, classes = load_dataset(dataset_path, 'train')
    X_val, y_val, _ = load_dataset(dataset_path, 'valid')
    X_test, y_test, _ = load_dataset(dataset_path, 'test')
    
    print(f"\n📊 Dataset loaded:")
    print(f"  Training:   {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test:       {len(X_test)} samples")
    print(f"  Classes:    {classes}")
    
    # Normalize features
    print("\n🔧 Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train OPTIMIZED Random Forest with GridSearch
    print("\n🌲 Training OPTIMIZED Random Forest classifier...")
    print("   🔍 Using GridSearchCV to find best hyperparameters...")
    
    from sklearn.model_selection import GridSearchCV
    
    # Define parameter grid - ENHANCED for maximum accuracy
    param_grid = {
        'n_estimators': [500, 800, 1000],  # More trees for better ensemble
        'max_depth': [25, 30, 40],  # Deeper trees to capture complex patterns
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True],
        'class_weight': ['balanced', 'balanced_subsample'],
        'min_impurity_decrease': [0.0, 0.001]  # Prevent overfitting
    }
    
    # Base classifier
    base_clf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=0,
        oob_score=True  # Out-of-bag score for validation
    )
    
    # Grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        base_clf,
        param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\n✅ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    print(f"\n📊 Best cross-validation score: {grid_search.best_score_*100:.2f}%")
    
    # Use best estimator
    clf = grid_search.best_estimator_
    
    # Evaluate
    print("\n" + "="*70)
    print("📊 EVALUATION")
    print("="*70)
    
    # Training accuracy
    train_pred = clf.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"\n📈 Training Accuracy: {train_acc*100:.2f}%")
    
    # Validation accuracy
    val_pred = clf.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"📈 Validation Accuracy: {val_acc*100:.2f}%")
    
    print("\nValidation Classification Report:")
    print(classification_report(y_val, val_pred, target_names=classes))
    
    # Test accuracy
    test_pred = clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"📈 Test Accuracy: {test_acc*100:.2f}%")
    
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_pred, target_names=classes))
    
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, test_pred)
    print("           ", "  ".join([f"{cls:>8}" for cls in classes]))
    for i, row in enumerate(cm):
        print(f"{classes[i]:>10}", "  ".join([f"{val:>8}" for val in row]))
    
    # Feature importance
    print("\n🎯 Top 10 Most Important Features:")
    feature_names = [
        'R_mean', 'R_std', 'G_mean', 'G_std', 'B_mean', 'B_std',
        'H_mean', 'H_std', 'S_mean', 'S_std', 'V_mean', 'V_std',
        # ... (features would need full naming but this gives the idea)
    ]
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. Feature {idx}: {importances[idx]:.4f}")
    
    # Save model
    print("\n💾 Saving model...")
    output_dir = Path(__file__).parent / 'traditional_ml_model'
    output_dir.mkdir(exist_ok=True)
    
    joblib.dump(clf, output_dir / 'random_forest.joblib')
    joblib.dump(scaler, output_dir / 'scaler.joblib')
    
    # Save classes
    with open(output_dir / 'classes.json', 'w') as f:
        json.dump(classes, f, indent=2)
    
    # Save metadata
    metadata = {
        'model_type': 'RandomForest',
        'n_features': X_train.shape[1],
        'classes': classes,
        'accuracy': {
            'train': float(train_acc),
            'validation': float(val_acc),
            'test': float(test_acc)
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model saved to: {output_dir}")
    print(f"   - random_forest.joblib (classifier)")
    print(f"   - scaler.joblib (feature scaler)")
    print(f"   - classes.json (class names)")
    print(f"   - metadata.json (model info)")
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    
    if val_acc >= 0.75:
        print("✅ Model performance is good!")
    else:
        print("⚠️  Model performance could be better - consider collecting more data")
    
    print("\n📝 To use this model, you'll need to:")
    print("   1. Extract features from new images using extract_features()")
    print("   2. Scale features using scaler.transform()")
    print("   3. Predict using clf.predict()")
    print("   4. (Optional) Create a Python backend API for the web app")


if __name__ == '__main__':
    main()
