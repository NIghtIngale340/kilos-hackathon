"""
Flask API for Pole Hazard Detection using Random Forest
Provides a REST API endpoint for the Next.js frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import joblib
import json
from pathlib import Path
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Load model at startup
MODEL_DIR = Path(__file__).parent / 'traditional_ml_model'
print("🚀 Loading Random Forest model...")
classifier = joblib.load(MODEL_DIR / 'random_forest.joblib')
scaler = joblib.load(MODEL_DIR / 'scaler.joblib')
with open(MODEL_DIR / 'classes.json', 'r') as f:
    classes = json.load(f)
with open(MODEL_DIR / 'metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"✅ Model loaded successfully!")
print(f"   Classes: {classes}")
print(f"   Test Accuracy: {metadata['accuracy']['test']*100:.1f}%")


def validate_pole_image(img):
    """
    Validate if the image contains a pole/electric infrastructure.
    Returns (is_pole: bool, pole_score: float)
    """
    img_resized = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    score = 0.0
    max_score = 6.0
    
    # 1. Edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size
    
    if 0.05 < edge_ratio < 0.30:
        score += 2.0
    elif edge_ratio < 0.03:
        score -= 1.0
    
    # 2. Vertical line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=30, maxLineGap=10)
    
    if lines is not None:
        vertical_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            if angle > 75 or angle < 15:
                vertical_count += 1
        
        vertical_ratio = vertical_count / (len(lines) + 1e-6)
        if vertical_ratio > 0.4:
            score += 2.0
        elif vertical_ratio > 0.2:
            score += 1.0
    
    # 3. Color histogram - poles are typically gray/brown
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    saturation_mean = np.mean(hsv[:, :, 1])
    
    if saturation_mean < 50:
        score += 1.0
    elif saturation_mean > 100:
        score -= 0.5
    
    # 4. Aspect ratio
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        aspect_ratio = float(h) / (w + 1e-6)
        
        if aspect_ratio > 1.5:
            score += 1.0
    
    pole_score = min(1.0, max(0.0, score / max_score))
    is_pole = pole_score >= 0.35
    
    return is_pole, pole_score


def extract_features(img):
    """Extract ENHANCED hand-crafted features from image (MUST match training!)"""
    # Resize to standard size
    img = cv2.resize(img, (224, 224))
    
    # Convert to different color spaces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    features = []
    
    # 1. ENHANCED COLOR FEATURES (42 features)
    for channel in range(3):
        features.append(np.mean(img[:,:,channel]))
        features.append(np.std(img[:,:,channel]))
        features.append(np.percentile(img[:,:,channel], 25))
        features.append(np.percentile(img[:,:,channel], 75))
    
    for channel in range(3):
        features.append(np.mean(hsv[:,:,channel]))
        features.append(np.std(hsv[:,:,channel]))
        features.append(np.percentile(hsv[:,:,channel], 25))
        features.append(np.percentile(hsv[:,:,channel], 75))
    
    for channel in range(3):
        features.append(np.mean(lab[:,:,channel]))
        features.append(np.std(lab[:,:,channel]))
    
    for channel in range(3):
        hist = cv2.calcHist([img], [channel], None, [4], [0, 256])
        features.extend(hist.flatten())
    
    # 2. TEXTURE FEATURES (9 features)
    features.append(np.mean(gray))
    features.append(np.std(gray))
    features.append(np.min(gray))
    features.append(np.max(gray))
    features.append(np.median(gray))
    
    edges = cv2.Canny(gray, 50, 150)
    features.append(np.sum(edges) / edges.size)
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    features.append(laplacian_var)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    features.append(np.mean(np.abs(sobelx)))
    features.append(np.mean(np.abs(sobely)))
    
    # 3. SHAPE/STRUCTURE FEATURES (13 features - ENHANCED)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    features.append(len(contours))
    
    if contours:
        max_area = max([cv2.contourArea(c) for c in contours])
        features.append(max_area)
        avg_area = np.mean([cv2.contourArea(c) for c in contours])
        features.append(avg_area)
        
        # Aspect ratio (pole should be tall and narrow)
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        aspect_ratio = float(h) / (w + 1e-6)
        features.append(aspect_ratio)
        
        # Vertical elongation indicator
        features.append(1.0 if aspect_ratio > 2.0 else 0.0)
    else:
        features.extend([0, 0, 0, 0, 0])
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    if lines is not None:
        features.append(len(lines))
        lengths = [np.sqrt((x2-x1)**2 + (y2-y1)**2) for x1,y1,x2,y2 in lines[:,0]]
        features.append(np.mean(lengths))
        features.append(np.std(lengths))
        
        # Vertical line ratio (poles have many vertical lines)
        vertical_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            if angle > 75 or angle < 15:
                vertical_lines += 1
        features.append(vertical_lines / (len(lines) + 1e-6))
    else:
        features.extend([0, 0, 0, 0])
    
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    features.append(len(corners) if corners is not None else 0)
    
    # 4. ENHANCED BRIGHTNESS & COLOR REGIONS (12 features)
    bright_mask = gray > 200
    features.append(np.sum(bright_mask) / gray.size)
    
    dark_mask = gray < 50
    features.append(np.sum(dark_mask) / gray.size)
    
    mid_mask = (gray >= 100) & (gray <= 150)
    features.append(np.sum(mid_mask) / gray.size)
    
    green_mask = (hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 85) & (hsv[:,:,1] > 40)
    features.append(np.sum(green_mask) / gray.size)
    
    brown_mask = (hsv[:,:,0] >= 10) & (hsv[:,:,0] <= 20) & (hsv[:,:,1] > 30)
    features.append(np.sum(brown_mask) / gray.size)
    
    wire_mask = (hsv[:,:,1] < 30) & (hsv[:,:,2] < 100)
    features.append(np.sum(wire_mask) / gray.size)
    
    # Spatial brightness distribution (6 more features)
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
    try:
        from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
        
        # LBP (5 features)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-6)
        features.extend(lbp_hist[:5])
        
        # Gabor filters (4 features)
        ksize = 31
        sigma = 4.0
        theta_values = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        for theta in theta_values:
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, 10.0, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            features.append(np.mean(np.abs(filtered)))
        
        # GLCM (6 features - added ASM)
        gray_norm = (gray / (gray.max() + 1e-6) * 255).astype(np.uint8)
        glcm = graycomatrix(gray_norm, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                            levels=256, symmetric=True, normed=True)
        features.append(graycoprops(glcm, 'contrast')[0, 0])
        features.append(graycoprops(glcm, 'dissimilarity')[0, 0])
        features.append(graycoprops(glcm, 'homogeneity')[0, 0])
        features.append(graycoprops(glcm, 'energy')[0, 0])
        features.append(graycoprops(glcm, 'correlation')[0, 0])
        features.append(graycoprops(glcm, 'ASM')[0, 0])  # Angular Second Moment
    except Exception as e:
        # Fallback if skimage not available
        features.extend([0] * 16)
    
    return np.array(features)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'Random Forest',
        'classes': classes,
        'accuracy': metadata['accuracy']
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict pole hazard from uploaded image
    
    Accepts:
    - multipart/form-data with 'image' file
    - JSON with base64 encoded 'image' string
    
    Returns:
    - hazardType: 'urgent' | 'moderate' | 'normal' | 'not_pole'
    - confidence: float (0-1)
    - rawScores: [urgent, moderate, normal] probabilities
    - allClasses: {class_name: probability} for all 4 classes
    - isPole: boolean - whether image appears to be a pole/wires
    """
    try:
        # Handle different input formats
        if 'image' in request.files:
            # Multipart form data
            file = request.files['image']
            img_bytes = file.read()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        elif request.is_json:
            # JSON with base64 image
            data = request.get_json()
            img_data = data.get('image', '')
            
            # Remove data URL prefix if present
            if ',' in img_data:
                img_data = img_data.split(',')[1]
            
            img_bytes = base64.b64decode(img_data)
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # **NON-POLE DETECTION**: Check if image is actually a pole/wires
        is_pole, pole_score = validate_pole_image(img)
        
        if not is_pole:
            return jsonify({
                'hazardType': 'not_pole',
                'confidence': 0.0,
                'isPole': False,
                'poleScore': pole_score,
                'message': 'Image does not appear to be an electric pole or wires. Please upload a valid pole image.',
                'rawScores': [0.0, 0.0, 0.0],
                'allClasses': {'Moderate': 0.0, 'Normal': 0.0, 'Spagetti': 0.0, 'Urgent': 0.0},
                'modelType': 'RandomForest'
            })
        
        # Extract features
        features = extract_features(img)
        features_scaled = scaler.transform([features])
        
        # Get prediction and probabilities
        prediction_idx = classifier.predict(features_scaled)[0]
        probabilities = classifier.predict_proba(features_scaled)[0]
        
        # Map to our class structure
        # Classes: ['Moderate', 'Normal', 'Spagetti', 'Urgent']
        # Map to: ['urgent', 'moderate', 'normal'] for frontend
        # Combine 'Spagetti' + 'Urgent' into 'urgent' category
        
        class_probs = {
            'Moderate': float(probabilities[0]),
            'Normal': float(probabilities[1]),
            'Spagetti': float(probabilities[2]),
            'Urgent': float(probabilities[3])
        }
        
        # Combine for frontend (same as TensorFlow model logic)
        urgent_score = max(class_probs['Spagetti'], class_probs['Urgent'])
        moderate_score = class_probs['Moderate']
        normal_score = class_probs['Normal']
        
        # Determine final prediction
        scores = {
            'urgent': urgent_score,
            'moderate': moderate_score,
            'normal': normal_score
        }
        hazard_type = max(scores, key=scores.get)
        confidence = scores[hazard_type]
        
        response = {
            'hazardType': hazard_type,
            'confidence': confidence,
            'isPole': True,
            'poleScore': pole_score,
            'rawScores': [urgent_score, moderate_score, normal_score],  # [urgent, moderate, normal]
            'allClasses': class_probs,  # All 4 original classes
            'modelType': 'RandomForest',
            'accuracy': metadata['accuracy']['test']
        }
        
        print(f"🎯 Prediction: {hazard_type} ({confidence*100:.1f}%) | Pole: {is_pole} ({pole_score:.2f}) | All: {class_probs}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/test', methods=['GET'])
def test():
    """Simple test endpoint"""
    return jsonify({
        'message': 'Pole Hazard Detection API is running!',
        'endpoints': {
            '/health': 'GET - Health check',
            '/predict': 'POST - Predict from image',
            '/test': 'GET - This endpoint'
        }
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌲 Random Forest Pole Hazard Detection API")
    print("="*70)
    print(f"📊 Model Accuracy: {metadata['accuracy']['test']*100:.1f}% on test set")
    print(f"🎯 Classes: {classes}")
    print("="*70)
    print("\n🚀 Starting Flask server...")
    print("   API will be available at: http://localhost:5000")
    print("   Test endpoint: http://localhost:5000/test")
    print("   Health check: http://localhost:5000/health")
    print("   Prediction: POST http://localhost:5000/predict")
    print("\n   Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
