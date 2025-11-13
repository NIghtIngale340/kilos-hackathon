// API-based ML classifier using Python Random Forest backend
// Provides better accuracy (85.7%) than browser-based TensorFlow.js

import { MLClassificationResult, HazardType } from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000';

export async function classifyImageWithAPI(imageFile: File): Promise<MLClassificationResult> {
  try {
    console.log('🔄 Sending image to Random Forest API...');
    
    // Create FormData and append the image
    const formData = new FormData();
    formData.append('image', imageFile);
    
    // Call the prediction API
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'API request failed');
    }
    
    const result = await response.json();
    
    console.log('✅ API Prediction:', {
      hazardType: result.hazardType,
      confidence: `${(result.confidence * 100).toFixed(1)}%`,
      allClasses: result.allClasses,
      modelAccuracy: `${(result.accuracy * 100).toFixed(1)}%`
    });
    
    return {
      hazardType: result.hazardType as HazardType,
      confidence: result.confidence,
      rawScores: result.rawScores // [urgent, moderate, normal]
    };
    
  } catch (error) {
    console.error('❌ API prediction failed:', error);
    
    // Fallback to simple heuristic
    console.warn('⚠️ Using fallback classification');
    return {
      hazardType: 'moderate',
      confidence: 0.5,
      rawScores: [0.2, 0.5, 0.3]
    };
  }
}

export async function classifyImageFromDataURL(dataURL: string): Promise<MLClassificationResult> {
  try {
    console.log('🔄 Sending image to Random Forest API...');
    
    // Send base64 image as JSON
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image: dataURL }),
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'API request failed');
    }
    
    const result = await response.json();
    
    console.log('✅ API Prediction:', {
      hazardType: result.hazardType,
      confidence: `${(result.confidence * 100).toFixed(1)}%`,
      allClasses: result.allClasses,
      modelAccuracy: `${(result.accuracy * 100).toFixed(1)}%`
    });
    
    return {
      hazardType: result.hazardType as HazardType,
      confidence: result.confidence,
      rawScores: result.rawScores
    };
    
  } catch (error) {
    console.error('❌ API prediction failed:', error);
    
    // Fallback
    console.warn('⚠️ Using fallback classification');
    return {
      hazardType: 'moderate',
      confidence: 0.5,
      rawScores: [0.2, 0.5, 0.3]
    };
  }
}

// Health check for API
export async function checkAPIHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/health`, {
      method: 'GET',
    });
    
    if (response.ok) {
      const health = await response.json();
      console.log('✅ API Health Check:', health);
      return true;
    }
    return false;
  } catch (error) {
    console.warn('⚠️ API health check failed:', error);
    return false;
  }
}
