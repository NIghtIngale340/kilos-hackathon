"""
Quick test script to verify the Random Forest API is working correctly
Tests all endpoints and sample predictions
"""

import requests
import json
from pathlib import Path

API_URL = "http://localhost:5000"

def test_health():
    """Test the health endpoint"""
    print("\n" + "="*60)
    print("🏥 Testing Health Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed!")
            print(f"   Status: {data['status']}")
            print(f"   Model: {data['model']}")
            print(f"   Classes: {data['classes']}")
            print(f"   Test Accuracy: {data['accuracy']['test']*100:.1f}%")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("   Make sure Flask server is running: python api_server.py")
        return False


def test_predict(image_path):
    """Test prediction on a single image"""
    print(f"\n📸 Testing prediction on: {image_path.name}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{API_URL}/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Predicted: {data['hazardType'].upper()}")
            print(f"   Confidence: {data['confidence']*100:.1f}%")
            print(f"   All Classes:")
            for cls, prob in data['allClasses'].items():
                print(f"      {cls}: {prob*100:.1f}%")
            return data
        else:
            error = response.json()
            print(f"   ❌ Prediction failed: {error.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def test_all_classes():
    """Test prediction on sample images from each class"""
    print("\n" + "="*60)
    print("🎯 Testing Predictions on All Classes")
    print("="*60)
    
    dataset_path = Path(__file__).parent.parent / 'pole-mock-dataset' / 'test'
    classes = ['Moderate', 'Normal', 'Spagetti', 'Urgent']
    
    results = {
        'total': 0,
        'correct': 0,
        'by_class': {}
    }
    
    for class_name in classes:
        class_path = dataset_path / class_name
        if not class_path.exists():
            print(f"\n⚠️ {class_name} folder not found")
            continue
        
        images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg')) + list(class_path.glob('*.png'))
        
        if not images:
            print(f"\n⚠️ No images in {class_name}")
            continue
        
        print(f"\n{'='*60}")
        print(f"CLASS: {class_name}")
        print(f"{'='*60}")
        
        class_correct = 0
        class_total = 0
        
        # Test first 3 images from each class
        for img_path in images[:3]:
            result = test_predict(img_path)
            
            if result:
                class_total += 1
                results['total'] += 1
                
                # Check if prediction is correct
                predicted = result['hazardType'].lower()
                actual = class_name.lower()
                
                # Handle special cases
                is_correct = False
                if actual == 'spagetti' and predicted == 'urgent':
                    is_correct = True  # Spagetti maps to urgent
                elif actual == 'urgent' and predicted == 'urgent':
                    is_correct = True
                elif predicted == actual:
                    is_correct = True
                
                if is_correct:
                    class_correct += 1
                    results['correct'] += 1
                    print(f"      ✓ CORRECT")
                else:
                    print(f"      ✗ WRONG (expected {actual}, got {predicted})")
        
        accuracy = (class_correct / class_total * 100) if class_total > 0 else 0
        results['by_class'][class_name] = {
            'correct': class_correct,
            'total': class_total,
            'accuracy': accuracy
        }
        
        print(f"\n   Class Accuracy: {class_correct}/{class_total} = {accuracy:.1f}%")
    
    # Overall results
    print("\n" + "="*60)
    print("📊 OVERALL RESULTS")
    print("="*60)
    
    overall_acc = (results['correct'] / results['total'] * 100) if results['total'] > 0 else 0
    print(f"\nTotal Accuracy: {results['correct']}/{results['total']} = {overall_acc:.1f}%")
    
    print("\nBy Class:")
    for cls, stats in results['by_class'].items():
        print(f"  {cls:12} {stats['correct']}/{stats['total']} ({stats['accuracy']:.1f}%)")
    
    return results


def main():
    print("="*60)
    print("🧪 Random Forest API Test Suite")
    print("="*60)
    
    # Test 1: Health check
    if not test_health():
        print("\n❌ API is not running. Please start it first:")
        print("   python api_server.py")
        return
    
    # Test 2: Test endpoint
    print("\n" + "="*60)
    print("🧪 Testing Test Endpoint")
    print("="*60)
    try:
        response = requests.get(f"{API_URL}/test")
        if response.status_code == 200:
            print("✅ Test endpoint working!")
            print(f"   {response.json()['message']}")
        else:
            print("⚠️ Test endpoint returned:", response.status_code)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Predictions on all classes
    results = test_all_classes()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETE!")
    print("="*60)
    
    if results['total'] > 0:
        overall_acc = (results['correct'] / results['total']) * 100
        if overall_acc >= 80:
            print(f"\n🎉 Great! Model accuracy is {overall_acc:.1f}%")
        elif overall_acc >= 60:
            print(f"\n✅ Good! Model accuracy is {overall_acc:.1f}%")
            print("   Consider collecting more training data to improve")
        else:
            print(f"\n⚠️ Model accuracy is {overall_acc:.1f}%")
            print("   You should retrain with more/better data")


if __name__ == '__main__':
    main()
