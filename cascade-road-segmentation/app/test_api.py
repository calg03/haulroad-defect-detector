#!/usr/bin/env python3
"""
Simple test script for the Road Defect Segmentation API
"""

import requests
import json
import time
import os
from pathlib import Path


def test_health_check(base_url: str):
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Basic health check passed")
        else:
            print(f"❌ Basic health check failed: {response.status_code}")
            return False
            
        # Test detailed health check
        response = requests.get(f"{base_url}/api/v1/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Detailed health check passed - Status: {health_data.get('status', 'unknown')}")
            print(f"   Model Status: {health_data.get('model_status', 'unknown')}")
            print(f"   Uptime: {health_data.get('uptime_seconds', 0):.1f}s")
        else:
            print(f"❌ Detailed health check failed: {response.status_code}")
            return False
            
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_model_info(base_url: str):
    """Test model info endpoint"""
    print("\n🤖 Testing model info...")
    
    try:
        response = requests.get(f"{base_url}/api/v1/model/info", timeout=10)
        if response.status_code == 200:
            model_info = response.json()
            print("✅ Model info retrieved successfully")
            print(f"   Architecture: {model_info.get('architecture', 'unknown')}")
            print(f"   Device: {model_info.get('device', 'unknown')}")
            print(f"   Classes: {model_info.get('num_classes', 0)}")
            print(f"   Status: {model_info.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info request failed: {e}")
        return False


def test_inference_with_sample_image(base_url: str):
    """Test inference with a sample image"""
    print("\n📸 Testing inference...")
    
    # Create a simple test image
    try:
        from PIL import Image
        import numpy as np
        
        # Create a 512x512 RGB test image
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(test_image)
        test_image_path = "/tmp/test_road_image.jpg"
        img.save(test_image_path, "JPEG")
        
        print(f"   Created test image: {test_image_path}")
        
    except ImportError:
        print("❌ PIL not available, skipping inference test")
        return False
    except Exception as e:
        print(f"❌ Failed to create test image: {e}")
        return False
    
    # Test inference
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            data = {
                'save_outputs': True,
                'overlay_alpha': 0.6,
                'confidence_threshold': 0.6
            }
            
            print("   Sending inference request...")
            start_time = time.time()
            response = requests.post(
                f"{base_url}/api/v1/predict/single",
                files=files,
                data=data,
                timeout=60  # Allow more time for inference
            )
            inference_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                data = result.get('data', {})
                print("✅ Inference completed successfully")
                print(f"   Processing time: {inference_time:.2f}s")
                print(f"   Image shape: {data.get('image_shape', 'unknown')}")
                print(f"   Road coverage: {data.get('road_coverage', 0):.1%}")
                print(f"   Total defect pixels: {data.get('total_defect_pixels', 0)}")
                print(f"   Mean confidence: {data.get('mean_confidence', 0):.3f}")
                return True
            else:
                print(f"❌ Inference failed: {result.get('message', 'unknown error')}")
                return False
        else:
            print(f"❌ Inference request failed: {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Inference request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Inference test error: {e}")
        return False
    finally:
        # Clean up test image
        try:
            os.remove(test_image_path)
        except:
            pass


def test_api_documentation(base_url: str):
    """Test if API documentation is accessible"""
    print("\n📚 Testing API documentation...")
    
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            print("✅ API documentation accessible at /docs")
            return True
        elif response.status_code == 404:
            print("ℹ️ API documentation disabled (production mode)")
            return True
        else:
            print(f"⚠️ API documentation returned status: {response.status_code}")
            return True
            
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Could not access API documentation: {e}")
        return True  # Not critical


def main():
    """Main test function"""
    print("🚀 Testing Road Defect Segmentation API")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:8000"
    
    # Check if server is running
    try:
        response = requests.get(base_url, timeout=5)
        print(f"✅ API server is running at {base_url}")
    except requests.exceptions.RequestException:
        print(f"❌ API server is not accessible at {base_url}")
        print("   Please start the server first:")
        print("   python run_dev.py")
        return False
    
    # Run tests
    tests = [
        ("Health Check", lambda: test_health_check(base_url)),
        ("Model Info", lambda: test_model_info(base_url)),
        ("API Documentation", lambda: test_api_documentation(base_url)),
        ("Inference", lambda: test_inference_with_sample_image(base_url))
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)