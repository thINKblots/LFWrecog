"""
Test script for facial recognition app
"""

import requests
import os

BASE_URL = "http://localhost:5000"

def test_status():
    """Test API status"""
    print("\n=== Testing Status ===")
    response = requests.get(f"{BASE_URL}/status")
    print(response.json())


def test_train():
    """Train the model"""
    print("\n=== Training Model ===")
    print("This may take several minutes...")
    
    response = requests.post(
        f"{BASE_URL}/train",
        json={"min_faces": 10}
    )
    print(response.json())


def test_recognize(image_path):
    """Test face recognition"""
    print(f"\n=== Recognizing Face in {image_path} ===")
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{BASE_URL}/recognize", files=files)
    
    print(response.json())


def test_verify(image1_path, image2_path):
    """Test face verification"""
    print(f"\n=== Verifying {image1_path} vs {image2_path} ===")
    
    with open(image1_path, 'rb') as f1, open(image2_path, 'rb') as f2:
        files = {
            'image1': f1,
            'image2': f2
        }
        response = requests.post(f"{BASE_URL}/verify", files=files)
    
    print(response.json())


if __name__ == "__main__":
    # Test status
    test_status()
    
    # Train model (uncomment to train)
    # test_train()
    
    # Example: Test recognition (replace with actual image path)
    # test_recognize("./data/lfw-deepfunneled/lfw-deepfunneled/George_W_Bush/George_W_Bush_0001.jpg")
    
    # Example: Test verification (replace with actual image paths)
    # test_verify(
    #     "./data/lfw-deepfunneled/lfw-deepfunneled/George_W_Bush/George_W_Bush_0001.jpg",
    #     "./data/lfw-deepfunneled/lfw-deepfunneled/George_W_Bush/George_W_Bush_0002.jpg"
    # )