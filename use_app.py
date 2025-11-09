"""
Complete usage examples for the facial recognition app
"""

import requests
import os

BASE_URL = "http://localhost:5000"

def check_status():
    """Check if API is ready"""
    print("\n" + "="*50)
    print("CHECKING API STATUS")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/status")
    status = response.json()
    
    print(f"Face Detector: {'✓' if status['face_detector_loaded'] else '✗'}")
    print(f"Face Model: {'✓' if status['face_model_loaded'] else '✗'}")
    print(f"Classifier: {'✓' if status['classifier_loaded'] else '✗'}")
    print(f"Ready: {'✓' if status['ready'] else '✗'}")
    
    return status['ready']


def recognize_face(image_path):
    """Recognize who is in the photo"""
    print("\n" + "="*50)
    print(f"RECOGNIZING FACE IN: {image_path}")
    print("="*50)
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{BASE_URL}/recognize", files=files)
    
    if response.status_code != 200:
        print(f"❌ Error: {response.json()}")
        return
    
    result = response.json()
    
    if 'error' in result:
        print(f"❌ {result['error']}")
    else:
        print(f"\n✓ Person Identified: {result['person']}")
        print(f"✓ Confidence: {result['confidence']:.2%}")
        print(f"\nTop 3 Matches:")
        
        # Sort by probability
        probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        for i, (name, prob) in enumerate(probs[:3], 1):
            print(f"  {i}. {name}: {prob:.2%}")


def verify_faces(image1_path, image2_path, threshold=0.6):
    """Verify if two photos are the same person"""
    print("\n" + "="*50)
    print(f"VERIFYING FACES")
    print("="*50)
    print(f"Image 1: {image1_path}")
    print(f"Image 2: {image2_path}")
    print(f"Threshold: {threshold}")
    
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        print("❌ Error: One or both images not found")
        return
    
    with open(image1_path, 'rb') as f1, open(image2_path, 'rb') as f2:
        files = {
            'image1': f1,
            'image2': f2
        }
        data = {'threshold': threshold}
        response = requests.post(f"{BASE_URL}/verify", files=files, data=data)
    
    if response.status_code != 200:
        print(f"❌ Error: {response.json()}")
        return
    
    result = response.json()
    
    if 'error' in result:
        print(f"❌ {result['error']}")
    else:
        print(f"\n✓ Same Person? {'YES' if result['is_same_person'] else 'NO'}")
        print(f"✓ Similarity Score: {result['similarity']:.2%}")
        
        if result['similarity'] >= threshold:
            print(f"✓ Above threshold ({threshold}) - MATCH!")
        else:
            print(f"✗ Below threshold ({threshold}) - NO MATCH")


def test_with_dataset():
    """Test with images from the dataset"""
    print("\n" + "="*50)
    print("TESTING WITH DATASET IMAGES")
    print("="*50)
    
    dataset_path = './data/lfw-deepfunneled/lfw-deepfunneled/'
    
    # Find some sample images
    people = []
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            images = [f for f in os.listdir(person_path) if f.endswith('.jpg')]
            if len(images) >= 2:
                people.append({
                    'name': person_name,
                    'path': person_path,
                    'images': images
                })
        
        if len(people) >= 3:
            break
    
    if not people:
        print("No suitable test images found in dataset")
        return
    
    # Test 1: Recognize a face
    print(f"\nTest 1: Recognizing {people[0]['name']}")
    test_image = os.path.join(people[0]['path'], people[0]['images'][0])
    recognize_face(test_image)
    
    # Test 2: Verify same person (should match)
    print(f"\nTest 2: Verifying same person ({people[0]['name']})")
    img1 = os.path.join(people[0]['path'], people[0]['images'][0])
    img2 = os.path.join(people[0]['path'], people[0]['images'][1])
    verify_faces(img1, img2, threshold=0.9)
    
    # Test 3: Verify different people (should not match)
    if len(people) >= 2:
        print(f"\nTest 3: Verifying different people ({people[0]['name']} vs {people[1]['name']})")
        img1 = os.path.join(people[0]['path'], people[0]['images'][0])
        img2 = os.path.join(people[1]['path'], people[1]['images'][0])
        verify_faces(img1, img2, threshold=0.8)


def main():
    """Main function"""
    print("\n" + "="*60)
    print("FACIAL RECOGNITION APP - USAGE EXAMPLES")
    print("="*60)
    
    # Check status
    if not check_status():
        print("\n❌ API is not ready. Make sure:")
        print("  1. Flask app is running (python app.py)")
        print("  2. Model has been trained (curl -X POST http://localhost:5000/train)")
        return
    
    print("\n✓ API is ready!")
    
    # Run tests with dataset
    test_with_dataset()
    
    print("\n" + "="*60)
    print("CUSTOM USAGE")
    print("="*60)
    print("\nTo recognize your own images:")
    print("  recognize_face('/path/to/your/photo.jpg')")
    print("\nTo verify two images:")
    print("  verify_faces('/path/to/photo1.jpg', '/path/to/photo2.jpg')")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
