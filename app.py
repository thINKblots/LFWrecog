"""Facial Recognition Application using LFW Dataset
This app demonstrates face detection, recognition, and verification using deep learning.
"""

from flask import Flask, request, jsonify, send_file, render_template
import cv2
import numpy as np
import os
from pathlib import Path
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import logging

#conf logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

#paths
DATASET_PATH = './data/lfw-deepfunneled/lfw-deepfunneled/'
MODEL_PATH = './models/'
Path(MODEL_PATH).mkdir(exist_ok=True)

#global model variables
face_detector = None
face_model = None
svm_classifier = None
label_encoder = None

#STEP 1: FACE DETECTION

def initialize_face_detector():
    global face_detector
    logger.info("Initializing MTCNN face detector...")
    face_detector = MTCNN()
    return face_detector

def detect_face(img_array):
    """
    Detect face in an image using MTCNN
    Args:
        img_array: numpy array of the image (RGB format)
    Returns:
        Cropped face array or None if no face detected
    """
    if face_detector is None:
        initialize_face_detector()
    
    #detect faces
    results = face_detector.detect_faces(img_array)

    if len(results) == 0:
        return None
    
    #get first face
    x, y, width, height = results[0]['box']
    x, y = abs(x), abs(y)

    #extract face with padding
    face= img_array[y:y+height, x:x+width]

    return face

#STEP 2: FACE EMBEDDING

def initialize_face_model():
    """
    Initialize VGG16 model for face embeddings
    We use VGG16 pre-trained on ImageNet as a feature extractor
    """
    global face_model
    logger.info("Initializing VGG16 model for embeddings...")

    #load VGG16 without top layers (classifier)
    face_model = VGG16(
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg'
    )
    return face_model

def get_face_embedding(face_pixels):
    """
    Extract face embedding (feature vector) from face image
    
    Args:
        face_pixels: numpy array of face image
    Returns:
        126-dimensional embedding vector
    """
    if face_model is None:
        initialize_face_model()

    #resize to 224x224 for VGG16
    face_pixels = cv2.resize(face_pixels, (224, 224))
    #convert to array with expand dimensions
    face_pixels = np.expand_dims(face_pixels, axis=0)
    #preprocess for VGG16
    face_pixels = preprocess_input(face_pixels)
    #get embedding
    embedding = face_model.predict(face_pixels, verbose=0)

    return embedding[0]

#STEP 3: LOAD DATASET AND CREATE EMBEDDINGS

def load_dataset(min_faces=10):
    """
    Load face dataset and create embeddings
    
    Args:
        min_faces: Minimum number of images per person to include
        
    Returns:
        embeddings: numpy array of face embeddings
        labels: list of person names
    """
    logger.info("Loading dataset from %s", DATASET_PATH)

    embeddings_list = []
    labels_list = []

    #iterate through each person's folder
    for person_name in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_name)

        if not os.path.isdir(person_path):
            continue

        #get all images for this person
        image_files = [f for f in os.listdir(person_path) if f.endswith('.jpg')]

        #skip if too few images
        if len(image_files) < min_faces:
            continue

        logger.info(f"Processing {person_name} ({len(image_files)} images)...")

        for img_file in image_files:
            img_path = os.path.join(person_path, img_file)

            try:
                #load image
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                #detect face
                face = detect_face(img_rgb)

                if face is None:
                    continue

                #get embedding
                embedding = get_face_embedding(face)

                embeddings_list.append(embedding)
                labels_list.append(person_name)
            
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue
    
    embeddings = np.array(embeddings_list)
    labels = np.array(labels_list)

    logger.info(f"Loaded {len(embeddings)} face embeddings from {len(set(labels))} people.")
    
    return embeddings, labels

#STEP 4: TRAIN FACE RECOGNITION MODEL

def train_face_recognizer(embeddings, labels):
    """
    Train SVM classifier for face recognition
    
    Args:
        embeddings: numpy array of face embeddings
        labels: numpy array of person names
    
    Returns:
        classifier: trained SVM model
        encoder: label encoder
    """
    global svm_classifier, label_encoder

    logger.info("Training face recognition model...")

    #encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    #train SVM classifier
    svm_classifier = SVC(kernel='linear', probability=True)
    svm_classifier.fit(embeddings, labels_encoded)

    #save models
    with open(os.path.join(MODEL_PATH, 'svm_classifier.pkl'), 'wb') as f:
        pickle.dump(svm_classifier, f)

    with open(os.path.join(MODEL_PATH, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    with open(os.path.join(MODEL_PATH, 'face_embeddings.pkl'), 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'labels': labels}, f)

    logger.info(f"Model trained on {len(set(labels))} people.")

    return svm_classifier, label_encoder

def load_trained_models():
    """Load pre-trained models from disk"""
    global svm_classifier, label_encoder

    try:
        with open(os.path.join(MODEL_PATH, 'svm_classifier.pkl'), 'rb') as f:
            svm_classifier = pickle.load(f)

        with open(os.path.join(MODEL_PATH, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)

        logger.info("Loaded trained models from disk.")
        return True
    except FileNotFoundError:
        logger.warning("No pre-trained models found.")
        return False
    
#STEP 5: FACE RECOGNITION FUNCTIONS

def recognize_face(img_array):
    """
    Recognize a face in an image
    
    Args:
        img_array:  numpy array of image (RGB format)
        
    Returns:
        dict with person name and confidence
    """
    if svm_classifier is None or label_encoder is None:
        return {"error": "Models not trained."}
    
    #detect face
    face = detect_face(img_array)

    if face is None:
        return {"error": "No face detected."}
    
    #get embedding
    embedding = get_face_embedding(face)
    embedding = embedding.reshape(1, -1)

    #predict
    prediction = svm_classifier.predict(embedding)[0]
    probabilities = svm_classifier.predict_proba(embedding)[0]

    #get person name and confidence
    person_name = label_encoder.inverse_transform([prediction])[0]
    confidence = probabilities[prediction]

    return {
        "person": person_name,
        "confidence": float(confidence),
        "all_probabilities": {
            label_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(probabilities)
        }
    }

def verify_faces(img1_array, img2_array, threshold=0.6):
    """
    Verify if two images are of the same person
    
    Args:
        img1_array: numpy array of first image (RGB format)
        img2_array: numpy array of second image (RGB format)
        threshold: cosine similarity threshold for verification (0-1)

    Returns:
        dict with similarity score and verification result
    """
    #detect faces
    face1 = detect_face(img1_array)
    face2 = detect_face(img2_array)

    if face1 is None or face2 is None:
        return {"error": "Face not detected in one or both images."}
    
    #get embeddings
    embedding1 = get_face_embedding(face1).reshape(1, -1)
    embedding2 = get_face_embedding(face2).reshape(1, -1)

    #compute cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)[0][0]

    is_same_person = similarity >= threshold

    return {
        "is_same_person": bool(similarity >= threshold),
        "similarity": float(similarity),
        "threshold": threshold
    }

#STEP 6: FLASK API ENDPOINTS

@app.route('/')
def home():
    return jsonify({
        "message": "Facial Recognition API",
        "endpoints": {
            "/train": "POST - Train the model on the dataset",
            "/recognize": "POST - Recognize face in an image",
            "/verify": "POST - Verify if two images are of the same person",
            "/status": "GET - Check API status"
        }
    })

@app.route('/status', methods=['GET'])
def status():
    """Check if models are loaded"""
    return jsonify({
        "face_detector_loaded": face_detector is not None,
        "face_model_loaded": face_model is not None,
        "classifier_loaded": svm_classifier is not None,
        "ready": all([face_detector, face_model, svm_classifier])
    })


@app.route('/train', methods=['POST'])
def train():
    """Train the face recognition model on the dataset"""
    try:
        # Initialize models
        initialize_face_detector()
        initialize_face_model()
        
        # Load dataset
        min_faces = request.json.get('min_faces', 10) if request.json else 10
        embeddings, labels = load_dataset(min_faces=min_faces)
        
        # Train classifier
        train_face_recognizer(embeddings, labels)
        
        return jsonify({
            "status": "success",
            "message": f"Model trained on {len(embeddings)} images from {len(set(labels))} people"
        })
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/recognize', methods=['POST'])
def recognize():
    """Recognize face in uploaded image"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Recognize face
        result = recognize_face(img_rgb)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/verify', methods=['POST'])
def verify():
    """Verify if two images are of the same person"""
    try:
        # Check if both image files are present
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({"error": "Two image files required (image1 and image2)"}), 400
        
        # Read first image
        img1_bytes = request.files['image1'].read()
        nparr1 = np.frombuffer(img1_bytes, np.uint8)
        img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        
        # Read second image
        img2_bytes = request.files['image2'].read()
        nparr2 = np.frombuffer(img2_bytes, np.uint8)
        img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Get threshold from request
        threshold = float(request.form.get('threshold', 0.6))
        
        # Verify faces
        result = verify_faces(img1_rgb, img2_rgb, threshold)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/web')
def web():
    """Serve the web interface"""
    return render_template('index.html')

# ============================================
# STEP 7: INITIALIZE AND RUN
# ============================================

if __name__ == '__main__':
    # Try to load pre-trained models
    initialize_face_detector()
    initialize_face_model()
    
    if not load_trained_models():
        logger.info("No pre-trained models found. Use /train endpoint to train the model.")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)