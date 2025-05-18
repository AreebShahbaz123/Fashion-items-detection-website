import cv2
import numpy as np
import tensorflow as tf
import os
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the model
def load_model():
    try:
        zip_path = os.path.join(os.path.dirname(__file__), 'model.zip')
        model_path = os.path.join(os.path.dirname(__file__), 'fashion_mnist_model.h5')
        
        # Extract model from zip if it doesn't exist
        if not os.path.exists(model_path):
            print(f"Extracting model from {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract('fashion_mnist_model.h5', os.path.dirname(__file__))
        
        print(f"Loading model from: {model_path}")  # Debug print
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")  # Debug print
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Preprocess the image using OpenCV for Fashion MNIST
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"cv2.imread failed for {image_path}")
            return None
        h, w = img.shape
        if h > w:
            pad = (h - w) // 2
            img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=255)
        elif w > h:
            pad = (w - h) // 2
            img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=255)
        img = cv2.resize(img, (28, 28))
        # Invert colors if background is white
        if np.mean(img) > 127:
            img = 255 - img
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)
        return img
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        return None

# Get prediction
def predict_image(image_path):
    try:
        model = load_model()
        if model is None:
            print("Model loading failed")  # Debug print
            return None, None

        processed_image = preprocess_image(image_path)
        if processed_image is None:
            print("Image preprocessing failed")  # Debug print
            return None, None

        print("Making prediction...")  # Debug print
        prediction = model.predict(processed_image, verbose=0)
        
        # Get top 2 predictions
        top_2_indices = np.argsort(prediction[0])[-2:][::-1]
        top_2_confidences = prediction[0][top_2_indices]
        
        # Fashion MNIST class names
        class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        
        # Get the highest confidence prediction
        highest_confidence = float(top_2_confidences[0])
        highest_class = class_names[top_2_indices[0]]
        
        # Get the second highest confidence
        second_confidence = float(top_2_confidences[1])
        second_class = class_names[top_2_indices[1]]
        
        print(f"Top prediction: {highest_class} with confidence {highest_confidence:.2%}")
        print(f"Second prediction: {second_class} with confidence {second_confidence:.2%}")
        
        print(f"Returning top prediction: {highest_class} with confidence {highest_confidence:.2%}")
        return highest_class, highest_confidence
            
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None, None 