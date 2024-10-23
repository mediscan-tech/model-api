import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.applications import EfficientNetB0, DenseNet121, MobileNet
from tensorflow.keras.utils import get_file
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define class names for each model
skin_class_names = ['Acne and Rosacea Photos','Melanoma Skin Cancer Nevi and Moles','vitiligo','Tinea Ringworm Candidiasis and other Fungal Infections','Eczema Photos']
nail_class_names = ['blue_finger', 'Acral_Lentiginous_Melanoma', 'pitting', 'Onychogryphosis', 'clubbing', 'Healthy_Nail']
mouth_class_names = ['Calculus', 'Caries', 'Gingivitis', 'Hypodontia', 'Mouth Ulcer', 'Tooth Discoloration']

# Model URLs
skin_model_url = 'https://mediscan.nyc3.cdn.digitaloceanspaces.com/skin_diseases_model.h5'
nail_model_url = 'https://mediscan.nyc3.cdn.digitaloceanspaces.com/nail_diseases_model.h5'
mouth_model_url = 'https://mediscan.nyc3.cdn.digitaloceanspaces.com/oral_diseases_model.h5'

# Safe model loading function
def safe_load_model(model_path, model_url):
    try:
        logging.info(f"Attempting to load model from {model_path}")
        model = tf.keras.models.load_model(get_file(model_path, model_url, cache_subdir='models'))
        logging.info(f"Successfully loaded model {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model {model_path}: {str(e)}")
        return None

# Load models
skin_model = safe_load_model('mediscan_nrfinal.h5', skin_model_url)
nail_model = safe_load_model('nail_disease_model.h5', nail_model_url)
mouth_model = safe_load_model('mouth_disease_model.h5', mouth_model_url)

# Check if all models loaded successfully
if not all([skin_model, nail_model, mouth_model]):
    logging.error("Not all models loaded successfully. Application may not function correctly.")

# Load base models for feature extraction
skin_base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(180, 180, 3))
nail_base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
mouth_base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def preprocess_image(img, target_size, preprocess_func):
    img = cv2.resize(img, target_size)
    img = preprocess_func(img)
    return np.expand_dims(img, axis=0)

def load_and_preprocess_image(img_file):
    img = cv2.imdecode(np.fromstring(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    skin_img = preprocess_image(img, (180, 180), efficientnet_preprocess)
    nail_img = preprocess_image(img, (224, 224), densenet_preprocess)
    mouth_img = preprocess_image(img, (224, 224), mobilenet_preprocess)
    
    return skin_img, nail_img, mouth_img

@app.route('/', methods=['GET'])
def home():
    return "<h1>Server is running</h1>"

@app.route('/predict', methods=['POST'])
def predict_disease():
    try:
        image_file = request.files['image']
        skin_img, nail_img, mouth_img = load_and_preprocess_image(image_file)

        # Extract features
        skin_features = skin_base_model.predict(skin_img)
        nail_features = nail_base_model.predict(nail_img)
        mouth_features = mouth_base_model.predict(mouth_img)

        # Reshape features
        skin_features = skin_features.reshape(1, -1)
        nail_features = nail_features.reshape(1, -1)
        mouth_features = mouth_features.reshape(1, -1)

        # Get predictions from all models
        skin_pred = skin_model.predict(skin_features) if skin_model else np.zeros((1, len(skin_class_names)))
        nail_pred = nail_model.predict(nail_features) if nail_model else np.zeros((1, len(nail_class_names)))
        mouth_pred = mouth_model.predict(mouth_features) if mouth_model else np.zeros((1, len(mouth_class_names)))

        # Combine predictions
        all_preds = np.concatenate([skin_pred, nail_pred, mouth_pred], axis=1)
        
        # Voting: select the class with the highest probability across all models
        final_class_index = np.argmax(all_preds)
        confidence = float(all_preds[0][final_class_index])

        # Determine which model and class the prediction belongs to
        if final_class_index < len(skin_class_names):
            final_class = skin_class_names[final_class_index]
            model_type = 'Skin Disease'
        elif final_class_index < len(skin_class_names) + len(nail_class_names):
            final_class = nail_class_names[final_class_index - len(skin_class_names)]
            model_type = 'Nail Disease'
        else:
            final_class = mouth_class_names[final_class_index - len(skin_class_names) - len(nail_class_names)]
            model_type = 'Mouth Disease'

        return jsonify({
            'predicted_class': final_class,
            'model_type': model_type,
            'confidence': confidence
        })

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)