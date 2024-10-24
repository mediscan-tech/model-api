import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify   
from tensorflow.keras.applications import EfficientNetB0, DenseNet121, MobileNet
from tensorflow.keras.utils import get_file
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
import os
import urllib.request

app = Flask(__name__)

# Define list of class names
skin_class_names = ['Acne and Rosacea Photos','Melanoma Skin Cancer Nevi and Moles','vitiligo','Tinea Ringworm Candidiasis and other Fungal Infections','Eczema Photos']
threshold_file_size_mb = 350.0
model_file_path = "skin_diseases_model.h5"
# model_file_url = 'https://mediscan.nyc3.cdn.digitaloceanspaces.com/skin_diseases_model.h5'

# Use get_file to fetch and cache the model file
model_file_path = os.path.join('models', 'skin_diseases_model.h5')
# Load the model
model = tf.keras.models.load_model(model_file_path)

nail_class_names = ['blue_finger', 'Acral_Lentiginous_Melanoma', 'pitting', 'Onychogryphosis', 'clubbing', 'Healthy_Nail']
mouth_class_names = ['Calculus', 'Caries', 'Gingivitis', 'Hypodontia', 'Mouth Ulcer', 'Tooth Discoloration']

nail_model = tf.keras.models.load_model('nail_diseases_model.h5')
mouth_model = tf.keras.models.load_model('mouth_diseases_model.h5')

skin_base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(180, 180, 3))
nail_base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
mouth_base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

print('all models loaded')

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
def predict_skin_disease():
    try:
        # Load and preprocess image
        image_file = request.files['image']

        print(image_file)

        skin_img, nail_img, mouth_img = load_and_preprocess_image(image_file)
        
        skin_features = skin_base_model.predict(skin_img)
        nail_features = nail_base_model.predict(nail_img)
        mouth_features = mouth_base_model.predict(mouth_img)

        print(skin_features)
        print(nail_features)
        print(mouth_features)

        print('finished base model predictions')

        # Reshape features
        skin_features = skin_features.reshape(1, -1)
        nail_features = nail_features.reshape(1, -1)
        mouth_features = mouth_features.reshape(1, -1)

        # Get predictions from all models
        skin_pred = skin_model.predict(skin_features)
        nail_pred = nail_model.predict(nail_features)
        mouth_pred = mouth_model.predict(mouth_features)

        print('finished predictions')

        all_preds = np.concatenate([skin_pred, nail_pred, mouth_pred], axis=1)
        final_class_index = np.argmax(all_preds)
        confidence = float(all_preds[0][final_class_index])

        if final_class_index < len(skin_class_names):
            final_class = skin_class_names[final_class_index]
            model_type = 'Skin Disease'
            print('skin disease detected')
        elif final_class_index < len(skin_class_names) + len(nail_class_names):
            final_class = nail_class_names[final_class_index - len(skin_class_names)]
            model_type = 'Nail Disease'
            print('nail disease detected')
        else:
            final_class = mouth_class_names[final_class_index - len(skin_class_names) - len(nail_class_names)]
            model_type = 'Mouth Disease'
            print('mouth disease detected')

        print(f'predicted_class: {final_class} \n model_type: {model_type}\nconfidence: {confidence}')

        return jsonify({
            'predicted_class': final_class,
            'model_type': model_type,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)