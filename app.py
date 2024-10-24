import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications import EfficientNetB0, DenseNet121, MobileNet
from tensorflow.keras.utils import get_file
from keras.applications.vgg16 import preprocess_input
import os
import json
import urllib.request

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://mediscan.care", "http://localhost:3000", "http://localhost:5000"]}})

skin_class_names = ['Acne and Rosacea Photos','Melanoma Skin Cancer Nevi and Moles','vitiligo','Tinea Ringworm Candidiasis and other Fungal Infections','Eczema Photos']
nail_class_names = ['Acral_Lentiginous_Melanoma','blue_finger', 'clubbing', 'Onychogryphosis', 'pitting']
mouth_class_names = ['Calculus', 'Caries', 'Gingivitis', 'Hypodontia', 'Mouth Ulcer', 'Tooth Discoloration']

sbase_model = EfficientNetB0(weights = 'imagenet',  include_top = False, input_shape = (180, 180, 3)) 

model_file_path = "skin_diseases_model.h5"
model_file_path = os.path.join('models', 'skin_diseases_model.h5')
skin_model = tf.keras.models.load_model(model_file_path)
nail_model = tf.keras.models.load_model('nail_diseases_model.h5')
mouth_model = tf.keras.models.load_model('mouth_diseases_model.h5')

def preprocess_skin_image(img_data):
    images=[]
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    img=cv2.resize(img,(180,180))
    images.append(img)
    x_test=np.asarray(images)
    test_img=preprocess_input(x_test)
    features_test=sbase_model.predict(test_img)
    num_test=x_test.shape[0]
    f_img=features_test.reshape(1, -1)
    return f_img

def preprocess_mouth_image(img_data):
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads images in BGR, convert to RGB
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def preprocess_nail_image(img_data):
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads images in BGR, convert to RGB
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/', methods=['GET'])
def home():
    return "<h1>Server is running</h1>"

@app.route('/predict', methods=['POST'])
def predict_skin_disease():
    try:
        model_type = request.form.get('model_type')
        print(model_type)
        image_file = request.files.get('image')
        print(image_file)
        image_data = image_file.read()
 
        if not model_type or not image_file:
            return jsonify({'error': 'Missing model_type or image'}), 400
        
        if(model_type == 'skin'):
            img = preprocess_skin_image(image_data)
            skin_prediction = skin_model.predict(img)
            predicted_class_index = np.argmax(skin_prediction)
            predicted_class_name = skin_class_names[predicted_class_index]
            confidence = float(skin_prediction[0][predicted_class_index])
            print(f'Predicted Skin Disease: {predicted_class_name} | Confidence Level: {confidence}')
            return jsonify({'predicted_class': predicted_class_name, 'confidence': confidence})
        elif(model_type == 'mouth'):
            img = preprocess_mouth_image(image_data)
            mouth_prediction = mouth_model.predict(img)
            predicted_class_index = np.argmax(mouth_prediction)
            predicted_class_name = mouth_class_names[predicted_class_index]
            confidence = float(mouth_prediction[0][predicted_class_index])
            print(f'Predicted Mouth Disease: {predicted_class_name} | Confidence Level: {confidence}')
            return jsonify({'predicted_class': predicted_class_name, 'confidence': confidence})
        elif(model_type == 'nail'):
            img = preprocess_nail_image(image_data)
            nail_prediction = nail_model.predict(img)
            predicted_class_index = np.argmax(nail_prediction)
            predicted_class_name = nail_class_names[predicted_class_index]
            confidence = float(nail_prediction[0][predicted_class_index])
            print(f'Predicted Nail Disease: {predicted_class_name} | Confidence Level: {confidence}')
            return jsonify({'predicted_class': predicted_class_name, 'confidence': confidence})

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
