import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.utils import get_file
from keras.applications.vgg16 import preprocess_input
import os
import urllib.request

app = Flask(__name__)

# Define list of class names
class_names = ['blue_finger', 'Acral_Lentiginous_Melanoma', 'pitting', 'Onychogryphosis', 'clubbing', 'Healthy_Nail']
# base_model = MobileNet(weights = 'imagenet',  include_top = False, input_shape = (224, 224, 3)) 

  
model = tf.keras.models.load_model('nail_diseases_model.h5')

@app.route('/', methods=['GET'])
def home():
    return "<h1>Server is running</h1>"

@app.route('/predict', methods=['POST'])
def predict_skin_disease():
    try:
        # Load and preprocess image
        image_file = request.files['image']

        def load_img(img_path):
            img = cv2.imdecode(np.fromstring(img_path.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads images in BGR, convert to RGB
            img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            return img

        img = load_img(image_file)
        
        # Make prediction on preprocessed image
        predictions = model.predict(img)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class_index]
        return jsonify({'predicted_class': predicted_class_name})

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)