import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.applications import VGG19, EfficientNetB0, VGG16, InceptionV3, ResNet50, EfficientNetB3
from tensorflow.keras.utils import get_file
from keras.applications.vgg16 import preprocess_input
import os
import urllib.request

app = Flask(__name__)

# Define list of class names
class_names = ['Acne and Rosacea Photos','Melanoma Skin Cancer Nevi and Moles','vitiligo','Tinea Ringworm Candidiasis and other Fungal Infections','Eczema Photos']
threshold_file_size_mb = 350.0
# model_file_url = 'https://mediscan.nyc3.digitaloceanspaces.com/mediscan_nrfinal.h5'
vgg_model = EfficientNetB0(weights = 'imagenet',  include_top = False, input_shape = (180, 180, 3)) 

model_file_path = os.path.join('models', 'skin_diseases_model.h5')
model = tf.keras.models.load_model(model_file_path)

@app.route('/', methods=['GET'])
def home():
    return "<h1>Server is running</h1>"

@app.route('/predict', methods=['POST'])
def predict_skin_disease():
    try:
        # Load and preprocess image
        image_file = request.files['image']

        def load_img(img_path):
            images=[]
            img = cv2.imdecode(np.fromstring(img_path.read(), np.uint8), cv2.IMREAD_COLOR)
            img=cv2.resize(img,(180,180))
            images.append(img)
            x_test=np.asarray(images)
            test_img=preprocess_input(x_test)
            features_test=vgg_model.predict(test_img)
            num_test=x_test.shape[0]
            f_img=features_test.reshape(1, -1)

            return f_img

        img = load_img(image_file)

        print(img.shape)
        
        # Make prediction on preprocessed image
        predicted_class_index = np.argmax(model.predict(img))
        predicted_class_name = class_names[predicted_class_index]
        return jsonify({'predicted_class': predicted_class_name})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)