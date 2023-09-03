import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define list of class names
class_names = ["Acne", "Eczema", "Atopic", "Psoriasis", "Tinea", "vitiligo"]
vgg_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(180, 180, 3))
model = tf.keras.models.load_model('6claass.h5')

@app.route('/', methods=['GET'])
def home():
    return "<h1>Server is running</h1>"

@app.route('/predict', methods=['POST'])
def predict_skin_disease():
    try:
        # Load and preprocess image
        image_file = request.files['image']
        img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (180, 180))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        img = vgg_model.predict(img)
        img = img.reshape(1, -1)
        
        # Make prediction on preprocessed image
        pred = model.predict(img)[0]
        predicted_class_index = np.argmax(pred)
        predicted_class_name = class_names[predicted_class_index]

        return jsonify({'predicted_class': predicted_class_name})

    except Exception as e:
        return jsonify({'error': str(e)})

main = app

if __name__ == "__main__":
    app.run(host='0.0.0.0')