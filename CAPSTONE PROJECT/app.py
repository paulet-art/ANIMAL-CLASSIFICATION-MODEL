# Import the necessary libraries
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
from flask import Flask, request, jsonify, render_template

# Load the saved model from the pickle file
with open('animals_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the Flask app
app = Flask(__name__)

# Define the route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']
    
    # Read the image file and preprocess it for prediction
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255
    
    # Make a prediction using the loaded model
    prediction = model.predict(img)
    classes = ['bird', 'cat', 'dog', 'horse']
    class_index = np.argmax(prediction)
    class_label = classes[class_index]
    
    # Return the predicted class label to the index.html template
    return render_template('index.html', prediction=class_label)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
