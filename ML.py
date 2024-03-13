from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Function to process uploaded image
def process_image(image):
    
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    return image

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Read the image
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    
    # Process the image
    processed_image = process_image(image)
    
    # Predict the model
    prediction = model.predict(processed_image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    if(class_name[0]=='0'):
        return f"Class: {class_name[2:]} Stage III,Treatment:radiation therapy Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%"
    elif(class_name[:2]=='11'):
        return f"Class: {class_name[2:]} Stage IIIB, Treatment: chemotherapy and immunotherapy Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%"
    elif(class_name[:2]=='10'):
        return f"Class: {class_name[2:]} Stage IV, Treatment: chemotherapy and stem cell transplantation Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%"
    elif(class_name[0]=='2'):
        return f"Class: {class_name[2:]} Stage II, Treatment: surgery followed by chemotherapy Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%"
    elif(class_name[0]=='4'):
        return f"Class: {class_name[2:]} Stage IV, Treatment: chemotherapy, radiation therapy, and possibly surgery Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%"

    return f"Class: {class_name[2:]}  Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%"

if __name__ == '__main__':
    app.run(debug=True)
