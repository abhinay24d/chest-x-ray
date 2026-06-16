"""
==========================================================================
AI Chest X-Ray Disease Detection System - Flask Backend Server
==========================================================================
This Flask server loads a pre-trained Random Forest model at startup,
preprocesses uploaded chest X-ray images (converts to grayscale, resizes to
64x64, flattens), and classifies them into NORMAL, PNEUMONIA, or TUBERCULOSIS.
"""

import os
import uuid
import joblib
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Config uploads folder and valid formats
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper: validate file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 1. Load the AI model once when the server starts
MODEL_PATH = os.path.join('model', 'tb_pneumonia_random_forest.pkl')
try:
    print(f"[SYSTEM] Loading machine learning model from '{MODEL_PATH}'...")
    model = joblib.load(MODEL_PATH)
    print("[SYSTEM] Model loaded successfully.")
    print(f"[SYSTEM] Model classes: {model.classes_}")
except Exception as e:
    print(f"[CRITICAL ERROR] Failed to load model: {str(e)}")
    model = None

# GET / - Render homepage
@app.route('/')
def index():
    return render_template('index.html')

# POST /predict - Process image upload and return prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Verify model is loaded
    if model is None:
        return jsonify({
            "success": False,
            "error": "Machine learning model is not initialized on the server."
        }), 500

    try:
        # Check if file part is present
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded in the request."
            }), 400

        file = request.files['file']
        
        # Check if empty filename
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected."
            }), 400

        # Validate file format
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "Invalid file format. Only JPG, JPEG, and PNG images are supported."
            }), 400

        # Generate unique secure filename to prevent collisions
        orig_filename = secure_filename(file.filename)
        unique_id = uuid.uuid4().hex
        filename = f"{unique_id}_{orig_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save file to upload directory
        file.save(filepath)

        # 3. Preprocess the image exactly like the training pipeline
        try:
            img = Image.open(filepath)
            img = img.convert("L")  # Grayscale conversion
            img = img.resize((64, 64))  # Resize to 64x64
            x = np.array(img).flatten().reshape(1, -1)  # Flatten to shape (1, 4096)
        except Exception as img_err:
            # Clean up saved image if preprocessing fails
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                "success": False,
                "error": f"Image preprocessing failed: {str(img_err)}"
            }), 400

        # 4. Predict Disease using loaded Random Forest model
        try:
            prediction = model.predict(x)[0]
            probabilities = model.predict_proba(x)[0]
        except Exception as infer_err:
            return jsonify({
                "success": False,
                "error": f"Model prediction failed: {str(infer_err)}"
            }), 500

        # Map classes dynamically using model.classes_ for safe index alignment
        # Class mappings: 0 -> NORMAL, 1 -> PNEUMONIA, 2 -> TUBERCULOSIS
        class_indices = {cls: idx for idx, cls in enumerate(model.classes_)}
        class_labels = {
            0: "NORMAL",
            1: "PNEUMONIA",
            2: "TUBERCULOSIS"
        }
        
        confidences = {
            "NORMAL": 0.0,
            "PNEUMONIA": 0.0,
            "TUBERCULOSIS": 0.0
        }

        # Safe probability assignment
        for cls_val, label in class_labels.items():
            if cls_val in class_indices:
                idx = class_indices[cls_val]
                confidences[label] = round(float(probabilities[idx]) * 100, 1)

        # Translate numerical prediction label to disease string name
        # If prediction corresponds to an unrecognized class, default to highest confidence category
        if int(prediction) in class_labels:
            disease_name = class_labels[int(prediction)]
        else:
            disease_name = max(confidences, key=confidences.get)

        # Return JSON prediction response
        return jsonify({
            "success": True,
            "disease": disease_name,
            "confidences": confidences,
            "image_path": f"static/uploads/{filename}"
        })

    except Exception as server_err:
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(server_err)}"
        }), 500

if __name__ == '__main__':
    # Run development server locally
    app.run(host='0.0.0.0', port=5000, debug=True)
