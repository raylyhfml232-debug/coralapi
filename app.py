# -*- coding: utf-8 -*-
"""
Coral Reef Classifier API
"""
# 1. Library imports
import uvicorn
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
import urllib.request

# 2. Create the app object
app = FastAPI(title="Coral Reef Classifier", 
              description="Classify coral images as healthy, bleached, or dead",
              version="1.0.0")

# Download model from GitHub Releases if not present
model_path = 'model/coral_final_model.keras'
if not os.path.exists(model_path):
    print("📥 Downloading model from GitHub Releases...")
    os.makedirs('model', exist_ok=True)
    url = "https://github.com/raylyhfml232-debug/coralapi/releases/download/v1.0.0/coral_final_model.keras"
    urllib.request.urlretrieve(url, model_path)
    print("✅ Model downloaded!")

# Load your trained model
print("Loading model...")
model = tf.keras.models.load_model('model/coral_final_model.keras')
with open('model/class_indices.json', 'r') as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}
print(f"Model loaded! Classes: {list(idx_to_class.values())}")

# 4. Home route
@app.get('/')
def index():
    return {
        'message': '🌊 Coral Reef Classifier API',
        'endpoints': {
            '/': 'This message',
            '/health': 'Health check',
            '/predict': 'POST an image file to classify'
        }
    }

# 5. Health check
@app.get('/health')
def health():
    return {'status': 'healthy', 'model_loaded': True}

# 6. Prediction endpoint
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """
    Upload a coral image to classify its health status
    """
    try:
        # Read image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        predicted_class = idx_to_class[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        # Return result
        return {
            "success": True,
            "filename": file.filename,
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                "healthy": float(predictions[0]),
                "bleached": float(predictions[1]),
                "dead": float(predictions[2])
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# 7. Run the API
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
