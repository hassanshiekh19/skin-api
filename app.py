from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

# Load trained model
model = load_model('skin_disease_model.h5')

# Make sure this exactly matches your model's output classes
CATEGORIES = ['Acne', 'Eczema', 'Psoriasis', 'Melanoma', 'BCC', 'Rosacea', 'Warts']

@app.route('/')
def home():
    return 'Skin Disease Prediction API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        # Convert bytes to numpy array, then decode to image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Resize image to 128x128 (adjust if your model expects different size)
        img = cv2.resize(img, (128, 128))

        # Normalize pixel values to [0,1]
        img = img.astype('float32') / 255.0

        # Expand dimensions to match model input shape (1, 128, 128, 3)
        img = np.expand_dims(img, axis=0)

        # Model prediction
        prediction = model.predict(img)
        pred_index = np.argmax(prediction)
        predicted_label = CATEGORIES[pred_index]
        confidence = float(prediction[0][pred_index]) * 100

        return jsonify({
            'prediction': predicted_label,
            'confidence': round(confidence, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
