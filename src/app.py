from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pickle
import os

print("Initializing Flask app...")
app = Flask(__name__)
CORS(app)

print("Loading model and vectorizer...")
try:
    model = pickle.load(open('models/phishing_model.pkl', 'rb'))
    print("✓ Model loaded")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

try:
    vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
    print("✓ Vectorizer loaded")
except Exception as e:
    print(f"✗ Error loading vectorizer: {e}")
    vectorizer = None

@app.route('/')
def home():
    html_path = os.path.join(os.path.dirname(__file__), 'index.html')
    return send_file(html_path, mimetype='text/html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or vectorizer is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.json
        email_text = data.get('email_text', '')
        
        if not email_text:
            return jsonify({"error": "No email text provided"}), 400
        
        # Vectorize
        email_transformed = vectorizer.transform([email_text])
        
        # Predict
        prediction = model.predict(email_transformed)[0]
        probability = model.predict_proba(email_transformed)[0]
        
        result = {
            "is_phishing": bool(prediction),
            "confidence": float(max(probability)),
            "probability_safe": float(probability[0]),
            "probability_phishing": float(probability[1])
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Flask server on http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)