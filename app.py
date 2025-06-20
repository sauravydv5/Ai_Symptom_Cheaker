from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and feature names
model, feature_names = joblib.load("disease_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms')

        print("✅ Received symptoms:", symptoms)
        print("🔢 Sum of 1s:", sum(symptoms))

        # Check input length matches model feature length
        if len(symptoms) != len(feature_names):
            return jsonify({
                "error": f"Expected {len(feature_names)} features, got {len(symptoms)}"
            }), 400

        # Create DataFrame and predict
        input_df = pd.DataFrame([symptoms], columns=feature_names)
        prediction = model.predict(input_df)

        return jsonify({ "disease": prediction[0] })

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({ "error": str(e) }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT env var or fallback to 5000
    app.run(debug=False, host="0.0.0.0", port=port)
