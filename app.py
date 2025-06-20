from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file variables

app = Flask(__name__)
CORS(app)

model, feature_names = joblib.load("disease_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symptoms = data['symptoms']

        print("Received symptoms:", symptoms)
        print("Sum of 1s:", sum(symptoms))

        if len(symptoms) != len(feature_names):
            return jsonify({
                "error": f"Expected {len(feature_names)} features, got {len(symptoms)}"
            }), 400

        input_df = pd.DataFrame([symptoms], columns=feature_names)
        prediction = model.predict(input_df)

        return jsonify({ "disease": prediction[0] })

    except Exception as e:
        return jsonify({ "error": str(e) }), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug_mode = bool(int(os.getenv("FLASK_DEBUG", 1)))
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
