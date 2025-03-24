from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests (e.g., Flutter app)

# Load the trained model and encoder safely
model_path = "fare_model.pkl"
encoder_path = "encoder.pkl"

if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    raise FileNotFoundError("Model or encoder file is missing!")

model = pickle.load(open(model_path, "rb"))
encoder = pickle.load(open(encoder_path, "rb"))

# ✅ Add a Home Route to Prevent 404 Errors
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fare Prediction API is Running!"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure request is in JSON format
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid input. JSON required"}), 400

        # Validate required fields
        required_fields = ["distance", "traffic", "time_of_day"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Convert and validate distance
        try:
            distance = float(data["distance"])
        except ValueError:
            return jsonify({"error": "Invalid distance value"}), 400

        traffic = data["traffic"]
        time_of_day = data["time_of_day"]

        # Prepare input DataFrame
        input_data = pd.DataFrame([[distance, traffic, time_of_day]], 
                                  columns=["distance", "Traffic", "Time_of_Day"])

        # One-hot encode categorical features
        input_encoded = encoder.transform(input_data[["Traffic", "Time_of_Day"]])
        input_encoded_df = pd.DataFrame(input_encoded, 
                                        columns=encoder.get_feature_names_out(["Traffic", "Time_of_Day"]))

        # Merge numerical & encoded categorical features
        final_input = pd.concat([input_data[["distance"]], input_encoded_df], axis=1)

        # Make prediction
        prediction = model.predict(final_input)

        # Format fare to 2 decimal places
        formatted_fare = round(float(prediction[0]), 2)

        return jsonify({"estimated_fare": formatted_fare}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Run the Flask app (for local development)
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
