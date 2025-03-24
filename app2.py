from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and encoder
model = pickle.load(open("fare_model.pkl", "rb"))  # Load trained model
encoder = pickle.load(open("encoder.pkl", "rb"))  # Load encoder

# Define car types available in dataset
car_types = ["Sedan", "SUV", "Hatchback", "Luxury"]  # Update as per dataset

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Extract form data
            distance = float(request.form["distance"])
            traffic = request.form["traffic"]
            time_of_day = request.form["time_of_day"]
            car_type = request.form["car_type"]

            # Prepare input DataFrame
            input_data = pd.DataFrame([[distance, traffic, time_of_day, car_type]], 
                                      columns=["distance", "Traffic", "Time_of_Day", "Type_of_Car"])

            # One-hot encode categorical features
            input_encoded = encoder.transform(input_data[["Traffic", "Time_of_Day", "Type_of_Car"]])
            input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(["Traffic", "Time_of_Day", "Type_of_Car"]))

            # Merge numerical & encoded categorical features
            final_input = pd.concat([input_data[["distance"]], input_encoded_df], axis=1)

            # Make prediction
            prediction = model.predict(final_input)

            # ✅ Fix: Format fare to 2 decimal places
            formatted_fare = f"{prediction[0]:.2f}"

            return render_template("index2.html", prediction_text=f"Estimated Fare: $ {formatted_fare}", car_types=car_types)

        except Exception as e:
            return render_template("index2.html", prediction_text="Error in prediction. Check inputs.", car_types=car_types)

    return render_template("index2.html", car_types=car_types)

if __name__ == "__main__":
    app.run(debug=True)
