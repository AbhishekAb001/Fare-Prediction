import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv("updated_dataset_with_traffic_time.csv")  # Ensure the correct dataset file

# Check column names
print("Columns in dataset:", df.columns)

# Selecting required columns
features = ['distance', 'Traffic', 'Time_of_Day', 'Type_of_Car']
X = df[features]
y = df['price']  # Assuming 'price' is the fare column

# One-Hot Encoding categorical variables
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # ✅ FIXED
X_encoded = encoder.fit_transform(X[['Traffic', 'Time_of_Day', 'Type_of_Car']])

# Convert to DataFrame
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['Traffic', 'Time_of_Day', 'Type_of_Car']))
X_final = pd.concat([X[['distance']], X_encoded_df], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train the Ridge Regression model
model = Ridge()
model.fit(X_train, y_train)

# Save model and encoder
pickle.dump(model, open("fare_model.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))

print("✅ Model training complete. Model and encoder saved!")
