# scripts/predict_risk.py

import pandas as pd
import joblib
import os

# Paths
MODEL_PATH = "data\\flood_model_small.pkl"

# Load trained model
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# Example: Input new data
# Replace these values with actual inputs
input_data = {
    "MonsoonIntensity": [0.8],
    "TopographyDrainage": [0.5],
    "RiverManagement": [0.6],
    "Deforestation": [0.2],
    "Urbanization": [0.7],
    "ClimateChange": [0.9],
    "DamsQuality": [0.5],
    "Siltation": [0.3],
    "AgriculturalPractices": [0.4],
    "Encroachments": [0.2],
    "IneffectiveDisasterPreparedness": [0.6],
    "DrainageSystems": [0.5],
    "CoastalVulnerability": [0.7],
    "Landslides": [0.3],
    "Watersheds": [0.6],
    "DeterioratingInfrastructure": [0.5],
    "PopulationScore": [0.8],
    "WetlandLoss": [0.2],
    "InadequatePlanning": [0.4],
    "PoliticalFactors": [0.5]
}

# Convert to DataFrame
df_input = pd.DataFrame(input_data)

# Predict flood risk class
predicted_class = model.predict(df_input)

print(f"Predicted Flood Risk Class: {predicted_class[0]}")

