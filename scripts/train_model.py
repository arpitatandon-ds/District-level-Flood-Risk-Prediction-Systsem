import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent   
PROCESSED_DATA_PATH = BASE_DIR / "data" / "flood_processed.csv"
MODEL_PATH = BASE_DIR / "app" / "data" / "flood_model.pkl"


# Load processed data
df = pd.read_csv(PROCESSED_DATA_PATH)

# Features and target
X = df.drop('FloodProbability', axis=1)
y = df['FloodProbability']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.3f}")
print(f"R2 Score: {r2:.3f}")

# Save trained model
joblib.dump(model, MODEL_PATH)
print(f"Trained regression model saved at {MODEL_PATH}")

