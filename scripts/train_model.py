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
MODEL_PATH = BASE_DIR / "app" / "data" / "flood_model_small.pkl"


# Load processed data
df = pd.read_csv(PROCESSED_DATA_PATH)

# Features and target
X = df.drop('FloodProbability', axis=1)
y = df['FloodProbability']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Initialize a smaller Random Forest model
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.5f}")
print(f"R2 Score: {r2:.5f}")

# Save as a separate file; preserve original model
SMALL_MODEL_PATH = BASE_DIR / "app" / "data" / "flood_model_small.pkl"
SMALL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, SMALL_MODEL_PATH, compress=3)

size_mb = SMALL_MODEL_PATH.stat().st_size / (1024 * 1024)

print(f"Smaller model saved at: {SMALL_MODEL_PATH}")
print(f"Compressed model size: {size_mb:.2f} MB")


