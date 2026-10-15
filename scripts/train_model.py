import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DATA_PATH = BASE_DIR / "data" / "flood_processed.csv"
MODEL_PATH = BASE_DIR / "app" / "data" / "flood_model_small.pkl"

# Load processed dataset
df = pd.read_csv(PROCESSED_DATA_PATH)

# Use all 20 features
X = df.drop("FloodProbability", axis=1)
y = df["FloodProbability"]

print("Total features:", X.shape[1])
print("Feature columns:")
print(X.columns.tolist())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Random Forest model
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.5f}")
print(f"R2 Score: {r2:.5f}")

# Feature importance ranking
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance Ranking:")
print(feature_importance.to_string())

# Save model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH, compress=3)

size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)

print(f"\nModel saved at: {MODEL_PATH}")
print(f"Compressed model size: {size_mb:.2f} MB")
print("Total features in saved model:", len(model.feature_names_in_))

