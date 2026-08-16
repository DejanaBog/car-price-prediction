import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import (
    
split_features_and_target
    
)

DATA_PATH = "data/cars_cleaned_with_features.csv"
MODEL_PATH = "models/random_forest_model.joblib"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Splitting features and target...")

X,y = split_features_and_target(df)

print("Creating the same train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Loading saved model...")

loaded_model = joblib.load(MODEL_PATH)

# pravljenje predikcija nad test skupom
print("Making predictions")

y_pred = loaded_model.predict(X_test)
print(y_pred[:10])

# računanje metrika regresije
print("Calculating regression metrics...")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

metrics = pd.DataFrame({
    "metric": ["MAE","MSE","RMSE","R2"],
    "value": [mae, mse, rmse, r2]
})

print("\nRegression metrics:")
print(metrics)

print("\nCreating prediction analysis table...")

prediction_analysis = pd.DataFrame({
    "actual_price": y_test.values,
    "predicted_price": y_pred
})

prediction_analysis["error_price"]=(
    prediction_analysis["actual_price"]
    - prediction_analysis["predicted_price"]
)

prediction_analysis["absolute_error_price"] = (
    prediction_analysis["error_price"].abs()
)

print("\nPrediction examples:")
print(prediction_analysis.sample(10, random_state=42))

# prikaz najvećih grešaka modela
print("\nLargest prediction errors:")

print(
    prediction_analysis
    .sort_values("absolute_error_price", ascending=False)
    .head(10)
)
