import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    TARGET_COLUMN,
    split_features_and_target,
    build_preprocessor)

DATA_PATH = "data/cars_cleaned_with_features.csv"
MODEL_PATH = "models/random_forest_model.joblib"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# razdvajanje ulaznih karakteristika i ciljne promjenljive

print("Splitting features and target...")
X, y = split_features_and_target(df)

print(X.shape)
print(y.shape)

# podjela podataka na trening i test skup

print("Splitting data into training and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# kreiranje kompletnog ML pipelinea

print("Creating model pipeline...")

model = Pipeline(
    steps = [
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ] 
)

# treniranje prvog regresionog modela

print("Training model...")
model.fit(X_train, y_train)

# čuvanje treniranog modela

print("Saving model...")
joblib.dump(model, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")

# pravljenje prvih predikcija