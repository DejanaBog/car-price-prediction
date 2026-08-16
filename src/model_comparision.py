import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    split_features_and_target,
    build_preprocessor
)

DATA_PATH = "data/cars_cleaned_with_features.csv"

df = pd.read_csv(DATA_PATH)

# razdvajanje podataka i podjela na trening i test skup

X, y = split_features_and_target(df)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# definisanje algoritama koje želimo da uporedimo

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# treniranje i evaluacija modela u petlji

results = []

for model_name, regressor in models.items():

    model = Pipeline(
        steps = [
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor)
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error (y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    results.append({
        "model": model_name,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    })

# prikaz i analiza rezultata

results_df = pd.DataFrame(results)

# rezultate sortiramo prema MAE metrici

results_df = results_df.sort_values(
    by = "mae",
    ascending=True

)

results_df = results_df.round(2)
print(results_df)