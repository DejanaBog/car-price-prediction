import pandas as pd
from datetime import datetime

# 1. Kreiranje nove kolone car_age

def _create_car_age(df: pd.DataFrame)-> pd.DataFrame:

    df = df.copy()
    trenutna_godina = datetime.now().year

    df["car_age"] = trenutna_godina - df["year"]

    return df

# 2.Kreiranje kolone mileage_per_year

def _create_mileage_per_year(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["mileage_per_year"] = df["mileage_kilometers"]/df["car_age"].replace(0,1)

    return df

# 3. Kreiranje kolone engine_volume_liters - zapremina motora u litrima
def _create_engine_volume_liters (df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["engine_volume_liters"] = df["volume_cm3"] / 1000

    return df

# 4. Kreiranje kolone s indikatorom da li je automobil novijeg godišta

def _create_is_newer_car (df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["is_newer_car"] = (df["year"] >= 2018).astype(int)

    return df

# 5. Kreiranje kolone s indikatorom da li automobil ima veliku kilometražu

def _create_is_high_mileage (df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["is_high_mileage"] = (df["mileage_kilometers"] >= 200000).astype(int)

    return df

# 6. Kreiranje kolone s kombinacijom marke i modela

def _create_brand_model(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["brand_model"] = (df["make"] + "_" + df["model"])

    return df

# Povezivanje svih koraka u pipeline za inženjering karakteristika

def build_features(df: pd.DataFrame) -> pd.DataFrame:

    df_features = (
        df
        .pipe(_create_car_age)
        .pipe(_create_mileage_per_year)
        .pipe(_create_engine_volume_liters)
        .pipe(_create_is_newer_car)
        .pipe(_create_is_high_mileage)
        .pipe(_create_brand_model)
        .reset_index(drop = True)
    )

    return df_features

# Dodavanje putanje

CLEANED_DATA_PATH = "data/cars_cleaned.csv"
FEATURES_DATA_PATH = "data/cars_cleaned_with_features.csv"

# Pokretanje pipeline za inženjering karakteristika

def main() -> None:

    print("Loading cleaned dataset...")

    df_cleaned = pd.read_csv(CLEANED_DATA_PATH)

    print(" Building features...")

    df_features = build_features(df_cleaned)

    print("Saving feature dataset...")

    df_features.to_csv(FEATURES_DATA_PATH, index = False)

    print (f"Feature dataset saved to: {FEATURES_DATA_PATH}")

if __name__ == "__main__":
    main()
    