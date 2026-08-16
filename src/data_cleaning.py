# Uvoženje potrebnih biblioteka
import re
import pandas as pd

RAW_DATA_PATH = "data/cars.csv"
CLEANED_DATA_PATH = "data/cars_cleaned.csv"

# 1. Standardizacija naziva kolona (snake_case)

def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    new_columns = []

    for col in df.columns:
        clean_col = col.strip().lower()

        clean_col = clean_col.replace("(", "_")
        clean_col = clean_col.replace(")", "")
        clean_col = clean_col.replace("-", "_")

        clean_col = re.sub(r"\s+", "_", clean_col)
        clean_col = re.sub(r"[^a-z0-9_]", "", clean_col)
        clean_col = re.sub(r"_+", "_", clean_col)
        clean_col = clean_col.strip("_")

        new_columns.append(clean_col)

    df.columns = new_columns

    return df

# 2. Uklanjanje viška razmaka iz tekstualnih vrijednosti

def _strip_string_values(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    text_columns = df.select_dtypes(include=["str"]).columns

    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()
            
    return df

# 3. Standardizacija nedostajućih vrijednosti

MISSING_LIKE_VALUES = {
    "",
    " ",
    "nan",
    "NaN",
    "NAN",
    "null",
    "Null",
    "NULL",
    "none",
    "None",
    "NONE"
}

def _replace_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df = df.replace(list(MISSING_LIKE_VALUES),pd.NA)

    return df

# 4. numeričke kolone su tipa int64 i float64; ipak će se izvršiti konverzija zbog neke naredne verzije fajla

def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    numeric_columns = [
        "year",
        "mileage_kilometers",
        "volume_cm3"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col],errors="coerce")

    return df

# 5. Standardizacija kategorijskih vrijednosti
def _clean_categorical_values(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    categorical_columns = [
        "make",
        "model",
        "condition",
        "fuel_type",
        "color",
        "transmission",
        "drive_unit",
        "segment"
    ]

    for col in categorical_columns:

        if col in df.columns:
            df[col] = (df[col].astype("string").str.strip().str.lower())

    return df

# 6.Uklanje duplih redova
def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df = df.drop_duplicates().reset_index(drop = True)

    return df

# 7. Uklanjanje nevalidnih redova
def _remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    uslovi = (
        (df["year"]>=1900) & (df["year"] <=2026) &
        (df["mileage_kilometers"] > 0) & (df["mileage_kilometers"] < 1000000) &
        (df["volume_cm3"] >=500) & (df["volume_cm3"] <=7000)
    )  

    df = df[uslovi].reset_index(drop = True)

    return df

# Povezivanje svih operacija u jedan pipeline za čišćenje

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = (
        df
        .pipe(_standardize_column_names)
        .pipe(_strip_string_values)
        .pipe(_replace_missing_values)
        .pipe(_convert_numeric_columns)
        .pipe(_clean_categorical_values)
        .pipe(_remove_duplicates)
        .pipe(_remove_invalid_rows)
        .reset_index(drop = True)
    )

    return df_clean

# Pokretanje pipeline za čišćenje podataka
def main() -> None:
    print("Loading raw dataset...")

    df_raw = pd.read_csv(RAW_DATA_PATH)

    print("Cleaning dataset...")

    df_cleaned = clean(df_raw)

    print("Saving cleaned dataset...")

    df_cleaned.to_csv(CLEANED_DATA_PATH, index = False)

    print(f"Cleaned dataset saved to: {CLEANED_DATA_PATH}")

    print(f"Shape of final dataset: {df_cleaned.shape}")

if __name__ == "__main__":
    main()