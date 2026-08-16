
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

TARGET_COLUMN = "priceusd"

# Definisanje numeričkih karakteristika

NUMERIC_FEATURES = [
    "year",
    "mileage_kilometers",
    "volume_cm3",
    "car_age",
    "mileage_per_year",
    "engine_volume_liters"
    ]

# Definisanje kategorijskih karakteristika

CATEGORICAL_FEATURES = [
    "make",
    "model",
    "condition",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment",
    "brand_model"
]

# Definisanje binarnih karakteristika

BINARY_FEATURES = [
    "is_newer_car",
    "is_high_mileage"
]

# Pomoćna funkcija koja vraća sve ulazne kolone koje će model koristiti

def get_all_feature_columns() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# Razdvajanje ulaznih karakteristika i ciljne promjenljive

def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[get_all_feature_columns()].copy()
    y = df[TARGET_COLUMN].copy()

    return X,y

# Pretprocesiranje numeričkih kolona
def _build_numeric_transformer() -> Pipeline:
     numeric_transformer = Pipeline(steps =[
             ("imputer", SimpleImputer(missing_values=pd.NA, strategy = "median")),
             ("scaler", StandardScaler())

          ])

     return numeric_transformer

# Pretprocesiranje kategorijskih kolona

def _build_categorical_transformer() -> Pipeline:

    categorical_transformer = Pipeline(
        steps = [
            ("imputer", SimpleImputer(missing_values=pd.NA, strategy = "most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    return categorical_transformer

# Pretprocesiranje kolona sa indikatorima (binarne kolone)
# Binarne kolone nemaju nedostajućih vrijednosti 

def _build_binary_transformer():
    return "passthrough"

# Spajanje transformera pomoću ColumnTransformera

def build_preprocessor() -> ColumnTransformer:

    preprocessor = ColumnTransformer(
        transformers =[
        ("num", _build_numeric_transformer(), NUMERIC_FEATURES),
        ("cat", _build_categorical_transformer(), CATEGORICAL_FEATURES),
        ("bin", _build_binary_transformer(), BINARY_FEATURES),
        ],
        remainder = "drop"
    
    )
    return preprocessor
    