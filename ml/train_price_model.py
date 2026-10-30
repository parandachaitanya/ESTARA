
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
CSV_FILE = ROOT / "data" / "Hyderbad_House_price.csv"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "price_model.joblib"
METRICS_FILE = MODEL_DIR / "price_model_metrics.json"


def extract_bhk(title):
    match = re.search(r"\b(\d+)\s*BHK\b", str(title), re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_property_type(title):
    title = str(title).lower()

    if "apartment" in title or "flat" in title:
        return "Apartment"
    if "villa" in title:
        return "Villa"
    if "house" in title:
        return "Independent House"
    if "plot" in title:
        return "Plot"

    return "Other"


def main():
    if not CSV_FILE.is_file():
        raise FileNotFoundError(f"Dataset not found: {CSV_FILE}")

    df = pd.read_csv(CSV_FILE)

    required = {"title", "location", "price(L)", "area_insqft"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")

    df["location"] = df["location"].astype("string").str.strip()

    df["area_insqft"] = pd.to_numeric(
        df["area_insqft"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    df["price(L)"] = pd.to_numeric(
        df["price(L)"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    df = df.dropna(
        subset=["location", "area_insqft", "price(L)", "title"]
    )

    df = df[
        (df["location"] != "")
        & (df["area_insqft"] > 0)
        & (df["price(L)"] > 0)
    ].copy()

    if len(df) < 10:
        raise ValueError("Not enough valid rows to train the model.")

    df["bhk"] = df["title"].apply(extract_bhk)
    df["property_type"] = df["title"].apply(extract_property_type)

    if "building_status" in df.columns:
        df["building_status"] = (
            df["building_status"]
            .fillna("Not specified")
            .astype(str)
            .str.strip()
        )
    else:
        df["building_status"] = "Not specified"

    # Convert price from lakhs to rupees.
    y = df["price(L)"] * 100_000

    numeric_features = ["area_insqft", "bhk"]
    categorical_features = [
        "location",
        "property_type",
        "building_status",
    ]

    X = df[numeric_features + categorical_features]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    metrics = {
        "model": "RandomForestRegressor",
        "dataset_rows_used": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "mae_rupees": round(float(mae), 2),
        "r2_score": round(float(r2), 4),
        "target": "Listing price in rupees",
        "warning": (
            "Evaluation uses a random holdout of listing data. "
            "It does not establish future market accuracy."
        ),
    }

    METRICS_FILE.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nESTARA ML MODEL TRAINED")
    print("-----------------------")
    print(f"Valid rows: {len(df)}")
    print(f"MAE: Rs. {mae:,.2f}")
    print(f"R2 score: {r2:.4f}")
    print(f"Model saved: {MODEL_FILE}")
    print(f"Metrics saved: {METRICS_FILE}")
    print("\nThese are dataset evaluation results,")
    print("not a guarantee of actual market prices.")


if __name__ == "__main__":
    main()
