
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "price_model.joblib"


def predict_listing_price(
    area_insqft,
    bhk,
    location,
    property_type,
    building_status,
):
    """Estimate a property's listing price using the trained ML model."""

    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            "ML model not found. Run ml/train_price_model.py first."
        )

    if area_insqft <= 0:
        raise ValueError("Area must be greater than zero.")

    if bhk is not None and bhk <= 0:
        raise ValueError("BHK must be greater than zero.")

    model = joblib.load(MODEL_FILE)

    property_data = pd.DataFrame([{
        "area_insqft": float(area_insqft),
        "bhk": bhk,
        "location": str(location).strip(),
        "property_type": str(property_type).strip(),
        "building_status": str(building_status).strip(),
    }])

    predicted_price = float(model.predict(property_data)[0])

    return {
        "estimated_price_rupees": round(max(0, predicted_price), 2),
        "estimated_price_lakhs": round(max(0, predicted_price) / 100000, 2),
        "currency": "INR",
        "message": (
            "This is an experimental estimate based on the available "
            "listing dataset, not a guaranteed market price."
        ),
    }
