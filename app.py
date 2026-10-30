import csv
import re
import json
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from backend.price_prediction import predict_listing_price
from backend.intelligence import parse_property_query, rank_properties
from backend.vision_ai import load_image, find_similar_properties
from backend.condition_ai import analyze_property_image


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT / ".env")

DATA_FILE = ROOT / "data" / "properties.json"
CSV_FILE = ROOT / "data" / "Hyderbad_House_price.csv"
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_FILE = ROOT / "data" / "estara_messages.db"

app = Flask(
    __name__,
    template_folder="frontend",
    static_folder="frontend",
    static_url_path="/static"
)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


# --------------------------------------------------
# ILLUSTRATIVE PROPERTY IMAGES
# These are sample images, not verified listing photos.
# --------------------------------------------------

PROPERTY_IMAGES = [
    "https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?auto=format&fit=crop&w=800&q=80",
    "https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?auto=format&fit=crop&w=800&q=80",
    "https://images.unsplash.com/photo-1600566753086-00f18fb6b3ea?auto=format&fit=crop&w=800&q=80",
    "https://images.unsplash.com/photo-1600047509807-ba8f99d2cdde?auto=format&fit=crop&w=800&q=80",
    "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=800&q=80",
]


# --------------------------------------------------
# BUYER-SELLER MESSAGING DATABASE
# --------------------------------------------------

def get_db_connection():
    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_message_database():
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with get_db_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                property_id TEXT NOT NULL,
                property_name TEXT NOT NULL,
                sender_name TEXT NOT NULL,
                sender_email TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


initialize_message_database()


# --------------------------------------------------
# PROPERTY DATA
# --------------------------------------------------

def format_price_label(price):
    if price >= 10_000_000:
        return f"₹{price / 10_000_000:.2f} crore"

    if price >= 100_000:
        return f"₹{price / 100_000:.2f} lakh"

    return f"₹{price:,.0f}"


def load_kaggle_properties():
    properties = []

    with CSV_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row_number, row in enumerate(reader, start=1):
            try:
                title = (row.get("title") or "").strip()
                location = (row.get("location") or "").strip()

                if not title or not location:
                    continue

                price_lakhs = float(
                    (row.get("price(L)") or "")
                    .replace(",", "")
                    .strip()
                )

                area = float(
                    (row.get("area_insqft") or "")
                    .replace(",", "")
                    .strip()
                )

                if price_lakhs <= 0 or area <= 0:
                    continue

                price = round(price_lakhs * 100_000)

                bhk_match = re.search(
                    r"\b(\d+)\s*BHK\b",
                    title,
                    re.IGNORECASE
                )

                bhk = (
                    int(bhk_match.group(1))
                    if bhk_match
                    else None
                )

                title_lower = title.lower()

                if "apartment" in title_lower or "flat" in title_lower:
                    property_type = "Apartment"
                elif "villa" in title_lower:
                    property_type = "Villa"
                elif "house" in title_lower:
                    property_type = "Independent House"
                elif "plot" in title_lower:
                    property_type = "Plot"
                else:
                    property_type = title

                status = (
                    row.get("building_status") or ""
                ).strip()

                image_url = PROPERTY_IMAGES[
                    (row_number - 1) % len(PROPERTY_IMAGES)
                ]

                rate_text = (
                    row.get("rate_persqft") or "0"
                ).replace(",", "").strip()

                properties.append({
                    "id": f"KG-{row_number:05d}",
                    "name": title,
                    "location": location,
                    "price": price,
                    "priceLabel": format_price_label(price),
                    "type": property_type,
                    "bhk": bhk,
                    "area": area,
                    "areaSqFt": area,
                    "ratePerSqFt": float(rate_text or 0),
                    "buildingStatus": status,
                    "description": (
                        "Listing sourced from the Hyderabad "
                        "house-price dataset. "
                        f"Building status: {status or 'Not specified'}. "
                        "Verify property details with the seller."
                    ),
                    "features": [],
                    "image": image_url,
                    "imageIsIllustrative": True
                })

            except (ValueError, TypeError):
                continue

    return properties


def load_properties():
    if CSV_FILE.is_file():
        try:
            kaggle_properties = load_kaggle_properties()

            if kaggle_properties:
                return kaggle_properties

        except (OSError, csv.Error, UnicodeError) as error:
            app.logger.error(
                "Could not load Kaggle dataset: %s",
                error
            )

    try:
        with DATA_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("properties.json must contain a JSON list.")

        return data

    except FileNotFoundError:
        app.logger.exception("Property data files were not found.")
        return []


def find_property(property_id):
    for property_info in load_properties():
        current_id = str(
            property_info.get(
                "id",
                property_info.get("_id", "")
            )
        )

        if current_id == str(property_id):
            return property_info

    return None


# --------------------------------------------------
# WEBSITE PAGES
# --------------------------------------------------

@app.get("/")
def home():
    return render_template("discover.html")


@app.get("/search")
def search_page():
    return render_template("search.html")


@app.get("/results")
def results_page():
    return render_template("results.html")


@app.get("/property-details")
def details_page():
    return render_template("property-details.html")


@app.get("/inbox")
def inbox_page():
    return render_template("inbox.html")


@app.get("/compare")
def compare_page():
    return render_template("compare.html")


@app.get("/future-insights")
def future_insights_page():
    return render_template("future-insights.html")


@app.get("/vision-ai")
def vision_ai_page():
    return render_template("vision-ai.html")


@app.get("/condition-ai")
def condition_ai_page():
    return render_template("condition-ai.html")


# --------------------------------------------------
# VISIONAI: VISUAL PROPERTY SEARCH
# Requires backend/vision_ai.py and its dependencies.
# --------------------------------------------------

@app.post("/api/vision-search")
def api_vision_search():
    uploaded = request.files.get("image")

    if uploaded is None or not uploaded.filename:
        return jsonify({
            "error": "Please upload a property image."
        }), 400

    filename = secure_filename(uploaded.filename)

    extension = (
        filename.rsplit(".", 1)[-1].lower()
        if "." in filename else ""
    )

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": "Upload a PNG, JPG, JPEG, or WEBP image."
        }), 400

    try:
        query_image = load_image(uploaded.stream)
        properties = load_properties()

        if not properties:
            return jsonify({
                "error": "No property listings are available."
            }), 503

        matches = find_similar_properties(
            query_image,
            properties,
            top_k=5
        )

        if not matches:
            return jsonify({
                "error": (
                    "No property images could be compared. "
                    "Check your image URLs and internet connection."
                )
            }), 503

        return jsonify({
            "success": True,
            "count": len(matches),
            "matches": [
                {
                    **match["property"],
                    "similarity": match["similarity"]
                }
                for match in matches
            ],
            "notice": (
                "Results are ranked by visual similarity. "
                "They do not indicate property quality or value. "
                "Some listing images may be illustrative."
            )
        })

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except OSError:
        return jsonify({
            "error": "The image could not be processed."
        }), 400

    except Exception:
        app.logger.exception("VisionAI search failed")

        return jsonify({
            "error": (
                "VisionAI search failed. "
                "Check the Flask terminal for details."
            )
        }), 500


# --------------------------------------------------
# PROPERTY CONDITION AI
# Prototype image-quality screening, not damage detection.
# Requires backend/condition_ai.py and its dependencies.
# --------------------------------------------------

@app.post("/api/condition-ai")
def api_condition_ai():
    uploaded = request.files.get("image")

    if uploaded is None or not uploaded.filename:
        return jsonify({
            "error": "Please select a property image."
        }), 400

    filename = secure_filename(uploaded.filename)

    extension = (
        filename.rsplit(".", 1)[-1].lower()
        if "." in filename else ""
    )

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": "Upload a PNG, JPG, JPEG, or WEBP image."
        }), 400

    try:
        image_bytes = uploaded.read()
        result = analyze_property_image(image_bytes)
        return jsonify(result)

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("Property Condition AI failed")
        return jsonify({
            "error": "Analysis failed. Check the Flask terminal."
        }), 500


# --------------------------------------------------
# MACHINE LEARNING: PRICE PREDICTION
# --------------------------------------------------

@app.get("/price-prediction")
def price_prediction_page():
    return render_template("price-prediction.html")


@app.post("/api/predict-price")
def api_predict_price():
    payload = request.get_json(silent=True) or request.form

    try:
        area = float(payload.get("area_insqft", 0))

        if not (0 < area < 100_000_000):
            raise ValueError(
                "Area must be greater than zero and within a valid range."
            )

        bhk_value = payload.get("bhk")
        bhk = (
            int(bhk_value)
            if bhk_value not in (None, "")
            else None
        )

        if bhk is not None and not (1 <= bhk <= 100):
            raise ValueError("BHK must be between 1 and 100.")

        location = str(payload.get("location", "")).strip()
        property_type = str(
            payload.get("property_type", "Apartment")
        ).strip()
        building_status = str(
            payload.get("building_status", "Not specified")
        ).strip()

        if not location:
            return jsonify({
                "error": "Please enter a property location."
            }), 400

        if len(location) > 200:
            raise ValueError("Location must be 200 characters or fewer.")

        if not property_type or len(property_type) > 100:
            raise ValueError("Please enter a valid property type.")

        if not building_status or len(building_status) > 100:
            raise ValueError("Please enter a valid building status.")

        result = predict_listing_price(
            area_insqft=area,
            bhk=bhk,
            location=location,
            property_type=property_type,
            building_status=building_status,
        )

        return jsonify(result)

    except (TypeError, ValueError, OverflowError) as error:
        return jsonify({
            "error": f"Please check your inputs: {error}"
        }), 400

    except FileNotFoundError as error:
        return jsonify({
            "error": str(error)
        }), 503

    except Exception:
        app.logger.exception("ML price prediction failed")

        return jsonify({
            "error": "Prediction failed. Check the Flask terminal."
        }), 500


# --------------------------------------------------
# PROFIT / LOSS CALCULATOR API
# --------------------------------------------------

@app.post("/api/profit-loss")
def api_profit_loss():
    payload = request.get_json(silent=True) or {}

    def read_amount(field):
        value = payload.get(field)

        if value is None or isinstance(value, bool):
            raise ValueError(f"{field} is required.")

        amount = float(value)

        if not (0 <= amount < 1_000_000_000_000):
            raise ValueError(
                f"{field} must be a non-negative valid amount."
            )

        return amount

    try:
        purchase_price = read_amount("purchase_price")
        expected_sale_price = read_amount("expected_sale_price")
        purchase_costs = read_amount("purchase_costs")
        selling_costs = read_amount("selling_costs")

    except (ValueError, TypeError, OverflowError) as error:
        return jsonify({
            "error": f"Please check your inputs: {error}"
        }), 400

    if purchase_price <= 0:
        return jsonify({
            "error": "Purchase price must be greater than zero."
        }), 400

    total_investment = purchase_price + purchase_costs
    net_sale_proceeds = expected_sale_price - selling_costs
    profit_or_loss = net_sale_proceeds - total_investment

    return jsonify({
        "purchase_price": round(purchase_price, 2),
        "purchase_costs": round(purchase_costs, 2),
        "total_investment": round(total_investment, 2),
        "expected_sale_price": round(expected_sale_price, 2),
        "selling_costs": round(selling_costs, 2),
        "net_sale_proceeds": round(net_sale_proceeds, 2),
        "profit_or_loss": round(profit_or_loss, 2),
        "return_percentage": round(
            (profit_or_loss / total_investment) * 100, 2
        ),
        "result": (
            "profit" if profit_or_loss > 0
            else "loss" if profit_or_loss < 0
            else "break_even"
        ),
        "notice": (
            "This is a calculation based on your inputs, "
            "not a guarantee of future property prices."
        )
    })


# --------------------------------------------------
# PROPERTY APIs
# --------------------------------------------------

@app.get("/api/properties")
def api_properties():
    return jsonify(load_properties())


@app.post("/api/parse-query")
def api_parse_query():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", ""))[:500]

    parsed = parse_property_query(query)

    ranked = rank_properties(
        load_properties(),
        parsed
    )

    return jsonify({
        "filters": parsed,
        "properties": ranked
    })


@app.get("/api/recommendations")
def api_recommendations():
    preferences = {
        "query": request.args.get("query", ""),
        "location": request.args.get("location"),
        "propertyType": request.args.get("propertyType"),
        "bhk": request.args.get("bhk", type=int),
        "minBudget": request.args.get("minBudget", type=int),
        "maxBudget": request.args.get("maxBudget", type=int)
    }

    recommendations = rank_properties(
        load_properties(),
        preferences
    )

    return jsonify(recommendations)


# --------------------------------------------------
# BUYER-SELLER MESSAGING API
# Demo only: authentication is not implemented.
# --------------------------------------------------

@app.post("/api/messages")
def api_send_message():
    payload = request.get_json(silent=True) or {}

    property_id = str(payload.get("property_id", "")).strip()
    sender_name = str(payload.get("sender_name", "")).strip()
    sender_email = str(payload.get("sender_email", "")).strip()
    message = str(payload.get("message", "")).strip()

    if not all([
        property_id, sender_name, sender_email, message
    ]):
        return jsonify({
            "error": "Please fill in every field."
        }), 400

    if len(sender_name) > 100:
        return jsonify({
            "error": "Name must be 100 characters or fewer."
        }), 400

    if (
        len(sender_email) > 254
        or not re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            sender_email
        )
    ):
        return jsonify({
            "error": "Please enter a valid email address."
        }), 400

    if len(message) > 2000:
        return jsonify({
            "error": "Message must be 2000 characters or fewer."
        }), 400

    property_info = find_property(property_id)

    if property_info is None:
        return jsonify({
            "error": "Property not found. Please select a valid listing."
        }), 404

    property_name = str(
        property_info.get(
            "name",
            property_info.get("title", "Property")
        )
    )[:200]

    created_at = datetime.now(timezone.utc).isoformat()

    try:
        with get_db_connection() as connection:
            cursor = connection.execute("""
                INSERT INTO messages (
                    property_id, property_name, sender_name,
                    sender_email, message, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                property_id, property_name, sender_name,
                sender_email, message, created_at
            ))

            message_id = cursor.lastrowid

        return jsonify({
            "success": True,
            "message": "Your message has been saved.",
            "message_id": message_id
        }), 201

    except sqlite3.Error:
        app.logger.exception("Could not save buyer-seller message")

        return jsonify({
            "error": "Could not save your message. Please try again."
        }), 500


@app.get("/api/messages")
def api_list_messages():
    # DEMO ONLY: no login or authorization is implemented.
    # Do not expose real buyer contact information publicly.
    property_id = request.args.get("property_id", "").strip()

    try:
        with get_db_connection() as connection:
            if property_id:
                rows = connection.execute("""
                    SELECT id, property_id, property_name,
                           sender_name, sender_email,
                           message, created_at
                    FROM messages
                    WHERE property_id = ?
                    ORDER BY id DESC
                """, (property_id,)).fetchall()
            else:
                rows = connection.execute("""
                    SELECT id, property_id, property_name,
                           sender_name, sender_email,
                           message, created_at
                    FROM messages
                    ORDER BY id DESC
                """).fetchall()

        return jsonify([dict(row) for row in rows])

    except sqlite3.Error:
        app.logger.exception("Could not load messages")

        return jsonify({
            "error": "Could not load messages."
        }), 500


# --------------------------------------------------
# OPTIONAL PROPERTY IMAGE ANALYSIS
# Requires HF_TOKEN in .env.
# --------------------------------------------------

@app.post("/api/analyze-image")
def api_analyze_image():
    uploaded = request.files.get("image")

    if not uploaded or not uploaded.filename:
        return jsonify({
            "error": "Choose an image file first."
        }), 400

    filename = secure_filename(uploaded.filename)

    extension = (
        filename.rsplit(".", 1)[-1].lower()
        if "." in filename
        else ""
    )

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": "Use PNG, JPG, JPEG, or WEBP."
        }), 400

    token = os.getenv("HF_TOKEN", "").strip()

    if not token:
        return jsonify({
            "error": (
                "Image analysis is not configured. "
                "Set HF_TOKEN to enable this feature."
            )
        }), 503

    model_name = os.getenv(
        "HF_IMAGE_MODEL",
        "google/vit-base-patch16-224"
    )

    endpoint = (
        "https://api-inference.huggingface.co/models/"
        f"{model_name}"
    )

    try:
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {token}"},
            data=uploaded.read(),
            timeout=45
        )

        if not response.ok:
            app.logger.error(
                "Image provider returned HTTP %s",
                response.status_code
            )

            return jsonify({
                "error": "The image model could not process this image.",
                "providerStatus": response.status_code
            }), 502

        return jsonify({
            "model": model_name,
            "results": response.json(),
            "notice": (
                "Image labels are predictions, not a "
                "professional property inspection."
            )
        })

    except requests.RequestException:
        app.logger.exception("Image analysis request failed")

        return jsonify({
            "error": "The image analysis service could not be reached."
        }), 502


# --------------------------------------------------
# ESTARA AI PROPERTY ASSISTANT
# Requires GROQ_API_KEY in .env and the groq package.
# --------------------------------------------------

@app.post("/api/ai-property-assistant")
def api_ai_property_assistant():
    payload = request.get_json(silent=True) or {}

    question = str(payload.get("question", "")).strip()
    property_info = payload.get("property", {})

    if not question:
        return jsonify({
            "error": "Please enter a question."
        }), 400

    if len(question) > 1000:
        return jsonify({
            "error": "Keep your question under 1000 characters."
        }), 400

    if not isinstance(property_info, dict):
        return jsonify({
            "error": (
                "Property information is invalid. "
                "Please reload the page."
            )
        }), 400

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        app.logger.error(
            "GROQ_API_KEY is missing. Check the .env file beside app.py."
        )

        return jsonify({
            "error": (
                "ESTARA AI is not configured. Check GROQ_API_KEY "
                "in your .env file and restart Flask."
            )
        }), 503

    features = property_info.get("features", [])

    if not isinstance(features, list):
        features = []

    safe_property = {
        "name": str(
            property_info.get(
                "name", property_info.get("title", "Property")
            )
        )[:200],
        "location": str(
            property_info.get("location", "Not provided")
        )[:200],
        "type": str(
            property_info.get(
                "type",
                property_info.get("propertyType", "Not provided")
            )
        )[:100],
        "bhk": str(
            property_info.get("bhk", "Not provided")
        )[:30],
        "price": str(
            property_info.get(
                "priceLabel",
                property_info.get("price", "Not provided")
            )
        )[:100],
        "description": str(
            property_info.get("description", "")
        )[:1500],
        "features": [
            str(feature)[:150] for feature in features[:20]
        ]
    }

    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are ESTARA AI, a friendly property discovery "
                        "assistant. Answer in simple, clear language using "
                        "the supplied property details. Never invent property "
                        "facts, approvals, ownership details, market prices, "
                        "rental yields, or investment returns. If information "
                        "is missing, say so and explain how to verify it. "
                        "For legal, financial, structural, or ownership "
                        "questions, recommend checking with a qualified "
                        "professional. Keep answers useful and concise."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Selected property details:\n"
                        + json.dumps(safe_property, ensure_ascii=False)
                        + "\n\nUser question:\n"
                        + question
                    )
                }
            ],
            temperature=0.4,
            max_completion_tokens=700
        )

        answer = completion.choices[0].message.content

        if not answer:
            return jsonify({
                "error": "ESTARA AI returned an empty answer. Please try again."
            }), 502

        return jsonify({"answer": answer})

    except ImportError:
        app.logger.exception("Groq package is unavailable")

        return jsonify({
            "error": "Groq is not installed. Run python -m pip install groq"
        }), 500

    except Exception:
        app.logger.exception("ESTARA AI request failed")

        return jsonify({
            "error": (
                "ESTARA AI could not get a response. "
                "Check the Flask terminal for the error details."
            )
        }), 502


# --------------------------------------------------
# START THE FLASK SERVER
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
