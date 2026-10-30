
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from backend.intelligence import parse_property_query, rank_properties


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

# Load environment variables from the .env file beside app.py.
load_dotenv(ROOT / ".env")

DATA_FILE = ROOT / "data" / "properties.json"
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(
    __name__,
    template_folder="frontend",
    static_folder="frontend",
    static_url_path="/static"
)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


# --------------------------------------------------
# PROPERTY DATA
# --------------------------------------------------

def load_properties():
    with DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


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


# --------------------------------------------------
# PROPERTY API
# --------------------------------------------------

@app.get("/api/properties")
def api_properties():
    return jsonify(load_properties())


@app.post("/api/parse-query")
def api_parse_query():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", ""))[:500]

    parsed = parse_property_query(query)
    ranked = rank_properties(load_properties(), parsed)

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
# OPTIONAL PROPERTY IMAGE ANALYSIS
# Requires HF_TOKEN if you use this feature.
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

    model = os.getenv(
        "HF_IMAGE_MODEL",
        "google/vit-base-patch16-224"
    )

    endpoint = (
        f"https://api-inference.huggingface.co/models/{model}"
    )

    try:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {token}"
            },
            data=uploaded.read(),
            timeout=45
        )

        if not response.ok:
            app.logger.error(
                "Image provider returned HTTP %s",
                response.status_code
            )

            return jsonify({
                "error": (
                    "The image model could not process this image."
                ),
                "providerStatus": response.status_code
            }), 502

        return jsonify({
            "model": model,
            "results": response.json(),
            "notice": (
                "Image labels are predictions, "
                "not a professional property inspection."
            )
        })

    except requests.RequestException:
        app.logger.exception("Image analysis request failed")

        return jsonify({
            "error": "The image analysis service could not be reached."
        }), 502


# --------------------------------------------------
# ESTARA AI PROPERTY ASSISTANT
# Requires:
#   python -m pip install groq
#   GROQ_API_KEY in .env
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
                "ESTARA AI is not configured. "
                "Check GROQ_API_KEY in your .env file "
                "and restart Flask."
            )
        }), 503

    # Keep only the property details needed for the AI.
    features = property_info.get("features", [])

    if not isinstance(features, list):
        features = []

    safe_property = {
        "name": str(
            property_info.get(
                "name",
                property_info.get("title", "Property")
            )
        )[:200],
        "location": str(
            property_info.get("location", "Not provided")
        )[:200],
        "type": str(
            property_info.get(
                "type",
                property_info.get(
                    "propertyType",
                    "Not provided"
                )
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
            str(feature)[:150]
            for feature in features[:20]
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
                        "You are ESTARA AI, a friendly property "
                        "discovery assistant. Answer in simple, "
                        "clear language using the supplied property "
                        "details. Never invent property facts, "
                        "approvals, ownership details, market prices, "
                        "rental yields, or investment returns. If "
                        "information is missing, say so and explain "
                        "how to verify it. For legal, financial, "
                        "structural, or ownership questions, recommend "
                        "checking with a qualified professional. "
                        "Keep answers useful and reasonably concise."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Selected property details:\n"
                        + json.dumps(
                            safe_property,
                            ensure_ascii=False
                        )
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
                "error": (
                    "ESTARA AI returned an empty answer. "
                    "Please try again."
                )
            }), 502

        return jsonify({
            "answer": answer
        })

    except ImportError:
        app.logger.exception("Groq package is unavailable")

        return jsonify({
            "error": (
                "Groq is not installed. Run "
                "python -m pip install groq"
            )
        }), 500

    except Exception:
        # The full diagnostic appears in the Flask terminal.
        # Do not expose the API key or provider credentials.
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
