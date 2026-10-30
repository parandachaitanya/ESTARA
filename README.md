# ESTARA — Intelligent Property Discovery

A beginner-friendly full-stack starter project using:
- Frontend: HTML, CSS, JavaScript
- Backend: Python + Flask
- NLP: rule-based extraction of location, BHK, property type, and budget from natural-language queries
- ML-style recommendations: scikit-learn TF-IDF similarity + preference scoring
- DL image analysis: optional Hugging Face image-classification API integration (requires an access token)
- AI assistant: a simple property-search assistant powered by the NLP parser and listing data

## 1. Project structure

```text
ESTARA_Project/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   └── properties.json
├── backend/
│   ├── __init__.py
│   └── intelligence.py
├── frontend/
│   ├── discover.html
│   ├── search.html
│   ├── results.html
│   ├── property-details.html
│   └── assets/
│       └── css/
│           └── styles.css
└── uploads/
```

## 2. Put your existing hero image in the right place

Copy your current `hero.png` into:

`frontend/assets/images/hero.png`

Create the `images` folder if it does not exist. This project uses that path so the hero image works when served by Flask.

## 3. Install and run on Windows

Open the ESTARA_Project folder in VS Code. Open Terminal → New Terminal and run:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

Then open: http://127.0.0.1:5000

If `py` is not recognized, install Python 3.11+ and select **Add Python to PATH** during setup.

## 4. How the intelligence features work

### NLP
The `/api/parse-query` endpoint extracts known locations, BHK count, property type, and budget from natural-language sentences. This is a practical rule-based NLP prototype, not a large language model.

Example:
`Find a 3 BHK apartment in Gachibowli under 90 lakh`

### ML recommendations
The recommendation endpoint uses TF-IDF text similarity from scikit-learn and preference matching to rank the available sample listings. This is a real ML technique, but the sample dataset is small and it is not a production-trained real-estate model.

### Deep-learning image analysis
The image-analysis endpoint sends an uploaded property image to a Hugging Face image-classification model when `HF_TOKEN` is configured. It is optional and requires internet access and a valid Hugging Face token. It classifies image contents; it does not certify property condition, value, authenticity, or safety.

### AI assistant
The assistant combines NLP parsing and the sample listing dataset to explain matching properties. It is a lightweight assistant, not a generative LLM.

## 5. Important limitations

- The included properties are fictional demo data for development. Replace them with verified listings and appropriate permission before public deployment.
- The app is a learning prototype, not a real estate transaction platform.
- Add authentication, authorization, database persistence, input validation, rate limiting, privacy controls, and security review before deployment.
- The DL endpoint requires an external service token; never commit real tokens to GitHub.
