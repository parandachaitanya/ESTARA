
import re
from difflib import SequenceMatcher

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


LOCATION_ALIASES = {
    "gachibowli": ["gachibowli", "gachi bowli"],
    "kondapur": ["kondapur", "kondapur"],
    "hitech city": ["hitech city", "hi tech city", "hi-tech city"],
    "miyapur": ["miyapur"],
    "madhapur": ["madhapur"],
    "manikonda": ["manikonda"],
    "jubilee hills": ["jubilee hills"],
    "banjara hills": ["banjara hills"],
    "kokapet": ["kokapet"],
    "narsingi": ["narsingi"],
    "tellapur": ["tellapur"],
    "financial district": ["financial district"],
    "nanakramguda": ["nanakramguda"],
    "lb nagar": ["lb nagar", "l b nagar"],
    "uppal": ["uppal"],
    "kompally": ["kompally"],
    "bachupally": ["bachupally"],
    "bowenpally": ["bowenpally"],
    "secunderabad": ["secunderabad"],
    "hyderabad": ["hyderabad"],
}

TYPE_ALIASES = {
    "Apartment": [
        "apartment", "apartments", "flat", "flats",
        "2bhk flat", "3bhk flat", "2 bhk flat", "3 bhk flat"
    ],
    "Villa": ["villa", "villas", "independent villa"],
    "Independent House": [
        "independent house", "independent home",
        "individual house", "standalone house"
    ],
    "Plot": ["plot", "plots", "land", "open plot"],
    "Commercial": [
        "commercial", "office space", "shop",
        "showroom", "commercial property"
    ],
}


def normalize(text):
    """Normalize text for easier matching."""
    text = str(text or "").lower()
    text = text.replace("₹", " rs ")
    text = text.replace(",", "")
    text = re.sub(r"[^a-z0-9.\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def amount_to_rupees(number, unit=""):
    """Convert lakh/crore amounts to rupees."""
    number = float(number)
    unit = (unit or "").lower()

    if unit in ("crore", "crores", "cr", "crs"):
        return int(number * 10000000)

    if unit in ("lakh", "lakhs", "lac", "lacs", "l"):
        return int(number * 100000)

    return int(number)


def extract_amounts(text):
    """Extract monetary values such as 90 lakh and 1.2 crore."""
    pattern = (
        r"(\d+(?:\.\d+)?)\s*"
        r"(crores?|crs?|cr|lakhs?|lacs?|lac|l|rupees?|rs)?"
    )

    amounts = []

    for match in re.finditer(pattern, text):
        number = match.group(1)
        unit = match.group(2) or ""

        # Ignore numbers that are clearly part of a BHK expression.
        before = text[max(0, match.start() - 8):match.start()]
        after = text[match.end():match.end() + 12]

        if re.search(r"\b(bhk|bedroom|bedrooms|beds?)\s*$", before):
            continue

        if re.match(r"\s*(bhk|bedroom|bedrooms)\b", after):
            continue

        # Ignore numbers without a monetary unit unless preceded by Rs.
        if not unit and not re.search(r"\brs\s*$", before):
            continue

        amounts.append({
            "value": amount_to_rupees(number, unit),
            "start": match.start(),
            "end": match.end(),
        })

    return amounts


def find_location(text):
    """Identify a known Hyderabad locality."""
    padded = f" {text} "

    matches = []

    for canonical, aliases in LOCATION_ALIASES.items():
        for alias in aliases:
            alias_norm = normalize(alias)

            if f" {alias_norm} " in padded:
                matches.append((len(alias_norm), canonical))

    if not matches:
        return None

    # Prefer the longest matching locality name.
    matches.sort(reverse=True)
    return matches[0][1]


def find_property_type(text):
    """Identify the requested property type."""
    matches = []

    for canonical, aliases in TYPE_ALIASES.items():
        for alias in aliases:
            alias_norm = normalize(alias)

            if re.search(r"\b" + re.escape(alias_norm) + r"\b", text):
                matches.append((len(alias_norm), canonical))

    if not matches:
        return None

    matches.sort(reverse=True)
    return matches[0][1]


def parse_property_query(query):
    """
    Convert a natural-language property request into structured filters.

    Example:
    '3 BHK apartment in Gachibowli under 90 lakh'
    """
    original = str(query or "").strip()
    text = normalize(original)

    result = {
        "query": original,
        "location": find_location(text),
        "bhk": None,
        "propertyType": find_property_type(text),
        "minBudget": None,
        "maxBudget": None,
    }

    # Bedroom count: 2 BHK, 3bhk, 2 bedroom flat, etc.
    bhk_match = re.search(
        r"\b([1-9])\s*(?:-\s*)?"
        r"(?:bhk|bedrooms?|beds?)\b",
        text
    )

    if bhk_match:
        result["bhk"] = int(bhk_match.group(1))

    amounts = extract_amounts(text)

    # Budget ranges: between 50 lakh and 1 crore.
    range_match = re.search(
        r"\bbetween\b(.+?)\band\b(.+)",
        text
    )

    if range_match:
        first = extract_amounts(range_match.group(1))
        second = extract_amounts(range_match.group(2))

        if first and second:
            result["minBudget"] = min(
                first[-1]["value"], second[0]["value"]
            )
            result["maxBudget"] = max(
                first[-1]["value"], second[0]["value"]
            )

    # Budget ranges: 50 lakh to 1 crore.
    if result["minBudget"] is None and len(amounts) >= 2:
        between_match = re.search(r"\bto\b|\b-\b", text)

        if between_match:
            first = [
                item for item in amounts
                if item["start"] < between_match.start()
            ]
            second = [
                item for item in amounts
                if item["start"] > between_match.end()
            ]

            if first and second:
                result["minBudget"] = min(
                    first[-1]["value"], second[0]["value"]
                )
                result["maxBudget"] = max(
                    first[-1]["value"], second[0]["value"]
                )

    # Single budget amount: under 90 lakh, above 50 lakh, etc.
    if amounts and result["minBudget"] is None:
        amount = amounts[-1]["value"]

        if re.search(
            r"\b(above|over|at least|minimum|min|more than)\b",
            text
        ):
            result["minBudget"] = amount

        elif re.search(
            r"\b(under|below|within|less than|maximum|max|budget)\b",
            text
        ):
            result["maxBudget"] = amount

        else:
            # A single unqualified amount is treated as a maximum.
            result["maxBudget"] = amount

    return result


def listing_text(property_item):
    """Build searchable text from a property listing."""
    fields = [
        property_item.get("name", ""),
        property_item.get("title", ""),
        property_item.get("location", ""),
        property_item.get("area", ""),
        property_item.get("propertyType", ""),
        property_item.get("type", ""),
        property_item.get("description", ""),
        property_item.get("features", ""),
        property_item.get("amenities", ""),
    ]

    return " ".join(
        str(value) for value in fields if value is not None
    )


def rank_properties(properties, preferences):
    """
    Rank listings using TF-IDF text similarity and structured preferences.
    Returns all listings, ordered by relevance.
    """
    properties = list(properties or [])
    preferences = preferences or {}

    if not properties:
        return []

    query = str(preferences.get("query", "") or "").strip()

    documents = [listing_text(item) for item in properties]

    if query and any(document.strip() for document in documents):
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                stop_words="english"
            )

            matrix = vectorizer.fit_transform(documents + [query])
            similarities = cosine_similarity(
                matrix[-1], matrix[:-1]
            ).flatten()

        except ValueError:
            similarities = [0.0] * len(properties)
    else:
        similarities = [0.0] * len(properties)

    ranked = []

    for index, original in enumerate(properties):
        item = dict(original)
        score = float(similarities[index]) * 100
        reasons = []

        location = normalize(
            item.get("location", item.get("area", ""))
        )

        requested_location = normalize(
            preferences.get("location", "")
        )

        if requested_location:
            if requested_location in location:
                score += 30
                reasons.append("Matches your preferred location")
            else:
                score -= 10

        requested_type = normalize(
            preferences.get("propertyType", "")
        )

        actual_type = normalize(
            item.get("propertyType", item.get("type", ""))
        )

        if requested_type:
            if requested_type == actual_type:
                score += 25
                reasons.append("Matches your property type")
            else:
                score -= 8

        requested_bhk = preferences.get("bhk")

        if requested_bhk not in (None, ""):
            try:
                requested_bhk = int(requested_bhk)
                actual_bhk = int(
                    item.get("bhk", item.get("bedrooms", 0)) or 0
                )

                if actual_bhk == requested_bhk:
                    score += 25
                    reasons.append(
                        f"Matches your {requested_bhk} BHK preference"
                    )
                else:
                    score -= 8
            except (ValueError, TypeError):
                pass

        price = item.get(
            "priceValue",
            item.get("price", item.get("budget"))
        )

        try:
            price_text = normalize(price)
            price_match = re.search(
                r"(\d+(?:\.\d+)?)\s*"
                r"(crores?|crs?|cr|lakhs?|lacs?|lac|l)?",
                price_text
            )

            if price_match:
                price_value = amount_to_rupees(
                    price_match.group(1),
                    price_match.group(2) or ""
                )
            else:
                price_value = float(price)

            minimum = preferences.get("minBudget")
            maximum = preferences.get("maxBudget")

            if minimum not in (None, ""):
                if price_value >= float(minimum):
                    score += 10
                    reasons.append("Within your minimum budget")
                else:
                    score -= 15

            if maximum not in (None, ""):
                if price_value <= float(maximum):
                    score += 15
                    reasons.append("Within your maximum budget")
                else:
                    score -= 15

        except (ValueError, TypeError):
            pass

        item["recommendationScore"] = round(score, 2)
        item["recommendationReasons"] = reasons
        ranked.append(item)

    ranked.sort(
        key=lambda item: item["recommendationScore"],
        reverse=True
    )

    return ranked
