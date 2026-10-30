
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image, ImageOps, ImageStat


MAX_IMAGE_BYTES = 8 * 1024 * 1024
TIMEOUT = (2, 4)

# Cache listing image features for faster repeat searches.
PROPERTY_CACHE = {}


def load_image(source, is_url=False):
    if is_url:
        response = requests.get(
            source,
            timeout=TIMEOUT,
            stream=True,
            headers={"User-Agent": "ESTARA-VisionAI/1.0"}
        )
        response.raise_for_status()

        data = bytearray()

        for chunk in response.iter_content(65536):
            data.extend(chunk)
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError("Image exceeds 8 MB.")

        image = Image.open(BytesIO(data))
    else:
        image = Image.open(source)

    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((256, 256))

    return image


def image_embedding(image):
    # Resize to a small fixed size.
    image = image.resize((32, 32))

    # Colour features.
    color_features = np.asarray(
        image.resize((8, 8)),
        dtype=np.float32
    ).flatten() / 255.0

    # Grayscale structure features.
    gray = image.convert("L").resize((16, 16))
    gray_features = np.asarray(
        gray,
        dtype=np.float32
    ).flatten() / 255.0

    vector = np.concatenate([
        color_features,
        gray_features
    ])

    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def get_property_features(property_item):
    image_url = property_item.get("image")

    if not image_url:
        return None

    if image_url in PROPERTY_CACHE:
        return PROPERTY_CACHE[image_url]

    try:
        image = load_image(image_url, is_url=True)
        features = image_embedding(image)

        PROPERTY_CACHE[image_url] = features

        return features

    except (requests.RequestException, OSError, ValueError):
        return None


def find_similar_properties(uploaded_image, properties, top_k=5):
    query_vector = image_embedding(uploaded_image)

    def process_property(property_item):
        features = get_property_features(property_item)

        if features is None:
            return None

        similarity = float(np.dot(query_vector, features))

        return {
            "property": property_item,
            "similarity": round(
                max(0.0, min(1.0, similarity)), 4
            )
        }

    # Process property images concurrently.
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(process_property, properties)
        )

    results = [item for item in results if item is not None]

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return results[:max(1, min(int(top_k), 10))]
