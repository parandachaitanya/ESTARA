
import os
import json
import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from groq import Groq


ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_IMAGE_SIZE = 8 * 1024 * 1024

VISION_MODEL = "qwen/qwen3.8-27b"


def analyze_property_image(image_bytes):
    """
    ESTARA Property Condition AI.

    Combines traditional image-quality measurements with
    Groq Vision-based preliminary visual observations.

    This is not a certified structural inspection.
    """

    if not image_bytes:
        raise ValueError("Please upload a property image.")

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise ValueError("Image must be smaller than 8 MB.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image_format = image.format

            if image_format not in ALLOWED_FORMATS:
                raise ValueError(
                    "Upload a JPG, PNG, or WEBP image."
                )

            image = image.convert("RGB")
            image.thumbnail((1000, 1000))
            rgb_image = np.array(image)

    except (UnidentifiedImageError, OSError):
        raise ValueError(
            "The uploaded file is not a valid image."
        )

    gray_image = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2GRAY
    )

    height, width = gray_image.shape

    if width < 100 or height < 100:
        raise ValueError(
            "Please upload a clearer image at least "
            "100 x 100 pixels."
        )

    # Traditional image-quality measurements
    brightness = float(np.mean(gray_image))
    contrast = float(np.std(gray_image))

    edges = cv2.Canny(gray_image, 80, 160)
    edge_percentage = float(np.mean(edges > 0) * 100)

    sharpness = float(
        cv2.Laplacian(
            gray_image,
            cv2.CV_64F
        ).var()
    )

    observations = []

    if brightness < 55:
        observations.append(
            "The image is dark; some details may be difficult to inspect."
        )
    elif brightness > 220:
        observations.append(
            "The image is very bright; surface details may be washed out."
        )
    else:
        observations.append(
            "The image brightness appears suitable for preliminary review."
        )

    if sharpness < 40:
        observations.append(
            "The image may be blurry. Try taking a sharper photo."
        )
    else:
        observations.append(
            "The image contains measurable visual detail."
        )

    if contrast < 25:
        observations.append(
            "The image has low contrast, which can hide surface details."
        )
    else:
        observations.append(
            "The image has measurable contrast."
        )

    # Determine image MIME type
    mime_types = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp"
    }

    mime_type = mime_types[image_format]

    # Encode the original uploaded image for Groq Vision
    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Check your .env file."
        )

    client = Groq(api_key=api_key)

    prompt = """
You are ESTARA's preliminary property-condition visual assistant.

Carefully inspect the supplied property photograph.

Look for visible evidence of:
1. Cracks in walls, ceilings, or visible surfaces.
2. Possible dampness, water stains, or leakage marks.
3. Peeling, flaking, or bubbling paint.
4. Visible surface deterioration or plaster damage.
5. Other clearly visible property-condition concerns.

Do not invent defects. Distinguish visible evidence from uncertainty.
Do not claim to detect hidden structural damage.

Return ONLY a valid JSON object with this structure:

{
  "overall_condition": "Good / Fair / Poor / Unclear",
  "summary": "Short visual assessment",
  "findings": [
    {
      "type": "Cracks",
      "status": "Observed / Not observed / Unclear",
      "severity": "Low / Medium / High / Unknown",
      "description": "What is visibly present",
      "location": "Approximate image location or Unknown",
      "confidence": "Low / Medium / High"
    }
  ],
  "recommendations": [
    "Practical next step"
  ],
  "limitations": [
    "What cannot be determined from this photograph"
  ]
}

Include findings for all five categories, even when no defect is visible.
Use cautious language. A stain alone does not prove active leakage.
A visible line alone does not prove a structural crack.
"""

    try:
        completion = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:{mime_type};base64,"
                                    f"{encoded_image}"
                                )
                            }
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_completion_tokens=1500,
            response_format={
                "type": "json_object"
            }
        )

        ai_text = completion.choices[0].message.content

        if not ai_text:
            raise ValueError(
                "Groq returned an empty image-analysis response."
            )

        damage_analysis = json.loads(ai_text)

    except Exception as error:
        if isinstance(error, ValueError):
            raise

        raise RuntimeError(
            "Groq Vision analysis failed. Check your API key, "
            "internet connection, model availability, and Flask terminal."
        ) from error

    return {
        "success": True,
        "title": "ESTARA AI Property Condition Analysis",

        "image_width": int(width),
        "image_height": int(height),

        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "edge_percentage": round(edge_percentage, 2),
        "sharpness": round(sharpness, 2),

        "observations": observations,

        "damage_analysis": damage_analysis,

        "disclaimer": (
            "AI-generated preliminary visual assessment only. "
            "It cannot confirm hidden defects, active water leakage, "
            "or structural safety. Findings must be verified by a "
            "qualified property inspector."
        )
    }
