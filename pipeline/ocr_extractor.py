"""
pipeline/ocr_extractor.py
--------------------------
Extracts raw text from form images using Tesseract OCR (pytesseract).
Runs entirely locally. Supports English and Spanish (eng+spa).

The OCR text is passed to the LLM extractor as supplemental context
to improve extraction accuracy — the LLM is the primary extractor.
"""

import logging
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


def extract_text(image: Image.Image, language: str = "eng+spa") -> str:
    """
    Run Tesseract OCR on a PIL Image and return the raw extracted text.

    Args:
        image:    PIL Image (should be RGB, 300 DPI for best results).
        language: Tesseract language string, e.g. 'eng+spa', 'eng', 'spa'.

    Returns:
        Extracted text as a string. Empty string on failure.

    Raises:
        RuntimeError: If Tesseract is not installed.
    """
    try:
        import pytesseract
    except ImportError:
        raise RuntimeError(
            "pytesseract is not installed. Run: pip install pytesseract"
        )

    try:
        # PSM 6: Assume a single uniform block of text — good for forms
        custom_config = r"--oem 3 --psm 6"
        text = pytesseract.image_to_string(
            image,
            lang=language,
            config=custom_config,
        )
        logger.debug(f"OCR extracted {len(text)} characters")
        return text.strip()

    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Tesseract OCR is not installed or not in PATH. "
            "Windows: download installer from https://github.com/UB-Mannheim/tesseract/wiki "
            "macOS:   brew install tesseract && brew install tesseract-lang "
            "Linux:   sudo apt-get install -y tesseract-ocr tesseract-ocr-spa"
        )
    except pytesseract.TesseractError as e:
        # Often means the language pack is missing
        if "spa" in language and "Failed loading language" in str(e):
            logger.warning(
                f"Spanish language pack not found. Retrying with English only. "
                f"To fix: install tesseract-ocr-spa (Linux) or download spa.traineddata."
            )
            return extract_text(image, language="eng")
        logger.error(f"Tesseract error: {e}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected OCR error: {e}")
        return ""


def get_ocr_confidence(image: Image.Image, language: str = "eng+spa") -> float:
    """
    Return the mean confidence score from Tesseract's word-level data.

    Args:
        image:    PIL Image.
        language: Tesseract language string.

    Returns:
        Mean confidence in [0.0, 1.0]. Returns 0.0 on error.
    """
    try:
        import pytesseract
        import pandas as pd

        data = pytesseract.image_to_data(
            image,
            lang=language,
            output_type=pytesseract.Output.DATAFRAME,
        )
        # Filter out rows with no word (-1 confidence) and empty text
        valid = data[(data["conf"] > 0) & (data["text"].str.strip() != "")]
        if valid.empty:
            return 0.0
        return float(valid["conf"].mean()) / 100.0

    except Exception:
        return 0.0
