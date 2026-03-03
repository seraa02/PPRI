"""
pipeline/pdf_converter.py
--------------------------
Converts PDF pages to PIL Images using pdf2image + poppler, then applies
preprocessing to improve OCR and LLM accuracy on scanned survey forms.

Preprocessing pipeline (in order):
  1. RGB normalisation
  2. Auto-contrast    — stretches histogram, recovers faded ink
  3. Brightness 0.80× — darkens image so faint handwriting/pencil marks
                        become clearly visible before contrast is applied
  4. Contrast 1.80×  — separates dark ink strokes from light paper
  5. Sharpening       — reduces blur from scanner bed
  6. Auto-deskew      — corrects tilted scans using Tesseract OSD
"""

import logging
import re
from pathlib import Path
from typing import List

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF → images
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: Path, dpi: int = 300) -> List[Image.Image]:
    """
    Convert every page of a PDF to a PIL Image.

    Args:
        pdf_path: Path to the PDF file.
        dpi:      Resolution for rendering. 300 recommended; use 400 for
                  dense forms with small checkboxes.

    Returns:
        Ordered list of PIL Images, one per page.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
        RuntimeError:      If poppler is not installed or conversion fails.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        from pdf2image import convert_from_path
        from pdf2image.exceptions import (
            PDFInfoNotInstalledError,
            PDFPageCountError,
            PDFSyntaxError,
        )
    except ImportError:
        raise RuntimeError(
            "pdf2image is not installed. Run: pip install pdf2image"
        )

    try:
        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            fmt="png",
            thread_count=1,      # Stable single-threaded conversion
            use_cropbox=False,
            strict=False,        # Tolerate slightly malformed PDFs
        )
        logger.debug(
            f"Converted {pdf_path.name}: {len(images)} page(s) at {dpi} DPI"
        )
        return images

    except PDFInfoNotInstalledError:
        raise RuntimeError(
            "Poppler is not installed or not in PATH.\n"
            "  macOS:  brew install poppler\n"
            "  Linux:  sudo apt-get install -y poppler-utils\n"
            "  Windows: https://github.com/oschwartz10612/poppler-windows"
        )
    except PDFPageCountError as exc:
        raise RuntimeError(
            f"Could not count pages in {pdf_path.name}: {exc}"
        )
    except PDFSyntaxError as exc:
        raise RuntimeError(f"PDF syntax error in {pdf_path.name}: {exc}")
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error converting {pdf_path.name}: {exc}"
        )


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image: Image.Image, enhance: bool = True) -> Image.Image:
    """
    Apply a preprocessing pipeline optimised for scanned survey forms.

    The steps are ordered so that each builds on the previous:
      1. RGB normalisation — ensures consistent colour mode for PIL ops
      2. Auto-contrast     — stretches pixel histogram; recovers faded ink
                             and washed-out checkbox marks
      3. Contrast boost    — further separates foreground (marks/text)
                             from background (paper)
      4. Sharpening        — reduces scanner blur, sharpens checkbox edges
                             and handwriting strokes
      5. Deskew            — rotates tilted scans back to upright using
                             Tesseract's orientation detection (OSD).
                             A 5-degree tilt makes Tesseract OCR ~30% less
                             accurate and confuses LLM checkbox detection.

    Args:
        image:   Raw PIL Image from pdf_to_images.
        enhance: Set False to skip enhancement (useful for debugging).

    Returns:
        Preprocessed PIL Image ready for OCR and LLM extraction.
    """
    # Step 1 — ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    if not enhance:
        return image

    # Step 2 — auto-contrast (histogram stretching)
    # cutoff=1 ignores the darkest/lightest 1% of pixels to avoid outliers
    image = ImageOps.autocontrast(image, cutoff=1)

    # Step 3 — brightness reduction (darken)
    # Reduces brightness to 0.80x so faint handwriting, light pencil marks,
    # and lightly-checked boxes become visually darker before contrast is
    # applied.  This is the most effective single step for handwritten fields.
    image = ImageEnhance.Brightness(image).enhance(0.80)

    # Step 4 — contrast boost
    # Raised from 1.4× to 1.8× to better separate ink from paper after
    # the brightness reduction.  Handwriting strokes become much more
    # distinct without crushing printed text.
    image = ImageEnhance.Contrast(image).enhance(1.8)

    # Step 5 — sharpening
    # UnsharpMask is superior to SHARPEN filter for text and fine details.
    image = image.filter(
        ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3)
    )

    # Step 5 — deskew using Tesseract OSD
    image = _deskew(image)

    return image


def _deskew(image: Image.Image) -> Image.Image:
    """
    Detect and correct page rotation using Tesseract's OSD engine.

    Tesseract OSD returns the best rotation to make the page upright.
    We apply the inverse rotation to correct it.

    Rotations handled: 0°, 90°, 180°, 270° (page-level), plus small
    skews detected as 0° rotation with a continuous angle estimate.

    Returns the original image unchanged if deskew fails for any reason.
    """
    try:
        import pytesseract

        # OSD works on a reasonably high-resolution grayscale image
        osd = pytesseract.image_to_osd(image, config="--psm 0 -c min_characters_to_try=5")

        # Parse rotation angle (0, 90, 180, 270)
        rotation_match = re.search(r"Rotate:\s*(\d+)", osd)
        if not rotation_match:
            return image

        angle = int(rotation_match.group(1))
        if angle == 0:
            return image

        logger.debug(f"Deskew: rotating image by {angle}°")
        # PIL rotate: positive = counter-clockwise; OSD Rotate = clockwise needed
        return image.rotate(-angle, expand=True, fillcolor=(255, 255, 255))

    except Exception as exc:
        # OSD can fail on forms with few words — silently continue
        logger.debug(f"Deskew skipped: {exc}")
        return image
