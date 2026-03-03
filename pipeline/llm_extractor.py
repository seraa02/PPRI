"""
pipeline/llm_extractor.py
--------------------------
Structured form-data extraction using a locally hosted Ollama vision model
(llava, qwen2.5vl, minicpm-v, llama3.2-vision, etc.)

Two extraction modes
--------------------
1. SCHEMA-AWARE (recommended, much more accurate)
   Call extract_from_image(..., schema=loaded_schema_dict).
   The prompt is built from the schema so the model knows exactly which
   questions exist, what type they are, and what the valid answer options
   are.  This eliminates the most common failure modes:
     - Reading printed option text instead of detecting marks (Yes→No errors)
     - Answers outside the valid option set
     - Missed personal-info fields

2. SCHEMA-FREE (fallback, original behaviour)
   Call extract_from_image(..., schema=None).
   The generic discovery prompt is used. Accuracy is lower.

PRIVACY GUARANTEE: All requests go to http://localhost:11434.
No data ever leaves this machine.
"""

import base64
import json
import logging
import re
from io import BytesIO
from typing import Optional

import requests
from PIL import Image

from config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generic (schema-free) extraction prompt — used when no schema is provided
# ---------------------------------------------------------------------------
_GENERIC_PROMPT = """You are a form data extraction assistant. You are looking at a scanned paper survey form. The form may be in English or Spanish or both.

Your job is to extract ONLY the answers that the participant has selected or written. Do NOT list all available options. Only extract what the participant actually chose or wrote.

For each question you can identify on the form, extract the following:
- The question number or label as it appears on the form
- The question text
- The question type: checkbox, true_false, multiple_choice, scale, handwritten, or fill_blank
- Only the selected or written answer. If multiple options are selected, list all of them.
- A confidence score from 0.0 to 1.0 reflecting how certain you are about the extracted answer

IMPORTANT for checkboxes and circles:
- Look for VISUAL MARKS: checkmarks ✓, X marks, filled circles ●, circled text, pen marks.
- Do NOT select an option just because you can read its printed text.
- ONLY report what the participant physically marked.

If handwriting is unclear, do your best to transcribe it and lower your confidence score accordingly. If a checkbox or circle is ambiguous, lower your confidence score. Never guess a personal name if it is not clearly legible — instead write UNCLEAR and set confidence to 0.0.

Respond ONLY in the following JSON format with no extra text before or after:
{
  "questions": [
    {
      "question_number": "Q1",
      "question_text": "the question as it appears on the form",
      "question_type": "checkbox",
      "selected_answer": "Yes",
      "confidence": 0.95
    }
  ],
  "personal_info": {
    "name": "participant name or UNCLEAR",
    "date": "date as written or UNCLEAR",
    "other": {}
  }
}"""

# Fallback empty result
_EMPTY_RESULT = {
    "questions": [],
    "personal_info": {"name": "UNCLEAR", "date": "UNCLEAR", "other": {}},
}


# ---------------------------------------------------------------------------
# Image encoding + resizing for LLM
# ---------------------------------------------------------------------------

# Qwen2.5-VL and other vision LLMs split images into patches.
# A 300 DPI A4 page (~2480×3508 px) produces ~11,000 image tokens,
# which fills the context window before any text can be generated.
# 1280 px on the longest side gives ~2,100 image tokens — enough
# detail to read checkboxes and handwriting while leaving room for
# the schema prompt and JSON output.
_LLM_MAX_SIDE_PX = 1280


def _resize_for_llm(image: Image.Image) -> Image.Image:
    """
    Downscale a PIL Image so its longest side is at most _LLM_MAX_SIDE_PX.
    Images already within the limit are returned unchanged (no upscaling).
    """
    w, h = image.size
    longest = max(w, h)
    if longest <= _LLM_MAX_SIDE_PX:
        return image
    scale = _LLM_MAX_SIDE_PX / longest
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def _image_to_base64(image: Image.Image) -> str:
    """Resize for LLM, then encode as base64 PNG."""
    image = _resize_for_llm(image)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# JSON extraction from LLM response
# ---------------------------------------------------------------------------

def _clean_llm_response(text: str) -> str:
    """
    Normalise common LLM output artifacts before JSON parsing.

    Fixes:
      - Escaped underscores (llava markdown artifact: backslash+underscore) → plain underscore
      - Strip leading/trailing whitespace
    """
    text = text.replace("\\_", "_")
    return text.strip()


def _extract_json_from_response(text: str) -> Optional[dict]:
    """
    Attempt to parse a JSON object from the LLM's raw response.

    Four strategies in order:
      1. Direct parse after cleaning.
      2. Regex-extract the outermost {...} block.
      3. Extract from a ```json ... ``` code fence.
      4. Truncated JSON recovery — append closing delimiters.
    """
    cleaned = _clean_llm_response(text)

    # Strategy 1
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence:
        try:
            return json.loads(_clean_llm_response(fence.group(1)))
        except json.JSONDecodeError:
            pass

    # Strategy 4 — truncated response recovery
    for suffix in ("}}]}", "}]}", "}}", "}"):
        try:
            return json.loads(cleaned + suffix)
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Result normalisation
# ---------------------------------------------------------------------------

def _normalize_result(parsed: dict) -> dict:
    """Ensure the parsed dict has all expected keys with correct types."""
    if "questions" not in parsed or not isinstance(parsed["questions"], list):
        parsed["questions"] = []
    if "personal_info" not in parsed or not isinstance(parsed["personal_info"], dict):
        parsed["personal_info"] = {"name": "UNCLEAR", "date": "UNCLEAR", "other": {}}
    else:
        pi = parsed["personal_info"]
        pi.setdefault("name", "UNCLEAR")
        pi.setdefault("date", "UNCLEAR")
        pi.setdefault("other", {})

    for q in parsed["questions"]:
        q.setdefault("question_number", "UNKNOWN")
        q.setdefault("question_text", "")
        q.setdefault("question_type", "unknown")
        q.setdefault("selected_answer", "")
        try:
            q["confidence"] = max(0.0, min(1.0, float(q.get("confidence", 0.5))))
        except (TypeError, ValueError):
            q["confidence"] = 0.5

    return parsed


# ---------------------------------------------------------------------------
# Hybrid confidence scoring
# ---------------------------------------------------------------------------

def _hybrid_confidence(
    llm_confidence: float,
    answer: str,
    ocr_text: str,
    question_type: str,
) -> float:
    """
    Adjust LLM confidence by cross-checking against OCR text.

    Rationale: The LLM reports how certain *it feels*, but can be confidently
    wrong.  If the OCR and LLM agree on a closed-ended answer, that is a
    stronger signal.  If they disagree on something clearly readable by OCR,
    the LLM confidence is penalised.

    This does NOT flag handwritten fields because OCR is unreliable on those.

    Returns an adjusted confidence score in [0.0, 1.0].
    """
    if question_type in ("handwritten", "fill_blank") or not ocr_text.strip():
        return llm_confidence   # no cross-check for open-ended fields

    if not answer or answer == "UNCLEAR":
        return llm_confidence

    answer_lower = str(answer).lower()
    ocr_lower    = ocr_text.lower()

    if answer_lower in ocr_lower:
        # OCR sees the same answer text → small confidence boost (cap at 1.0)
        return min(1.0, llm_confidence + 0.05)

    # OCR doesn't contain the answer — slight penalty for closed questions
    # where the mark detection may be wrong
    if question_type in ("checkbox", "true_false", "multiple_choice"):
        return max(0.0, llm_confidence - 0.1)

    return llm_confidence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_from_image(
    image: Image.Image,
    ocr_text: str = "",
    schema: Optional[dict] = None,
) -> dict:
    """
    Send a form-page image to the local Ollama vision model.

    Args:
        image:    PIL Image of a single form page (preprocessed).
        ocr_text: Tesseract OCR text for supplemental context and
                  hybrid confidence scoring.
        schema:   Parsed form schema dict from schema_loader.load_schema().
                  When provided, a schema-aware prompt is used that
                  dramatically improves accuracy.  Pass None to use the
                  generic discovery prompt.

    Returns:
        dict with keys:
            "questions"    – list of question extraction dicts
            "personal_info"– dict with name, date, other
            "parse_error"  – True if LLM returned non-JSON (optional)
            "raw_response" – preview of raw response on error (optional)
    """
    img_b64 = _image_to_base64(image)

    # ------------------------------------------------------------------
    # Build prompt — schema-aware or generic
    # ------------------------------------------------------------------
    if schema is not None:
        try:
            from forms.schema_loader import build_schema_aware_prompt
            prompt = build_schema_aware_prompt(schema)
            logger.debug("Using schema-aware extraction prompt.")
        except Exception as exc:
            logger.warning(
                f"Schema prompt builder failed ({exc}), falling back to generic prompt."
            )
            prompt = _GENERIC_PROMPT
    else:
        prompt = _GENERIC_PROMPT

    # Append OCR text as supplemental context (capped to avoid token overflow)
    if ocr_text.strip():
        prompt += (
            f"\n\n{'='*60}\n"
            f"OCR-EXTRACTED TEXT (use as reference, may contain errors):\n"
            f"{'='*60}\n"
            f"{ocr_text.strip()[:2000]}"
        )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.05,   # Very low → deterministic, consistent output
            "top_p": 0.9,
            "num_predict": 4096,   # Max tokens to generate (covers large forms)
            "num_ctx": 16384,      # Context window: image (~2100) + prompt (~1600) + output (4096)
        },
    }

    # ------------------------------------------------------------------
    # Call Ollama
    # ------------------------------------------------------------------
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=600,   # CPU inference for qwen2.5vl:7b can take 5-8 min/page
        )
        response.raise_for_status()

        raw_text: str = response.json().get("response", "")
        parsed = _extract_json_from_response(raw_text)

        if parsed is None:
            logger.warning(
                f"LLM returned non-JSON response. "
                f"Preview: {raw_text[:300]!r}"
            )
            result = dict(_EMPTY_RESULT)
            result["parse_error"] = True
            result["raw_response"] = raw_text[:500]
            return result

        result = _normalize_result(parsed)

        # Apply hybrid confidence scoring
        for q in result.get("questions", []):
            q["confidence"] = _hybrid_confidence(
                llm_confidence=q["confidence"],
                answer=str(q.get("selected_answer", "")),
                ocr_text=ocr_text,
                question_type=q.get("question_type", "unknown"),
            )

        return result

    except requests.exceptions.ConnectionError:
        msg = (
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running: ollama serve"
        )
        logger.error(msg)
        raise RuntimeError(msg)

    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out. Consider reducing DPI or page count.")
        result = dict(_EMPTY_RESULT)
        result["parse_error"] = True
        result["raw_response"] = "TIMEOUT"
        return result

    except requests.exceptions.HTTPError as exc:
        logger.error(f"Ollama HTTP error: {exc}")
        result = dict(_EMPTY_RESULT)
        result["parse_error"] = True
        result["raw_response"] = str(exc)
        return result

    except Exception as exc:
        logger.error(f"Unexpected error calling Ollama: {exc}")
        result = dict(_EMPTY_RESULT)
        result["parse_error"] = True
        result["raw_response"] = str(exc)
        return result


def check_ollama_connection() -> bool:
    """
    Verify Ollama is running and the configured model is available.

    Returns:
        True if connected and model found, False otherwise.
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()

        models = resp.json().get("models", [])
        available = [m.get("name", "").split(":")[0] for m in models]
        target   = OLLAMA_MODEL.split(":")[0]

        if target not in available:
            logger.warning(
                f"Model '{OLLAMA_MODEL}' not in Ollama. "
                f"Available: {available}. "
                f"Run: ollama pull {OLLAMA_MODEL}"
            )
            return False

        logger.info(f"Ollama connected. Model '{OLLAMA_MODEL}' is ready.")
        return True

    except requests.exceptions.ConnectionError:
        logger.error(
            f"Ollama is not running at {OLLAMA_BASE_URL}. "
            "Start it with: ollama serve"
        )
        return False
    except Exception as exc:
        logger.error(f"Could not verify Ollama: {exc}")
        return False
