"""
pipeline/llm_extractor.py
--------------------------
Structured form-data extraction using a Hugging Face vision-language model
(Qwen2-VL) running entirely locally via PyTorch + CUDA.

Cluster-friendly design
------------------------
- No servers, no background daemons, no network calls.
- Model is loaded once on first call and cached for the lifetime of the process.
- Runs inside a standard SLURM job with --gres=gpu:t4:1 (or similar).

Two extraction modes
--------------------
1. SCHEMA-AWARE (recommended)
   Call extract_from_image(..., schema=loaded_schema_dict).
   A schema-aware prompt is built so the model knows every question,
   its type, and valid answer options.

2. SCHEMA-FREE (fallback / generic)
   Call extract_from_image(..., schema=None).
   The generic discovery prompt is used. Accuracy is lower.

PRIVACY GUARANTEE: All inference runs on the local compute node.
No participant data, images, or text ever leaves the machine.
"""

import json
import logging
import re
from typing import Optional

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from config.settings import HF_MODEL, CONFIDENCE_THRESHOLD

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
# Image resizing
# ---------------------------------------------------------------------------

# Qwen2-VL splits images into patches. A 300 DPI A4 page (~2480×3508 px)
# produces ~11,000 image tokens, filling the context before any text.
# 1280 px on the longest side gives ~2,100 tokens — enough to read
# checkboxes and handwriting while leaving room for the prompt and output.
_LLM_MAX_SIDE_PX = 1280


def _resize_for_llm(image: Image.Image) -> Image.Image:
    """Downscale so the longest side is at most _LLM_MAX_SIDE_PX."""
    w, h = image.size
    longest = max(w, h)
    if longest <= _LLM_MAX_SIDE_PX:
        return image
    scale = _LLM_MAX_SIDE_PX / longest
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Lazy model loading — loaded once, cached for the process lifetime
# ---------------------------------------------------------------------------

_model: Optional[Qwen2VLForConditionalGeneration] = None
_processor: Optional[AutoProcessor] = None


def _get_model_and_processor():
    """Load (or return cached) model + processor."""
    global _model, _processor
    if _model is None:
        logger.info(f"Loading vision model: {HF_MODEL}  (first call — may take a minute)")
        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            HF_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",         # places layers on GPU(s) automatically
            attn_implementation="sdpa" # scaled-dot-product attention, memory-efficient
        )
        _model.eval()
        _processor = AutoProcessor.from_pretrained(HF_MODEL)
        device = next(_model.parameters()).device
        logger.info(f"Model ready on {device}: {HF_MODEL}")
    return _model, _processor


# ---------------------------------------------------------------------------
# JSON extraction helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _clean_llm_response(text: str) -> str:
    """Normalise common LLM output artifacts before JSON parsing."""
    text = text.replace("\\_", "_")
    return text.strip()


def _extract_json_from_response(text: str) -> Optional[dict]:
    """
    Attempt to parse a JSON object from the model's raw output.

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
# Result normalisation (unchanged from original)
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
# Hybrid confidence scoring (unchanged from original)
# ---------------------------------------------------------------------------

def _hybrid_confidence(
    llm_confidence: float,
    answer: str,
    ocr_text: str,
    question_type: str,
) -> float:
    """Adjust LLM confidence by cross-checking against OCR text."""
    if question_type in ("handwritten", "fill_blank") or not ocr_text.strip():
        return llm_confidence

    if not answer or answer == "UNCLEAR":
        return llm_confidence

    answer_lower = str(answer).lower()
    ocr_lower    = ocr_text.lower()

    if answer_lower in ocr_lower:
        return min(1.0, llm_confidence + 0.05)

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
    Run form-data extraction on a single page image using the local HF model.

    Args:
        image:    PIL Image of a single form page (preprocessed).
        ocr_text: Tesseract OCR text for supplemental context and hybrid
                  confidence scoring.
        schema:   Parsed form schema dict from schema_loader.load_schema().
                  When provided, a schema-aware prompt is used that
                  dramatically improves accuracy.  Pass None for generic mode.

    Returns:
        dict with keys:
            "questions"    – list of question extraction dicts
            "personal_info"– dict with name, date, other
            "parse_error"  – True if the model returned non-JSON (optional)
            "raw_response" – preview of raw output on error (optional)
    """
    # Resize before passing to the model
    image = _resize_for_llm(image)
    # Ensure RGB (Qwen2-VL does not accept RGBA or palette-mode images)
    if image.mode != "RGB":
        image = image.convert("RGB")

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

    # ------------------------------------------------------------------
    # Run HuggingFace inference
    # ------------------------------------------------------------------
    try:
        model, processor = _get_model_and_processor()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply the model's chat template to format input correctly
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,      # greedy decoding — deterministic output
            )

        # Decode only the newly generated tokens (skip the echoed input)
        input_len = inputs["input_ids"].shape[1]
        raw_text: str = processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )

        parsed = _extract_json_from_response(raw_text)

        if parsed is None:
            logger.warning(
                f"Model returned non-JSON response. "
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

    except Exception as exc:
        logger.error(f"Unexpected error during model inference: {exc}")
        result = dict(_EMPTY_RESULT)
        result["parse_error"] = True
        result["raw_response"] = str(exc)
        return result


def check_hf_model() -> bool:
    """
    Verify the HuggingFace model can be loaded.

    Triggers the lazy load on first call, then returns True.
    Returns False and logs an error if loading fails.
    """
    try:
        _get_model_and_processor()
        return True
    except Exception as exc:
        logger.error(
            f"Could not load model '{HF_MODEL}': {exc}\n"
            f"Make sure the model is downloaded or HF_HOME points to a valid cache.\n"
            f"Download with: python -c \"from transformers import AutoProcessor, "
            f"Qwen2VLForConditionalGeneration; "
            f"Qwen2VLForConditionalGeneration.from_pretrained('{HF_MODEL}'); "
            f"AutoProcessor.from_pretrained('{HF_MODEL}')\""
        )
        return False
