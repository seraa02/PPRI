"""
pipeline/form_processor.py
---------------------------
Orchestrates end-to-end extraction for a single survey PDF:

  1. PDF → page images            (pdf_converter)
  2. Each page → OCR text         (ocr_extractor)
  3. Each page + OCR → structured JSON  (llm_extractor, schema-aware)
  4. Answer validation against schema   (forms/schema_loader)
  5. Merge multi-page results into one ParticipantRecord
  6. Flag low-confidence / invalid answers

Schema-aware mode
-----------------
When PRE_SCHEMA_FILE or POST_SCHEMA_FILE is set in .env, the processor
loads the YAML schema and passes it to the LLM extractor, enabling
schema-aware prompting.  This significantly improves accuracy for
checkboxes, numeric fields, and personal-info extraction.

Returns a ParticipantRecord dict consumed by batch_processor/output_writer.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from config.settings import (
    CONFIDENCE_THRESHOLD,
    DPI,
    FLAGGED_LOG,
    LANGUAGE,
    REVIEW_FOLDER,
)
from pipeline.pdf_converter import pdf_to_images, preprocess_image
from pipeline.ocr_extractor import extract_text
from pipeline.llm_extractor import extract_from_image

logger = logging.getLogger(__name__)

# Type alias
ParticipantRecord = dict[str, Any]

# Schema cache — loaded once per session per schema path
_schema_cache: dict[str, Optional[dict]] = {}

# Strings that should always be normalised to "UNCLEAR"
_UNCLEAR_VARIANTS = {
    "unclear",
    # Old prompt-example placeholder text (BUG-1 artefact)
    "participant name or unclear",
    "date as written or unclear",
    # New prompt-example placeholder text (after BUG-1 fix)
    "maria lopez",
    "03/15/2024",
}


def _normalise_unclear(value: str) -> str:
    """
    Return 'UNCLEAR' if *value* is empty, a recognised UNCLEAR variant,
    or a left-over prompt-example placeholder from schema_loader.py.
    All comparisons are case-insensitive.
    """
    stripped = value.strip()
    if not stripped or stripped.lower() in _UNCLEAR_VARIANTS:
        return "UNCLEAR"
    return stripped


# ---------------------------------------------------------------------------
# Schema loading (cached)
# ---------------------------------------------------------------------------

def _load_schema_cached(schema_path: Optional[Path]) -> Optional[dict]:
    """Load and cache the form schema. Returns None if no path given."""
    if schema_path is None:
        return None
    key = str(schema_path)
    if key not in _schema_cache:
        from forms.schema_loader import load_schema
        _schema_cache[key] = load_schema(schema_path)
    return _schema_cache[key]


# ---------------------------------------------------------------------------
# Answer validation against schema
# ---------------------------------------------------------------------------

def _validate_against_schema(
    questions_map: dict,
    schema: Optional[dict],
) -> dict:
    """
    Post-process extracted answers by validating them against the schema.

    For each extracted question that matches a schema question:
      - Closed questions (checkbox/MCQ/true_false): ensure answer is in options.
        If not → lower confidence to 0.2 so it gets flagged.
      - Numeric/scale: ensure value is within defined range.
        If not → lower confidence to 0.2.

    Questions not in the schema are left unchanged.

    Returns a new questions dict with adjusted confidences.
    """
    if not schema:
        return questions_map

    try:
        from forms.schema_loader import validate_answer
    except ImportError:
        return questions_map

    # Build a lookup: question number → schema question definition
    schema_q_by_num = {
        str(q.get("number", "")).strip(): q
        for q in schema.get("questions", [])
    }

    validated = {}
    for q_num, q_info in questions_map.items():
        q_def = schema_q_by_num.get(q_num)
        if q_def is None:
            validated[q_num] = q_info
            continue

        raw_answer   = q_info.get("answer", "")
        orig_conf    = q_info.get("confidence", 0.5)

        validated_answer, validation_conf = validate_answer(raw_answer, q_def)

        # Use the minimum of LLM confidence and validation confidence
        # so that a confident but wrong answer still gets flagged
        final_conf = min(orig_conf, validation_conf)

        validated[q_num] = dict(q_info)
        validated[q_num]["answer"]     = validated_answer
        validated[q_num]["confidence"] = final_conf

        if final_conf < orig_conf:
            logger.debug(
                f"{q_num}: confidence adjusted {orig_conf:.2f} → {final_conf:.2f} "
                f"(answer '{raw_answer}' failed validation)"
            )

    return validated


# ---------------------------------------------------------------------------
# Flagged-log helpers
# ---------------------------------------------------------------------------

def _load_flagged_log() -> dict:
    REVIEW_FOLDER.mkdir(parents=True, exist_ok=True)
    if FLAGGED_LOG.exists():
        try:
            return json.loads(FLAGGED_LOG.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"flagged_entries": []}


def _save_flagged_log(log: dict) -> None:
    REVIEW_FOLDER.mkdir(parents=True, exist_ok=True)
    FLAGGED_LOG.write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _append_flagged_entry(
    filename: str,
    page_number: int,
    question_number: str,
    raw_ocr_text: str,
    confidence: float,
    selected_answer: str,
    reason: str = "",
) -> None:
    log = _load_flagged_log()
    entry = {
        "filename": filename,
        "page_number": page_number,
        "question_number": question_number,
        "raw_ocr_text": raw_ocr_text[:500],
        "confidence_note": (
            f"Confidence {confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}"
            + (f" — {reason}" if reason else "")
        ),
        "selected_answer": selected_answer,
    }
    log["flagged_entries"].append(entry)
    _save_flagged_log(log)


# ---------------------------------------------------------------------------
# Question-number sort key
# ---------------------------------------------------------------------------

def _question_sort_key(label: str) -> tuple:
    digits = "".join(filter(str.isdigit, label))
    prefix = "".join(filter(str.isalpha, label)).upper()
    return (prefix, int(digits) if digits else 0, label)


# ---------------------------------------------------------------------------
# Core processor
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: Path,
    survey_type: str,
    schema_path: Optional[Path] = None,
) -> ParticipantRecord:
    """
    Extract all answers and personal info from a single survey PDF.

    Args:
        pdf_path:    Path to the PDF file.
        survey_type: "PRE" or "POST".
        schema_path: Optional path to a YAML form schema file.
                     When provided, schema-aware extraction and answer
                     validation are both enabled.

    Returns:
        ParticipantRecord dict.
    """
    filename = pdf_path.name
    logger.debug(f"Processing: {filename} (schema: {schema_path})")

    record: ParticipantRecord = {
        "filename": filename,
        "survey_type": survey_type,
        "personal_info": {"name": "UNCLEAR", "date": "UNCLEAR", "other": {}},
        "questions": {},
        "flagged_questions": [],
        "status": "success",
        "pages_processed": 0,
        "schema_used": str(schema_path) if schema_path else None,
        "error": None,
    }

    # ------------------------------------------------------------------
    # Load schema (cached after first load)
    # ------------------------------------------------------------------
    schema = _load_schema_cached(schema_path)

    # ------------------------------------------------------------------
    # Step 1 — Convert PDF to images
    # ------------------------------------------------------------------
    try:
        raw_images = pdf_to_images(pdf_path, dpi=DPI)
    except Exception as exc:
        record["status"] = "error"
        record["error"] = str(exc)
        logger.error(f"PDF conversion failed for {filename}: {exc}")
        return record

    if not raw_images:
        record["status"] = "error"
        record["error"] = "PDF produced no images (empty or corrupt file)."
        return record

    record["pages_processed"] = len(raw_images)

    # ------------------------------------------------------------------
    # Step 2 & 3 — Per-page: preprocess → OCR → LLM extraction
    # ------------------------------------------------------------------
    personal_info_set = False

    for page_idx, raw_image in enumerate(raw_images, start=1):
        page_num = page_idx
        image = preprocess_image(raw_image, enhance=True)

        # OCR — supplemental context + hybrid confidence input
        try:
            ocr_text = extract_text(image, language=LANGUAGE)
        except RuntimeError as exc:
            logger.warning(f"OCR skipped for page {page_num} of {filename}: {exc}")
            ocr_text = ""

        # LLM extraction (schema-aware if schema is loaded)
        try:
            page_result = extract_from_image(
                image,
                ocr_text=ocr_text,
                schema=schema,
            )
        except RuntimeError:
            record["status"] = "error"
            record["error"] = "Vision model inference failed. Check logs for details."
            return record

        # ------------------------------------------------------------------
        # Capture personal info — prefer first page with a real name
        # ------------------------------------------------------------------
        pi = page_result.get("personal_info", {})
        pi_name = _normalise_unclear(pi.get("name") or "")
        pi_date = _normalise_unclear(pi.get("date") or "")

        if not personal_info_set or pi_name != "UNCLEAR":
            record["personal_info"] = {
                "name": pi_name,
                "date": pi_date,
                "other": pi.get("other", {}),
            }
            if pi_name != "UNCLEAR":
                personal_info_set = True

        # ------------------------------------------------------------------
        # Merge questions — keep the entry with higher confidence per question
        # ------------------------------------------------------------------
        for q in page_result.get("questions", []):
            q_num   = str(q.get("question_number", "UNKNOWN")).strip()
            conf    = float(q.get("confidence", 0.5))
            answer  = q.get("selected_answer", "")
            q_text  = q.get("question_text", "")
            q_type  = q.get("question_type", "unknown")

            existing = record["questions"].get(q_num)
            if existing and existing["confidence"] >= conf:
                continue

            record["questions"][q_num] = {
                "text":       q_text,
                "type":       q_type,
                "answer":     answer,
                "confidence": conf,
                "page":       page_num,
            }

    # ------------------------------------------------------------------
    # Step 4 — Answer validation against schema
    # ------------------------------------------------------------------
    record["questions"] = _validate_against_schema(record["questions"], schema)

    # ------------------------------------------------------------------
    # Step 5 — Flag low-confidence answers
    # ------------------------------------------------------------------
    for page_num_ref in range(1, record["pages_processed"] + 1):
        # Re-run OCR to get the right page's text for the log entry
        # (we store it per question below using the stored page number)
        pass

    for q_num, q_info in record["questions"].items():
        if q_info["confidence"] < CONFIDENCE_THRESHOLD:
            if q_num not in record["flagged_questions"]:
                record["flagged_questions"].append(q_num)

            reason = ""
            if q_info["confidence"] <= 0.2:
                reason = "answer failed schema validation or is out of range"
            elif q_info["confidence"] == 0.0:
                reason = "illegible / UNCLEAR"

            _append_flagged_entry(
                filename=filename,
                page_number=q_info.get("page", 1),
                question_number=q_num,
                raw_ocr_text="[see processing_log.json for OCR context]",
                confidence=q_info["confidence"],
                selected_answer=str(q_info["answer"]),
                reason=reason,
            )

    # ------------------------------------------------------------------
    # Step 6 — Final status and sorted questions
    # ------------------------------------------------------------------
    if record["flagged_questions"]:
        record["status"] = "flagged"

    record["questions"] = dict(
        sorted(
            record["questions"].items(),
            key=lambda kv: _question_sort_key(kv[0]),
        )
    )

    logger.debug(
        f"{filename}: {len(record['questions'])} questions, "
        f"{len(record['flagged_questions'])} flagged, "
        f"status={record['status']}"
    )
    return record
