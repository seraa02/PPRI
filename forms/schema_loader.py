"""
forms/schema_loader.py
-----------------------
Loads a YAML form schema and builds a schema-aware extraction prompt
that dramatically improves LLM accuracy on checkboxes, numeric fields,
and personal info extraction.

The key accuracy insight
------------------------
Without a schema, the LLM must simultaneously:
  1. Find and parse the question structure
  2. Detect which answer is marked
  3. Format the output

This triple task leads to errors — especially for checkboxes, where the
model often reads the printed option text rather than detecting pen marks.

With a schema, the model is given the question structure and asked to do
only one task: "Which of these known options has a visible mark?"
This eliminates the systematic Yes→No / option-misread failure mode.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_schema(schema_path: Optional[Path]) -> Optional[dict]:
    """
    Load a YAML form schema from disk.

    Args:
        schema_path: Path to the .yaml schema file, or None to run schema-free.

    Returns:
        Parsed schema dict, or None if path is None or file is unreadable.
    """
    if schema_path is None:
        return None

    schema_path = Path(schema_path)
    if not schema_path.exists():
        logger.warning(
            f"Schema file not found: {schema_path}. "
            "Running without schema — accuracy will be lower. "
            "Create a schema file from forms/example_pre_schema.yaml."
        )
        return None

    try:
        import yaml
    except ImportError:
        logger.error(
            "PyYAML is not installed. Run: pip install pyyaml\n"
            "Schema-aware extraction is disabled."
        )
        return None

    try:
        with schema_path.open("r", encoding="utf-8") as fh:
            schema = yaml.safe_load(fh)
        logger.info(
            f"Loaded schema: {schema.get('study_name', '?')} "
            f"{schema.get('survey_type', '?')} "
            f"({len(schema.get('questions', []))} questions, "
            f"{len(schema.get('personal_info', []))} personal-info fields)"
        )
        return schema
    except Exception as exc:
        logger.error(f"Failed to load schema {schema_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_schema_aware_prompt(schema: dict) -> str:
    """
    Build a tightly constrained LLM extraction prompt from a form schema.

    The generated prompt:
      1. Explicitly lists every personal-info field with its label and type.
      2. Lists every question with its number, text, type, and valid options.
      3. Gives precise instructions for each question type (how to detect marks).
      4. Requires the model to output EVERY defined question — even unanswered ones.
      5. Constrains answers to the defined option set for closed questions.

    This eliminates the two most common failure modes:
      - Reading option *text* instead of detecting *marks* (Yes→No errors)
      - Hallucinating answers outside the valid option set

    Args:
        schema: Dict loaded from a YAML schema file.

    Returns:
        Prompt string to send to the Ollama vision model.
    """
    lines = []

    # ------------------------------------------------------------------
    # Header instructions
    # ------------------------------------------------------------------
    study_name = schema.get("study_name", "this study")
    survey_type = schema.get("survey_type", "survey")
    language_note = ""
    lang = schema.get("language", "eng")
    if "spa" in lang:
        language_note = " The form may be in English, Spanish, or both."

    lines.append(
        f"You are extracting participant responses from a scanned {survey_type} survey "
        f"form for {study_name}.{language_note}"
    )
    lines.append("")
    lines.append("=" * 60)
    lines.append("CRITICAL INSTRUCTIONS — READ CAREFULLY")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        "For CHECKBOX / TRUE-FALSE / MULTIPLE CHOICE questions:\n"
        "  - Look for VISUAL MARKS: checkmarks ✓, X marks, filled-in circles ●,\n"
        "    circled text, pen/pencil marks, or crossed boxes.\n"
        "  - DO NOT select an option just because you can read its printed text.\n"
        "  - ONLY report the option that has a visible participant mark.\n"
        "  - If you are unsure which box is marked, set confidence below 0.6.\n"
        "  - ONLY output answers from the provided options list — never invent new options."
    )
    lines.append("")
    lines.append(
        "For NUMERIC fields:\n"
        "  - Read the handwritten or printed number carefully.\n"
        "  - If a range is provided, the answer MUST be within that range.\n"
        "  - If the number is outside the expected range, set confidence to 0.2."
    )
    lines.append("")
    lines.append(
        "For HANDWRITTEN / FILL-IN-THE-BLANK fields:\n"
        "  - Transcribe exactly what the participant wrote.\n"
        "  - If handwriting is illegible, write 'UNCLEAR' and set confidence to 0.0.\n"
        "  - Never guess a person's name — if unclear, write 'UNCLEAR'."
    )
    lines.append("")
    lines.append(
        "For SCALE questions:\n"
        "  - Find the number that is circled, underlined, or otherwise marked.\n"
        "  - Output the marked number as a string, e.g. '7'."
    )
    lines.append("")
    lines.append(
        "OUTPUT RULE: You MUST output every question listed below, even if no answer\n"
        "is visible. For missing answers, use empty string '' and confidence 0.0."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Personal info section
    # ------------------------------------------------------------------
    personal_info_fields = schema.get("personal_info", [])
    if personal_info_fields:
        lines.append("=" * 60)
        lines.append("PERSONAL INFORMATION FIELDS (at the top/header of the form)")
        lines.append("=" * 60)
        lines.append("")

        other_fields = []
        for field in personal_info_fields:
            f_label = field.get("label", field.get("field", "?"))
            f_type  = field.get("type", "handwritten")
            f_name  = field.get("field", "unknown")
            f_desc  = field.get("description", "")
            f_fmt   = field.get("format", "")
            f_digits = field.get("digits", "")

            type_hint = _type_hint(f_type, digits=f_digits, fmt=f_fmt)
            desc_str = f" — {f_desc}" if f_desc else ""
            lines.append(f"  • {f_label} [{type_hint}]{desc_str}")

            if f_name not in ("name", "date"):
                other_fields.append(f_name)

        lines.append("")
        lines.append(
            f"Map name → personal_info.name\n"
            f"Map date → personal_info.date\n"
            f"Map all other fields → personal_info.other dict with keys: "
            f"{', '.join(other_fields) if other_fields else 'none'}"
        )
        lines.append("")

    # ------------------------------------------------------------------
    # Questions section
    # ------------------------------------------------------------------
    questions = schema.get("questions", [])
    if questions:
        lines.append("=" * 60)
        lines.append("QUESTIONS (output ALL of these in the JSON)")
        lines.append("=" * 60)
        lines.append("")

        for q in questions:
            q_num   = q.get("number", "?")
            q_text  = q.get("text", "")
            q_text_es = q.get("text_es", "")
            q_type  = q.get("type", "unknown")
            options = q.get("options", [])
            q_range = q.get("range", [])
            q_digits = q.get("digits", "")

            # Build the question block
            type_str = _type_hint(q_type, options=options, range_=q_range, digits=q_digits)
            lines.append(f"  {q_num}. [{type_str}]")
            lines.append(f"     EN: {q_text}")
            if q_text_es:
                lines.append(f"     ES: {q_text_es}")
            if options:
                lines.append(f"     Valid answers: {options}")
            if q_range and len(q_range) == 2:
                lines.append(f"     Valid range: {q_range[0]} to {q_range[1]}")
            lines.append("")

    # ------------------------------------------------------------------
    # JSON response format
    # ------------------------------------------------------------------
    lines.append("=" * 60)
    lines.append("RESPOND IN THIS EXACT JSON FORMAT — no other text:")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        "CRITICAL — personal_info rules:\n"
        "  • name: copy the handwritten name EXACTLY as it appears on the form.\n"
        "    If no name is present or it is illegible, write the string UNCLEAR.\n"
        "  • date: copy the date EXACTLY as written (e.g. 03/15/2024).\n"
        "    If absent or illegible, write UNCLEAR.\n"
        "  • NEVER output the word 'placeholder' or any template label.\n"
        "  • NEVER copy any example text from these instructions."
    )
    lines.append("")

    # Build example question entries from the schema
    example_questions = []
    for q in questions[:2]:   # show first 2 as examples
        ex_answer = q["options"][0] if q.get("options") else "example answer"
        example_questions.append(
            f'    {{"question_number": "{q["number"]}", '
            f'"question_text": "{q.get("text", "")[:60]}", '
            f'"question_type": "{q.get("type", "unknown")}", '
            f'"selected_answer": "{ex_answer}", '
            f'"confidence": 0.95}}'
        )

    # Build other fields example
    other_keys = [
        f.get("field") for f in personal_info_fields
        if f.get("field") not in ("name", "date")
    ]
    other_example = "{" + ", ".join(f'"{k}": "..."' for k in other_keys) + "}"

    lines.append('{')
    lines.append('  "questions": [')
    if example_questions:
        lines.append(",\n".join(example_questions))
        lines.append("    // ... one entry per question above")
    lines.append('  ],')
    lines.append('  "personal_info": {')
    lines.append('    "name": "Maria Lopez",')
    lines.append('    "date": "03/15/2024",')
    lines.append(f'    "other": {other_example}')
    lines.append('  }')
    lines.append('}')
    lines.append("")
    lines.append(
        "Replace \"Maria Lopez\" with the actual name from the form (or UNCLEAR).\n"
        "Replace \"03/15/2024\" with the actual date from the form (or UNCLEAR)."
    )

    return "\n".join(lines)


def _type_hint(
    q_type: str,
    options: list = None,
    range_: list = None,
    digits: str = "",
    fmt: str = "",
) -> str:
    """Return a human-readable type hint for the prompt."""
    options = options or []
    range_ = range_ or []
    base = q_type.upper()
    parts = [base]
    if options:
        parts.append(f"options: {options}")
    if range_ and len(range_) == 2:
        parts.append(f"range {range_[0]}–{range_[1]}")
    if digits:
        parts.append(f"{digits} digits")
    if fmt:
        parts.append(f"format: {fmt}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Answer validator
# ---------------------------------------------------------------------------

def validate_answer(answer, question_def: dict) -> tuple[str | list, float]:
    """
    Validate an extracted answer against the question definition.

    For closed questions (checkbox, multiple_choice, true_false):
      - If the answer is not in the options list, return it with a low
        confidence penalty so it gets flagged for review.

    For numeric/scale questions:
      - If the value is outside the defined range, penalise confidence.

    Args:
        answer:       The raw extracted answer (string or list).
        question_def: The question dict from the schema.

    Returns:
        (validated_answer, adjusted_confidence)
        adjusted_confidence is 0.0 when the answer is definitely invalid.
    """
    q_type  = question_def.get("type", "unknown")
    options = [str(o).strip().lower() for o in question_def.get("options", [])]
    q_range = question_def.get("range", [])

    # Normalise answer to string for comparison
    if isinstance(answer, list):
        answer_str = [str(a).strip() for a in answer]
    else:
        answer_str = str(answer).strip()

    # ---- Closed-ended questions ----
    if q_type in ("checkbox", "true_false", "multiple_choice") and options:
        if isinstance(answer_str, list):
            # Multiple selections — keep only valid ones
            valid = [a for a in answer_str if a.lower() in options]
            invalid = [a for a in answer_str if a.lower() not in options]
            if invalid:
                logger.debug(
                    f"Invalid option(s) filtered out: {invalid} "
                    f"(valid: {options})"
                )
            return (valid if valid else answer_str), (0.3 if invalid else 1.0)
        else:
            if answer_str.lower() in options:
                return answer_str, 1.0   # valid, no penalty
            # Try partial / case-insensitive match
            for opt in options:
                if opt in answer_str.lower() or answer_str.lower() in opt:
                    logger.debug(
                        f"Fuzzy option match: '{answer_str}' → '{opt}'"
                    )
                    return opt.capitalize() if len(opt) > 1 else opt.upper(), 0.7
            logger.debug(
                f"Answer '{answer_str}' not in options {options} — flagging"
            )
            return answer_str, 0.2   # invalid answer → low confidence → flagged

    # ---- Numeric / scale ----
    if q_type in ("numeric", "scale") and q_range and len(q_range) == 2:
        try:
            num = float(str(answer_str).replace(",", "."))
            lo, hi = float(q_range[0]), float(q_range[1])
            if lo <= num <= hi:
                return answer_str, 1.0
            else:
                logger.debug(
                    f"Numeric answer {num} out of range [{lo}, {hi}] — flagging"
                )
                return answer_str, 0.2
        except (ValueError, TypeError):
            return answer_str, 0.2

    # ---- Open-ended — no validation ----
    return answer_str, 1.0
