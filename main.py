#!/usr/bin/env python3
"""
main.py — PPRI Survey Form Processing Pipeline
================================================
Entry point for the fully local, privacy-first automated survey
processing system.

Usage:
    python main.py

Configure via .env (copy .env.example → .env and edit values).

PRIVACY GUARANTEE: All PDF conversion, OCR, and LLM inference runs
entirely on this machine. No participant data, images, or text is ever
sent to any external server, cloud service, or API of any kind.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: make sure the project root is on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import (
    BATCH_SIZE,
    CHECKPOINT_FOLDER,
    CONFIDENCE_THRESHOLD,
    DPI,
    FLAGGED_LOG,
    LANGUAGE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OUTPUT_FOLDER,
    PIPELINE_LOG,
    POST_FOLDER,
    PRE_FOLDER,
    REVIEW_FOLDER,
    STUDY_NAME,
)
from pipeline.llm_extractor import check_ollama_connection
from pipeline.batch_processor import BatchProcessor
from pipeline.output_writer import OutputWriter


# ---------------------------------------------------------------------------
# Directory bootstrap
# ---------------------------------------------------------------------------

def _bootstrap_directories() -> None:
    """Create all required directories and initialise empty log files."""
    for d in [PRE_FOLDER, POST_FOLDER, OUTPUT_FOLDER, CHECKPOINT_FOLDER, REVIEW_FOLDER]:
        d.mkdir(parents=True, exist_ok=True)

    if not FLAGGED_LOG.exists():
        FLAGGED_LOG.write_text(
            json.dumps({"flagged_entries": []}, indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    """Log to both stdout and output/pipeline.log."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)-8s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(PIPELINE_LOG), encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt,
                        datefmt=datefmt, handlers=handlers)


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _preflight_checks() -> bool:
    """
    Verify the environment before starting.
    Returns True if all checks pass, False if a fatal issue is found.
    """
    logger = logging.getLogger(__name__)
    ok = True

    # Check Ollama
    logger.info("Checking Ollama connection ...")
    if not check_ollama_connection():
        logger.error(
            "Ollama is not reachable. "
            f"Model: {OLLAMA_MODEL}  URL: {OLLAMA_BASE_URL}\n"
            "  → Start Ollama:       ollama serve\n"
            f"  → Pull the model:    ollama pull {OLLAMA_MODEL}"
        )
        ok = False

    # Check Tesseract (non-fatal — OCR is supplemental)
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        logger.info("Tesseract OCR: found.")
    except Exception:
        logger.warning(
            "Tesseract OCR is not installed or not in PATH. "
            "OCR context will be skipped. "
            "Install it for better accuracy:\n"
            "  macOS:  brew install tesseract tesseract-lang\n"
            "  Linux:  sudo apt-get install tesseract-ocr tesseract-ocr-spa\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki"
        )

    # Check pdf2image / poppler
    try:
        from pdf2image import convert_from_path   # noqa: F401
        logger.info("pdf2image: found.")
    except ImportError:
        logger.error("pdf2image is not installed. Run: pip install pdf2image")
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _bootstrap_directories()
    _setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 68)
    logger.info("  PPRI Survey Form Processing Pipeline")
    logger.info("  PRIVACY: 100%% local — no data leaves this machine")
    logger.info("=" * 68)
    logger.info(f"  Study:             {STUDY_NAME}")
    logger.info(f"  PRE folder:        {PRE_FOLDER.resolve()}")
    logger.info(f"  POST folder:       {POST_FOLDER.resolve()}")
    logger.info(f"  Output folder:     {OUTPUT_FOLDER.resolve()}")
    logger.info(f"  Ollama model:      {OLLAMA_MODEL}  @ {OLLAMA_BASE_URL}")
    logger.info(f"  OCR language:      {LANGUAGE}")
    logger.info(f"  DPI:               {DPI}")
    logger.info(f"  Batch size:        {BATCH_SIZE}")
    logger.info(f"  Confidence thresh: {CONFIDENCE_THRESHOLD}")
    logger.info("=" * 68)

    if not _preflight_checks():
        logger.error("Pre-flight checks failed. Fix the issues above and retry.")
        return 1

    # ------------------------------------------------------------------
    # Discover PDFs
    # ------------------------------------------------------------------
    pre_pdfs  = sorted(list(PRE_FOLDER.glob("*.pdf"))  + list(PRE_FOLDER.glob("*.PDF")))
    post_pdfs = sorted(list(POST_FOLDER.glob("*.pdf")) + list(POST_FOLDER.glob("*.PDF")))

    if not pre_pdfs and not post_pdfs:
        logger.error(
            "No PDF files found in PRE or POST folders. "
            "Add scanned survey PDFs and try again."
        )
        return 1

    logger.info(f"Found {len(pre_pdfs)} PRE PDF(s) and {len(post_pdfs)} POST PDF(s).")

    # ------------------------------------------------------------------
    # Process PRE surveys
    # ------------------------------------------------------------------
    pre_results = []
    if pre_pdfs:
        logger.info(f"\n{'─'*68}")
        logger.info(f"  Processing PRE surveys ({len(pre_pdfs)} file(s)) ...")
        logger.info(f"{'─'*68}")
        processor = BatchProcessor(survey_type="PRE")
        pre_results = processor.process(pre_pdfs)
    else:
        logger.warning("PRE folder is empty — skipping PRE processing.")

    # ------------------------------------------------------------------
    # Process POST surveys
    # ------------------------------------------------------------------
    post_results = []
    if post_pdfs:
        logger.info(f"\n{'─'*68}")
        logger.info(f"  Processing POST surveys ({len(post_pdfs)} file(s)) ...")
        logger.info(f"{'─'*68}")
        processor = BatchProcessor(survey_type="POST")
        post_results = processor.process(post_pdfs)
    else:
        logger.warning("POST folder is empty — skipping POST processing.")

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    logger.info(f"\n{'─'*68}")
    logger.info("  Writing output files ...")
    writer = OutputWriter(study_name=STUDY_NAME)
    paths = writer.write(pre_results=pre_results, post_results=post_results)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    all_results = pre_results + post_results
    total   = len(all_results)
    success = sum(1 for r in all_results if r.get("status") == "success")
    flagged = sum(1 for r in all_results if r.get("status") == "flagged")
    errors  = sum(1 for r in all_results if r.get("status") == "error")

    logger.info("=" * 68)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Total processed:    {total}")
    logger.info(f"  Success:            {success}")
    logger.info(f"  Flagged for review: {flagged}")
    logger.info(f"  Errors:             {errors}")
    logger.info("-" * 68)
    logger.info(f"  Excel output:  {paths.get('excel', 'N/A')}")
    logger.info(f"  PRE CSV:       {paths.get('pre_csv', 'N/A')}")
    logger.info(f"  POST CSV:      {paths.get('post_csv', 'N/A')}")
    logger.info(f"  Pipeline log:  {PIPELINE_LOG}")

    if flagged > 0:
        logger.info("-" * 68)
        logger.info(
            f"  {flagged} file(s) have low-confidence answers flagged for review."
        )
        logger.info("  To inspect them, run:")
        logger.info("    python review/review_viewer.py")

    logger.info("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
