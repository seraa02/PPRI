"""
config/settings.py
------------------
Central configuration for the PPRI Survey Processing Pipeline.
All values are loaded from the .env file (copy .env.example → .env).

PRIVACY: All paths and services are strictly local. Nothing here
points to any external server, cloud, or third-party API.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env from the project root (one level up from config/)
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Study identity
# ---------------------------------------------------------------------------
STUDY_NAME: str = os.getenv("STUDY_NAME", "Survey_Study")

# ---------------------------------------------------------------------------
# Input/output folders
# ---------------------------------------------------------------------------
PRE_FOLDER: Path = Path(os.getenv("PRE_FOLDER", "./PRE"))
POST_FOLDER: Path = Path(os.getenv("POST_FOLDER", "./POST"))
OUTPUT_FOLDER: Path = Path(os.getenv("OUTPUT_FOLDER", "./output"))

# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "50"))

# ---------------------------------------------------------------------------
# Ollama (local vision LLM)
# ---------------------------------------------------------------------------
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llava")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
LANGUAGE: str = os.getenv("LANGUAGE", "eng+spa")

# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

# ---------------------------------------------------------------------------
# PDF conversion
# ---------------------------------------------------------------------------
DPI: int = int(os.getenv("DPI", "300"))

# ---------------------------------------------------------------------------
# Form schemas (optional but strongly recommended for better accuracy)
# ---------------------------------------------------------------------------
# Path to YAML schema files describing the exact questions and answer options
# for each survey type. When provided, schema-aware extraction is used.
# Copy forms/example_pre_schema.yaml → forms/your_study_pre.yaml and edit.
_pre_schema_raw  = os.getenv("PRE_SCHEMA_FILE", "")
_post_schema_raw = os.getenv("POST_SCHEMA_FILE", "")

PRE_SCHEMA_FILE:  Optional[Path] = (
    Path(_pre_schema_raw)  if _pre_schema_raw  else None
)
POST_SCHEMA_FILE: Optional[Path] = (
    Path(_post_schema_raw) if _post_schema_raw else None
)

# ---------------------------------------------------------------------------
# Derived paths (not configurable via .env — computed automatically)
# ---------------------------------------------------------------------------
REVIEW_FOLDER: Path = _PROJECT_ROOT / "review"
FLAGGED_LOG: Path = REVIEW_FOLDER / "flagged_log.json"
CHECKPOINT_FOLDER: Path = OUTPUT_FOLDER / "checkpoints"
PIPELINE_LOG: Path = OUTPUT_FOLDER / "pipeline.log"
PROCESSING_LOG: Path = OUTPUT_FOLDER / "processing_log.json"
