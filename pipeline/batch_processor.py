"""
pipeline/batch_processor.py
-----------------------------
Processes a list of survey PDFs in configurable batches with:
  - tqdm progress bar
  - Per-file logging (filename, time taken, status)
  - Checkpoint/resume: after each batch the progress is saved so a
    crashed or interrupted run can resume from the last completed batch
    without reprocessing already-finished files.

Usage:
    processor = BatchProcessor(survey_type="PRE")
    results = processor.process(list_of_pdf_paths)
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from tqdm import tqdm

from config.settings import (
    BATCH_SIZE,
    CHECKPOINT_FOLDER,
    POST_SCHEMA_FILE,
    PRE_SCHEMA_FILE,
    PROCESSING_LOG,
    STUDY_NAME,
)
from pipeline.form_processor import process_pdf, ParticipantRecord

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch-processes PDF survey files with checkpointing and progress tracking."""

    def __init__(self, survey_type: str) -> None:
        """
        Args:
            survey_type: "PRE" or "POST" — used in checkpoint filenames.
        """
        self.survey_type = survey_type.upper()
        self.checkpoint_path = (
            CHECKPOINT_FOLDER
            / f"{STUDY_NAME}_{self.survey_type}_checkpoint.json"
        )
        # Schema path is selected based on survey type
        self.schema_path = (
            PRE_SCHEMA_FILE if self.survey_type == "PRE" else POST_SCHEMA_FILE
        )
        CHECKPOINT_FOLDER.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> dict:
        """Load an existing checkpoint or return a fresh state."""
        if self.checkpoint_path.exists():
            try:
                data = json.loads(
                    self.checkpoint_path.read_text(encoding="utf-8")
                )
                logger.info(
                    f"Checkpoint found for {self.survey_type}: "
                    f"{len(data.get('processed_files', []))} file(s) already done. "
                    "Resuming from last completed batch."
                )
                return data
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(f"Checkpoint file corrupt, starting fresh: {exc}")
        return {
            "study_name": STUDY_NAME,
            "survey_type": self.survey_type,
            "processed_files": [],
            "results": [],
            "last_batch": 0,
            "last_updated": None,
        }

    def _save_checkpoint(self, state: dict) -> None:
        """Persist the current state to disk after each batch."""
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.checkpoint_path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.debug(f"Checkpoint saved: {self.checkpoint_path}")

    # ------------------------------------------------------------------
    # Processing log
    # ------------------------------------------------------------------

    def _log_file_result(self, entry: dict) -> None:
        """Append a processing log entry to processing_log.json."""
        PROCESSING_LOG.parent.mkdir(parents=True, exist_ok=True)

        existing: list = []
        if PROCESSING_LOG.exists():
            try:
                existing = json.loads(PROCESSING_LOG.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

        existing.append(entry)
        PROCESSING_LOG.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, pdf_paths: List[Path]) -> List[ParticipantRecord]:
        """
        Process all PDFs in batches, checkpointing after each batch.

        Args:
            pdf_paths: Full list of PDF Paths to process.

        Returns:
            List of ParticipantRecord dicts (all records, including those
            loaded from a checkpoint).
        """
        state = self._load_checkpoint()
        already_done: set = set(state["processed_files"])

        # Filter out already-processed files
        pending = [p for p in pdf_paths if p.name not in already_done]
        total_pending = len(pending)
        total_all = len(pdf_paths)

        if not pending:
            logger.info(
                f"All {total_all} {self.survey_type} file(s) already processed "
                "(loaded from checkpoint)."
            )
            return state["results"]

        logger.info(
            f"Processing {total_pending} pending {self.survey_type} PDF(s) "
            f"(skipping {total_all - total_pending} already done)."
        )

        # Split pending files into batches
        batches = [
            pending[i : i + BATCH_SIZE]
            for i in range(0, total_pending, BATCH_SIZE)
        ]

        # Outer progress bar covers all pending files
        with tqdm(
            total=total_pending,
            desc=f"{self.survey_type} surveys",
            unit="file",
            dynamic_ncols=True,
        ) as pbar:
            for batch_idx, batch in enumerate(batches, start=state["last_batch"] + 1):
                logger.info(
                    f"Starting batch {batch_idx}/{len(batches) + state['last_batch']} "
                    f"({len(batch)} file(s))"
                )

                for pdf_path in batch:
                    start_time = time.perf_counter()
                    pbar.set_postfix_str(pdf_path.name[:40])

                    try:
                        record = process_pdf(
                            pdf_path,
                            survey_type=self.survey_type,
                            schema_path=self.schema_path,
                        )
                    except Exception as exc:
                        # Catch-all so one bad file never kills the whole run
                        logger.error(
                            f"Unhandled exception processing {pdf_path.name}: {exc}",
                            exc_info=True,
                        )
                        record = {
                            "filename": pdf_path.name,
                            "survey_type": self.survey_type,
                            "personal_info": {
                                "name": "UNCLEAR",
                                "date": "UNCLEAR",
                                "other": {},
                            },
                            "questions": {},
                            "flagged_questions": [],
                            "status": "error",
                            "pages_processed": 0,
                            "error": str(exc),
                        }

                    elapsed = time.perf_counter() - start_time

                    # Log per-file result
                    log_entry = {
                        "filename": pdf_path.name,
                        "survey_type": self.survey_type,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "time_taken_seconds": round(elapsed, 2),
                        "status": record.get("status", "error"),
                        "pages_processed": record.get("pages_processed", 0),
                        "questions_extracted": len(record.get("questions", {})),
                        "flagged_count": len(record.get("flagged_questions", [])),
                        "error": record.get("error"),
                    }
                    self._log_file_result(log_entry)

                    logger.info(
                        f"[{record['status'].upper():8s}] {pdf_path.name} "
                        f"| {elapsed:.1f}s "
                        f"| {len(record.get('questions', {}))} questions "
                        f"| {len(record.get('flagged_questions', []))} flagged"
                    )

                    # Accumulate
                    state["results"].append(record)
                    state["processed_files"].append(pdf_path.name)
                    pbar.update(1)

                # Save checkpoint after every complete batch
                state["last_batch"] = batch_idx
                self._save_checkpoint(state)
                logger.info(f"Checkpoint saved after batch {batch_idx}.")

        # Print a brief summary
        results = state["results"]
        statuses = [r.get("status", "error") for r in results]
        logger.info(
            f"{self.survey_type} complete — "
            f"total={len(results)}, "
            f"success={statuses.count('success')}, "
            f"flagged={statuses.count('flagged')}, "
            f"error={statuses.count('error')}"
        )

        return results
