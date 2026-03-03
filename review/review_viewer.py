#!/usr/bin/env python3
"""
review/review_viewer.py
------------------------
Interactive CLI tool for reviewing flagged survey responses.

Usage:
    python review/review_viewer.py

The tool lets a human:
  1. List all filenames that have flagged questions.
  2. Enter a filename to see which questions were flagged, the raw OCR text,
     and the LLM's extracted answer.
  3. Optionally open the original PDF for manual inspection.

Run this after the pipeline to inspect any low-confidence extractions.
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from textwrap import wrap

# ---------------------------------------------------------------------------
# Allow running directly from the review/ folder or from the project root
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from config.settings import FLAGGED_LOG, PRE_FOLDER, POST_FOLDER
except ImportError:
    # Fallback defaults if settings can't be imported
    FLAGGED_LOG  = _PROJECT_ROOT / "review" / "flagged_log.json"
    PRE_FOLDER   = _PROJECT_ROOT / "PRE"
    POST_FOLDER  = _PROJECT_ROOT / "POST"


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

_BOLD  = "\033[1m"
_CYAN  = "\033[96m"
_YELLOW = "\033[93m"
_RED   = "\033[91m"
_GREEN = "\033[92m"
_RESET = "\033[0m"


def _supports_color() -> bool:
    return sys.stdout.isatty() and platform.system() != "Windows"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if _supports_color() else text


def _print_separator(char: str = "─", width: int = 72) -> None:
    print(char * width)


def _wrap_print(text: str, indent: int = 4, width: int = 68) -> None:
    for line in wrap(str(text), width=width):
        print(" " * indent + line)


# ---------------------------------------------------------------------------
# Flagged-log loader
# ---------------------------------------------------------------------------

def load_flagged_log() -> list:
    """Load all flagged entries from flagged_log.json."""
    if not FLAGGED_LOG.exists():
        return []
    try:
        data = json.loads(FLAGGED_LOG.read_text(encoding="utf-8"))
        return data.get("flagged_entries", [])
    except (json.JSONDecodeError, OSError) as exc:
        print(_c(f"Error reading flagged log: {exc}", _RED))
        return []


def group_by_filename(entries: list) -> dict:
    """Group flagged entries by filename → list of entries."""
    grouped: dict = {}
    for entry in entries:
        fname = entry.get("filename", "unknown")
        grouped.setdefault(fname, []).append(entry)
    return grouped


# ---------------------------------------------------------------------------
# PDF opener
# ---------------------------------------------------------------------------

def find_pdf(filename: str) -> Path | None:
    """Search PRE and POST folders for the PDF."""
    for folder in [PRE_FOLDER, POST_FOLDER]:
        candidate = Path(folder) / filename
        if candidate.exists():
            return candidate
    return None


def open_pdf(pdf_path: Path) -> None:
    """Open the PDF with the system default viewer."""
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", str(pdf_path)])
        elif system == "Windows":
            os.startfile(str(pdf_path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(pdf_path)])
        print(_c(f"  Opened: {pdf_path}", _GREEN))
    except Exception as exc:
        print(_c(f"  Could not open PDF automatically: {exc}", _RED))
        print(f"  Please open manually: {pdf_path}")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display_flagged_entries(filename: str, entries: list) -> None:
    """Pretty-print all flagged questions for a given file."""
    _print_separator("═")
    print(_c(f"  File: {filename}", _BOLD + _CYAN))
    print(f"  Flagged questions: {len(entries)}")
    _print_separator("═")

    for i, entry in enumerate(entries, start=1):
        print()
        print(_c(f"  [{i}] Question {entry.get('question_number', '?')}"
                 f"  (page {entry.get('page_number', '?')})", _BOLD))
        _print_separator("·", 72)

        conf_note = entry.get("confidence_note", "")
        answer    = entry.get("selected_answer", "")

        print(f"  {_c('Confidence note:', _YELLOW)} {conf_note}")
        print(f"  {_c('Extracted answer:', _YELLOW)} {answer or '(empty)'}")

        ocr = entry.get("raw_ocr_text", "").strip()
        if ocr:
            print(f"  {_c('Raw OCR text:', _YELLOW)}")
            _wrap_print(ocr, indent=6)

    print()


def list_all_flagged_files(grouped: dict) -> None:
    """Display a numbered list of all filenames with flagged questions."""
    _print_separator("═")
    print(_c("  Files with flagged questions:", _BOLD))
    _print_separator()
    if not grouped:
        print("  (none — no flagged entries found)")
    else:
        for idx, (fname, entries) in enumerate(sorted(grouped.items()), start=1):
            print(f"  {idx:3}.  {fname}  ({len(entries)} flagged question(s))")
    _print_separator("═")


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print(_c("  ╔══════════════════════════════════════════════════════╗", _CYAN))
    print(_c("  ║      PPRI Survey — Flagged Response Reviewer         ║", _CYAN))
    print(_c("  ║      All data stays on this machine.                 ║", _CYAN))
    print(_c("  ╚══════════════════════════════════════════════════════╝", _CYAN))
    print()

    entries = load_flagged_log()
    if not entries:
        print(
            "  No flagged entries found. Either no forms were flagged or "
            f"the log file does not exist yet:\n  {FLAGGED_LOG}"
        )
        print()
        return

    grouped = group_by_filename(entries)
    total_flagged = sum(len(v) for v in grouped.values())

    print(f"  Loaded {total_flagged} flagged question(s) across {len(grouped)} file(s).")
    print()

    while True:
        print("  Commands:")
        print("    list         — show all files with flagged questions")
        print("    <filename>   — inspect flagged questions for that file")
        print("    open <file>  — open the PDF in your system viewer")
        print("    quit         — exit the viewer")
        print()

        try:
            raw = input(_c("  > ", _BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break

        if not raw:
            continue

        cmd = raw.lower()

        if cmd in ("quit", "exit", "q"):
            print("  Goodbye.")
            break

        elif cmd == "list":
            list_all_flagged_files(grouped)

        elif cmd.startswith("open "):
            target = raw[5:].strip()
            pdf = find_pdf(target)
            if pdf:
                open_pdf(pdf)
            else:
                print(
                    _c(f"  PDF not found: '{target}' — "
                       "check PRE/ and POST/ folders.", _RED)
                )

        else:
            # Treat input as a filename (partial match allowed)
            matches = [f for f in grouped if cmd in f.lower() or f == raw]
            if not matches:
                # Try numeric selection from list
                try:
                    idx = int(raw) - 1
                    sorted_files = sorted(grouped.keys())
                    if 0 <= idx < len(sorted_files):
                        matches = [sorted_files[idx]]
                except ValueError:
                    pass

            if not matches:
                print(
                    _c(f"  No file matching '{raw}'. "
                       "Type 'list' to see all flagged files.", _RED)
                )
            elif len(matches) > 1:
                print(_c(f"  Multiple matches:", _YELLOW))
                for m in matches:
                    print(f"    - {m}")
                print("  Please be more specific.")
            else:
                fname = matches[0]
                display_flagged_entries(fname, grouped[fname])

                pdf = find_pdf(fname)
                if pdf:
                    ans = input(
                        f"  Open original PDF ({fname}) for manual inspection? [y/N]: "
                    ).strip().lower()
                    if ans in ("y", "yes"):
                        open_pdf(pdf)
                else:
                    print(
                        _c(f"  Original PDF not found in PRE/ or POST/ folders.", _YELLOW)
                    )

        print()


if __name__ == "__main__":
    main()
