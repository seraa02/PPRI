# PPRI Survey Form Processing Pipeline

Automated, fully local extraction of participant answers from scanned paper survey PDFs.
Built for older-adult health research organizations running studies at senior centers across Texas.

---

## Privacy Guarantee

> **All processing runs entirely on this machine. No participant data — including names, dates, survey answers, images, or OCR text — is ever transmitted to any external server, cloud service, or API.**

This is enforced by design:

- PDF-to-image conversion uses **pdf2image + Poppler** — a local binary, no network access.
- OCR uses **Tesseract** — a local binary, no network access.
- Vision/LLM extraction uses **Ollama** running at `http://localhost:11434` — a local server, fully air-gapped from the internet.
- Output files are written to a local folder you specify.

There are no API keys, no cloud credentials, and no telemetry of any kind.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.11 or 3.12 |
| RAM | 8 GB | **16 GB** (required for llava on CPU) |
| GPU | Not required | NVIDIA GPU with 8 GB+ VRAM (greatly speeds up Ollama) |
| Disk | 5 GB free | 10 GB+ (for model weights + output files) |
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 | Latest LTS |

---

## Installation

### Step 1 — Install Python 3.10+

**Windows**
1. Download the installer from https://www.python.org/downloads/
2. Check **"Add Python to PATH"** during installation.
3. Verify: open Command Prompt → `python --version`

**macOS**
```bash
# Using Homebrew (recommended)
brew install python@3.11
python3.11 --version
```

**Linux (Ubuntu/Debian)**
```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip
python3.11 --version
```

---

### Step 2 — Install Poppler (required by pdf2image)

**Windows**
1. Download the latest release from https://github.com/oschwartz10612/poppler-windows/releases
2. Extract the ZIP anywhere, e.g. `C:\poppler\`
3. Add `C:\poppler\Library\bin` to your system `PATH`:
   - Search → "Environment Variables" → Edit `Path` → New → paste the path.
4. Restart Command Prompt and verify: `pdftoppm -v`

**macOS**
```bash
brew install poppler
pdftoppm -v
```

**Linux (Ubuntu/Debian)**
```bash
sudo apt-get install -y poppler-utils
pdftoppm -v
```

---

### Step 3 — Install Tesseract with the Spanish language pack

**Windows**
1. Download the installer from https://github.com/UB-Mannheim/tesseract/wiki
   (choose the version ending in `_with_languages_and_traineddata.exe`)
2. Run the installer — tick **Spanish** during the component selection step.
3. Add the Tesseract install directory (e.g. `C:\Program Files\Tesseract-OCR`) to `PATH`.
4. Verify: `tesseract --version` and `tesseract --list-langs` (should include `spa`)

**macOS**
```bash
brew install tesseract
brew install tesseract-lang   # installs all language packs including Spanish
tesseract --list-langs        # verify 'spa' is listed
```

**Linux (Ubuntu/Debian)**
```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa
tesseract --list-langs   # verify both 'eng' and 'spa' are listed
```

---

### Step 4 — Install Ollama and pull the llava model

**Windows & macOS**
1. Download and run the installer from https://ollama.com/download
2. After installation, open a terminal and run:
```bash
ollama serve          # starts the local server (keep this terminal open)
```
3. In a new terminal, pull the vision model:
```bash
ollama pull llava
```
4. Verify the model is available:
```bash
ollama list           # should show 'llava'
```

**Linux**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &        # start in background
ollama pull llava
ollama list
```

> **Note:** The llava model is ~4 GB. `llama3.2-vision` (~2 GB) is a lighter alternative — set `OLLAMA_MODEL=llama3.2-vision` in your `.env`.

---

### Step 5 — Install Python dependencies

```bash
# From the project root
pip install -r requirements.txt
```

Or with a virtual environment (recommended):

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

### Step 6 — Configure the pipeline

```bash
# Copy the template
cp .env.example .env   # macOS / Linux
copy .env.example .env  # Windows

# Open .env in any text editor and set your values:
```

Key settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `STUDY_NAME` | Name of the study (used in output filenames and Excel header) | `Survey_Study` |
| `PRE_FOLDER` | Path to folder with PRE survey PDFs | `./PRE` |
| `POST_FOLDER` | Path to folder with POST survey PDFs | `./POST` |
| `OUTPUT_FOLDER` | Where results are saved | `./output` |
| `BATCH_SIZE` | PDFs per batch before checkpoint | `50` |
| `OLLAMA_MODEL` | Ollama vision model name | `llava` |
| `OLLAMA_BASE_URL` | Local Ollama server | `http://localhost:11434` |
| `LANGUAGE` | Tesseract language string | `eng+spa` |
| `CONFIDENCE_THRESHOLD` | Answers below this score are flagged | `0.7` |
| `DPI` | PDF render resolution | `300` |

---

## Running the Pipeline

### Single study (standard workflow)

1. Place scanned PRE survey PDFs in the `PRE/` folder.
2. Place scanned POST survey PDFs in the `POST/` folder.
3. Make sure Ollama is running:
   ```bash
   ollama serve
   ```
4. Run the pipeline:
   ```bash
   python main.py
   ```
5. Output files appear in `output/`:
   ```
   output/
   ├── Falls_Prevention_results.xlsx   ← Excel with PRE, POST, and Summary sheets
   ├── Falls_Prevention_PRE_results.csv
   ├── Falls_Prevention_POST_results.csv
   ├── pipeline.log                    ← full run log
   ├── processing_log.json             ← per-file JSON log
   └── checkpoints/                    ← resume state
   ```

### Resuming an interrupted run

If the pipeline crashes or is stopped mid-run, simply run `python main.py` again. It will automatically detect the checkpoint file and skip all already-processed files, resuming from the last completed batch.

To force a full reprocess, delete the checkpoint file:
```bash
rm output/checkpoints/<STUDY_NAME>_PRE_checkpoint.json
rm output/checkpoints/<STUDY_NAME>_POST_checkpoint.json
```

---

## Multiple Studies on Separate Machines

Each study can run independently on a different machine:

1. **Set up each machine** following the installation steps above.
2. **Copy the pipeline code** to each machine (or use a shared network drive for code only — never for data).
3. **On each machine**, create a `.env` file pointing to that study's folders:
   ```ini
   # Machine A — Falls Prevention
   STUDY_NAME=Falls_Prevention
   PRE_FOLDER=/data/falls_prevention/PRE
   POST_FOLDER=/data/falls_prevention/POST
   OUTPUT_FOLDER=/data/falls_prevention/output

   # Machine B — Nutrition Study
   STUDY_NAME=Nutrition_Study
   PRE_FOLDER=/data/nutrition/PRE
   POST_FOLDER=/data/nutrition/POST
   OUTPUT_FOLDER=/data/nutrition/output
   ```
4. Run `python main.py` on each machine independently.
5. Collect the output Excel/CSV files from each machine for consolidation.

> Data never needs to travel between machines. Each machine processes only its own study's PDFs.

---

## Reviewing Flagged Responses

When the LLM confidence is below the threshold (default 0.7), the answer is flagged for human review. After the pipeline completes, run:

```bash
python review/review_viewer.py
```

The interactive CLI lets you:

```
  Commands:
    list         — show all files with flagged questions
    <filename>   — inspect flagged questions for that file (or type its number from list)
    open <file>  — open the original PDF in your system's default viewer
    quit         — exit the viewer
```

Example session:
```
> list
  Files with flagged questions:
   1.  participant_047.pdf  (3 flagged question(s))
   2.  participant_112.pdf  (1 flagged question(s))

> participant_047.pdf
  ════════════════════════════════════════
  File: participant_047.pdf
  Flagged questions: 3

  [1] Question Q3  (page 1)
  Confidence note: Confidence 0.45 below threshold 0.70
  Extracted answer: Maybe (ambiguous circle)
  Raw OCR text: Q3. Do you exercise regularly?  O Yes  O No  Q Ma...

> open participant_047.pdf
  Opened: /path/to/PRE/participant_047.pdf
```

Flagged entries are also stored in `review/flagged_log.json` for programmatic inspection.

---

## Output Excel File Structure

The Excel file (`{STUDY_NAME}_results.xlsx`) contains three sheets:

### Sheet: Summary
Overview table showing total, successful, flagged, and error counts for PRE and POST.

### Sheet: PRE (and Sheet: POST)

| Row | Content |
|-----|---------|
| **Row 1** | Study name, merged across all columns, dark blue background |
| **Row 2** | "PRE Survey Results" label, merged, medium blue background |
| **Row 3** | Column headers: Participant_ID, Name, Date, Q1, Q2, …, Flagged_Questions |
| **Row 4+** | One row per participant. Rows with flagged questions are highlighted yellow. |

**Columns:**

| Column | Description |
|--------|-------------|
| `Participant_ID` | PDF filename (without extension) |
| `Name` | Participant name extracted from the form, or `UNCLEAR` |
| `Date` | Date written on the form, or `UNCLEAR` |
| `Q1`, `Q2`, … | Extracted answer for each question (only what the participant wrote/selected) |
| `Flagged_Questions` | Comma-separated list of question numbers with low confidence |

Multiple selected answers (e.g. in a multiple-choice question) are joined with `; `.

---

## Troubleshooting

### "Tesseract not found" or "TesseractNotFoundError"
- **Windows:** Add the Tesseract install directory to `PATH` and restart your terminal.
- **macOS:** Run `brew install tesseract`. If using a venv, make sure the venv can see system binaries.
- **Linux:** Run `sudo apt-get install -y tesseract-ocr`.
- Set the path explicitly in your script if needed:
  ```python
  # At the top of main.py (Windows example)
  import pytesseract
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```

### "Cannot connect to Ollama" or connection refused on port 11434
- Make sure Ollama is running: `ollama serve` (keep the terminal open).
- On macOS/Windows, Ollama may auto-start as a background service after installation — check your system tray.
- Verify the URL in `.env`: `OLLAMA_BASE_URL=http://localhost:11434`
- Check Ollama's status: `curl http://localhost:11434/api/tags`

### "Model 'llava' not found"
- Pull the model first: `ollama pull llava`
- List available models: `ollama list`
- If using a different model, update `.env`: `OLLAMA_MODEL=llama3.2-vision`

### PDF conversion errors ("PDFInfoNotInstalledError")
- Poppler is not installed or not in `PATH`.
- See **Step 2** of the installation instructions above.
- Verify: open a terminal and run `pdftoppm -v`

### "Failed loading language 'spa'"
- The Spanish Tesseract language pack is missing.
- **Linux:** `sudo apt-get install -y tesseract-ocr-spa`
- **macOS:** `brew install tesseract-lang`
- **Windows:** Re-run the Tesseract installer and select the Spanish language.
- As a workaround, set `LANGUAGE=eng` in `.env` to use English OCR only.

### Low confidence rates (many responses flagged)
- **Increase DPI:** Set `DPI=400` in `.env` for higher-resolution images.
- **Check scan quality:** Blurry, skewed, or low-contrast scans reduce accuracy.
- **Try a different model:** `llama3.2-vision` sometimes performs better on certain form layouts. Pull it with `ollama pull llama3.2-vision` and set `OLLAMA_MODEL=llama3.2-vision`.
- **Lower the threshold temporarily:** Set `CONFIDENCE_THRESHOLD=0.5` to reduce flagging — but be aware this means less quality control.
- **Use a GPU:** Running Ollama with a GPU dramatically improves both speed and accuracy. See Ollama docs: https://ollama.com/blog/ollama-is-now-available-as-an-official-docker-image

### Slow processing
- A GPU is strongly recommended for Ollama (NVIDIA with 8 GB+ VRAM).
- On CPU, each form page may take 30–120 seconds. For 50 forms with 2 pages each, expect 1–4 hours.
- Reduce `DPI` to `200` to speed up PDF conversion (slight quality trade-off).
- Processing continues from the checkpoint if interrupted — you don't lose progress.

### "No PDF files found"
- Make sure PDFs are directly inside `PRE/` or `POST/` (not in sub-folders).
- Check that files end in `.pdf` or `.PDF`.
- Verify the paths in `.env` match your actual folder locations.

### Memory errors / Python crashes
- Ensure at least 16 GB RAM is available.
- Reduce `BATCH_SIZE` to `10` or `20` in `.env` to process fewer files at once.
- Close other applications to free RAM.

---

## Project Structure

```
PPRI/
├── .env.example              ← Configuration template
├── .env                      ← Your local config (gitignored)
├── requirements.txt          ← Python dependencies
├── main.py                   ← Pipeline entry point
│
├── config/
│   └── settings.py           ← Loads .env and exposes all config constants
│
├── pipeline/
│   ├── pdf_converter.py      ← PDF → PIL Images (pdf2image + Poppler)
│   ├── ocr_extractor.py      ← Image → raw text (Tesseract)
│   ├── llm_extractor.py      ← Image → structured JSON (Ollama llava)
│   ├── form_processor.py     ← Orchestrates one PDF: convert → OCR → LLM → flag
│   ├── batch_processor.py    ← Batch loop, progress bar, checkpointing, logging
│   └── output_writer.py      ← Writes Excel (.xlsx) and CSV files
│
├── review/
│   ├── flagged_log.json      ← Auto-generated: all low-confidence entries
│   └── review_viewer.py      ← Interactive CLI for human review
│
├── PRE/                      ← Drop PRE survey PDFs here
├── POST/                     ← Drop POST survey PDFs here
│
└── output/                   ← All generated files land here
    ├── {STUDY_NAME}_results.xlsx
    ├── {STUDY_NAME}_PRE_results.csv
    ├── {STUDY_NAME}_POST_results.csv
    ├── pipeline.log
    ├── processing_log.json
    └── checkpoints/
        ├── {STUDY_NAME}_PRE_checkpoint.json
        └── {STUDY_NAME}_POST_checkpoint.json
```

---

## License

Internal use only — PPRI research organization.
