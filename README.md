# PPRI Survey Processing Pipeline

> **Automated, fully local extraction of participant answers from scanned paper survey PDFs.**  
> Built for older-adult health research running studies at senior centers across Texas.

---

## What This Does

Researchers receive stacks of handwritten paper surveys after each study session. Manually entering data from hundreds of forms is slow, error-prone, and expensive.

This pipeline automates that entire process:

1. You drop scanned PDFs into a folder
2. Run one command
3. Get back a clean Excel workbook — one row per participant, one column per question

Everything runs locally on TAMU's HPRC supercomputer. No participant data ever touches the internet. No API keys. No cloud services. IRB-compliant by design.

---

## Privacy Guarantee

> **No participant data — names, dates, survey answers, images, or OCR text — is ever transmitted to any external server, cloud service, or API of any kind.**

- PDF → image conversion: **Poppler** (local binary)
- OCR text extraction: **Tesseract** (local binary)  
- AI answer extraction: **Qwen2-VL** (runs on GPU, fully offline)
- Output: local Excel/CSV files only

---

## How It Works

```
Scanned PDF
    │
    ▼  pdf_converter.py      — converts each page to a high-res image
Page Images
    │
    ▼  ocr_extractor.py      — extracts raw text using Tesseract OCR
OCR Text
    │
    ▼  llm_extractor.py      — AI reads the image + OCR, extracts answers as JSON
Structured Answers
    │
    ▼  output_writer.py      — writes Excel + CSV results
MockStudy_results.xlsx
```

Low-confidence answers are automatically flagged for human review.

---

## Project Structure

```
PPRI 3/
├── main.py                        ← entry point — run this
├── run_pipeline.slurm             ← SLURM job script for TAMU HPRC
├── requirements.txt               ← Python dependencies
├── .env.example                   ← copy this to .env and edit
│
├── config/
│   └── settings.py                ← loads .env, exposes config constants
│
├── pipeline/
│   ├── pdf_converter.py           ← PDF → PIL images (Poppler)
│   ├── ocr_extractor.py           ← image → raw text (Tesseract)
│   ├── llm_extractor.py           ← image → structured JSON (Qwen2-VL GPU)
│   ├── form_processor.py          ← runs one PDF end-to-end
│   ├── batch_processor.py         ← batch loop, checkpointing, progress bar
│   └── output_writer.py           ← writes .xlsx and .csv
│
├── forms/
│   ├── schema_loader.py           ← loads YAML schema, builds AI prompt
│   ├── falls_prevention_pre.yaml  ← schema for Falls Prevention PRE survey
│   ├── falls_prevention_post.yaml ← schema for Falls Prevention POST survey
│   ├── example_pre_schema.yaml    ← template — copy this for a new study
│   └── example_post_schema.yaml
│
├── review/
│   └── review_viewer.py           ← interactive CLI for reviewing flagged answers
│
├── PRE/                           ← drop PRE survey PDFs here
├── POST/                          ← drop POST survey PDFs here
└── output/                        ← all results land here
    ├── {STUDY_NAME}_results.xlsx
    ├── {STUDY_NAME}_PRE_results.csv
    ├── {STUDY_NAME}_POST_results.csv
    ├── pipeline.log
    └── checkpoints/
```

---

## Requirements

- TAMU HPRC Grace cluster account
- GPU node (T4 minimum — 16 GB VRAM)
- ~22 GB scratch storage

---

## Setup (One-Time — Already Done If You Are Continuing a Study)

### Step 1 — Upload project to HPRC (Mac Terminal)

```bash
scp PPRI_MAIN.zip saher02@grace.hprc.tamu.edu:/scratch/user/saher02/
```

Then on HPRC:
```bash
cd $SCRATCH
unzip PPRI_MAIN.zip
```

### Step 2 — Create conda environment (HPRC)

```bash
module purge
module load Anaconda3/2024.02-1
source /sw/eb/sw/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda create -p /scratch/user/saher02/ppri_env python=3.11 -y
conda activate /scratch/user/saher02/ppri_env
```

### Step 3 — Install packages (HPRC)

```bash
export PYTHONNOUSERSITE=1

# Install torch FIRST with correct CUDA version
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Then the rest
pip install -r requirements.txt

# Poppler and Tesseract via conda (system modules unavailable on Grace)
conda install -c conda-forge poppler tesseract -y
```

### Step 4 — Download AI model (HPRC login node — has internet)

```bash
export HF_HOME=/scratch/user/saher02/hf_cache
mkdir -p $HF_HOME

python -c "
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')
AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')
print('Model downloaded successfully.')
"
```

> ⚠️ This must be run on the **login node** — compute nodes have no internet.

### Step 5 — Optional speed fix

In `pipeline/llm_extractor.py`, change:
```python
max_new_tokens=4096,   # slow
```
to:
```python
max_new_tokens=2048,   # ~2x faster, no quality loss
```

---

## Every Run

### 1. Activate environment (run every SSH session)

```bash
# HPRC
module purge
module load Anaconda3/2024.02-1
source /sw/eb/sw/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate /scratch/user/saher02/ppri_env
export HF_HOME=/scratch/user/saher02/hf_cache
export PYTHONNOUSERSITE=1
cd "$SCRATCH/PPRI 3"
```

### 2. Upload PDFs (Mac Terminal)

```bash
# PRE surveys
scp /path/to/scans/pre/*.pdf 'saher02@grace.hprc.tamu.edu:/scratch/user/saher02/PPRI 3/PRE/'

# POST surveys  
scp /path/to/scans/post/*.pdf 'saher02@grace.hprc.tamu.edu:/scratch/user/saher02/PPRI 3/POST/'
```

> Always use **single quotes** — the space in "PPRI 3" breaks double-quoted paths.

### 3. Update study name (HPRC)

```bash
nano "$SCRATCH/PPRI 3/.env"
# Change: STUDY_NAME=YourStudyName
```

### 4. Clear old output for a new study (HPRC)

```bash
cd "$SCRATCH/PPRI 3"
rm -f output/checkpoints/*.json
rm -f output/*.xlsx output/*.csv output/pipeline.log
```

> Skip this step if resuming an interrupted run — checkpoints let it pick up where it stopped.

### 5. Submit job (HPRC)

```bash
sbatch run_pipeline.slurm
squeue -u saher02           # check status
tail -f slurm-JOBID.out     # watch live output
```

### 6. Download results (Mac Terminal)

```bash
scp 'saher02@grace.hprc.tamu.edu:/scratch/user/saher02/PPRI 3/output/STUDYNAME_results.xlsx' ~/Desktop/
scp 'saher02@grace.hprc.tamu.edu:/scratch/user/saher02/PPRI 3/output/STUDYNAME_PRE_results.csv' ~/Desktop/
scp 'saher02@grace.hprc.tamu.edu:/scratch/user/saher02/PPRI 3/output/STUDYNAME_POST_results.csv' ~/Desktop/
```

---

## Configuration (.env)

| Setting | Default | Description |
|---|---|---|
| `STUDY_NAME` | `MockStudy` | Used in output filenames |
| `PRE_FOLDER` | `./PRE` | Folder containing PRE PDFs |
| `POST_FOLDER` | `./POST` | Folder containing POST PDFs |
| `HF_MODEL` | `Qwen/Qwen2-VL-2B-Instruct` | AI model (runs locally) |
| `CONFIDENCE_THRESHOLD` | `0.9` | Below this → flagged for review |
| `DPI` | `300` | OCR image quality (does not affect AI speed) |
| `PRE_SCHEMA_FILE` | `forms/falls_prevention_pre.yaml` | Schema for PRE surveys |
| `POST_SCHEMA_FILE` | `forms/falls_prevention_post.yaml` | Schema for POST surveys |
| `TRANSFORMERS_OFFLINE` | `1` | Must stay 1 — compute nodes have no internet |

---

## Output Files

| File | Contents |
|---|---|
| `{STUDY_NAME}_results.xlsx` | Excel workbook — Summary, PRE, POST sheets. Yellow rows = flagged. |
| `{STUDY_NAME}_PRE_results.csv` | Flat CSV of PRE results |
| `{STUDY_NAME}_POST_results.csv` | Flat CSV of POST results |
| `pipeline.log` | Full timestamped run log |
| `checkpoints/*.json` | Resume state — delete only when starting a new study |

---

## Adding a New Study Schema

Copy the example schema and fill in your questions:

```bash
cp forms/example_pre_schema.yaml forms/mystudy_pre.yaml
nano forms/mystudy_pre.yaml
```

Then update `.env`:
```
PRE_SCHEMA_FILE=forms/mystudy_pre.yaml
POST_SCHEMA_FILE=forms/mystudy_post.yaml
```

Supported question types: `checkbox`, `true_false`, `multiple_choice`, `scale`, `numeric`, `handwritten`, `fill_blank`

---

## GPU Options

Edit `run_pipeline.slurm` to switch GPU:

```bash
#SBATCH --gres=gpu:t4:1      # T4 — 16GB VRAM (default, ~3–6 min/PDF)
#SBATCH --gres=gpu:a100:1    # A100 — 40GB VRAM (faster, ~1–2 min/PDF)
##SBATCH --gres=gpu:t4:1     # CPU only (double # = disabled, ~10–15 min/PDF)
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'torch'` | Run activation block — conda env not active |
| `Network is unreachable` | Download model on login node first (Step 4 above) |
| `Poppler not installed` | `conda install -c conda-forge poppler -y` |
| `Tesseract not in PATH` | `conda install -c conda-forge tesseract -y` |
| `already processed (checkpoint)` | `rm -f output/checkpoints/*.json` then resubmit |
| `scp: No such file or directory` | Use single quotes `'` around remote path with space |
| Job stuck PENDING | Switch GPU: `#SBATCH --gres=gpu:a100:1` |

---

## Built With

- [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) — vision-language model for form extraction
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — open source OCR engine
- [pdf2image](https://github.com/Belval/pdf2image) — PDF to image conversion
- [PyTorch](https://pytorch.org/) — GPU inference
- [TAMU HPRC](https://hprc.tamu.edu/) — Grace cluster compute infrastructure

---

## License

Internal use — PPRI / Texas A&M research organization.