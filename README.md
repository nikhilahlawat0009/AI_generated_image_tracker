# AI Generated Image Tracker

A full-stack web app that detects AI-generated images using a multi-signal detection pipeline — combining a fine-tuned Vision Transformer, frequency domain analysis, metadata forensics, and a Claude AI forensic agent.

**Author:** Nikhil Ahlawat

---

## What it does

Upload any image and the system tells you:
- Whether it is AI-generated or a real photograph
- Confidence score (0–100%)
- Which signals triggered (spectral artifacts, missing EXIF, generator signatures)
- A detailed forensic report written by a Claude vision agent

---

## Architecture

```
frontend/          React + Vite + TailwindCSS
backend/
  main.py          FastAPI — image upload, analysis endpoint, history
  detectors/
    model_detector.py      Fine-tuned ViT-B/16 classifier
    frequency_analyzer.py  FFT spectral artifact detection
    metadata_analyzer.py   EXIF and PNG chunk forensics
    ensemble.py            Weighted scoring → verdict
  agents/
    provenance_agent.py    Claude vision agent — forensic report
  data_pipeline/
    download_dataset.py    CIFAKE dataset downloader (HuggingFace streaming)
    dataset_curator.py     Stratified split, deduplication, leakage check
  training/
    finetune_vit.py        Transfer learning fine-tuning script
    evaluate.py            Precision, recall, AUC-ROC evaluation
  db/
    database.py            SQLite persistence for analysis history
```

---

## ML Pipeline

### Dataset
- **CIFAKE** (Bird & Lotfi, 2023) — 60k real CIFAR-10 photographs + 60k Stable Diffusion equivalents
- Downloaded 10,000 images (5,000 per class) via HuggingFace streaming
- Deduplicated using MD5 hashing — removed 7 duplicate AI images
- Stratified 70/15/15 split → 6,995 train / 1,498 val / 1,500 test
- Leakage verification: zero overlap between splits confirmed

### Model
- **Architecture:** ViT-B/16 (Vision Transformer, 86M parameters)
- **Training:** Transfer learning — froze first 10 of 12 transformer blocks, fine-tuned last 2 + classification head (16.5% of parameters trainable)
- **Why ViT over CNN:** AI generation artifacts are often global (anatomically inconsistent hands, incoherent lighting). ViT's self-attention attends to the full image simultaneously, catching long-range inconsistencies that CNN convolution misses.
- **Hardware:** Apple Silicon GPU (MPS), 5 epochs, ~25 minutes

### Results

| Metric | Score |
|---|---|
| Accuracy | 96.4% |
| Precision | 96.3% |
| Recall | 96.5% |
| F1 Score | 96.4% |
| AUC-ROC | **0.9918** |

Confusion matrix on 1,500 held-out test images:
- 724 AI images correctly detected
- 26 AI images missed (false negatives)
- 28 real images falsely flagged (false positives)
- 722 real images correctly passed

### Detection signals (ensemble)

| Detector | Weight | Method |
|---|---|---|
| Fine-tuned ViT | 50% | Neural classifier trained on CIFAKE |
| Frequency analysis | 25% | FFT spectral flatness, 1/f noise slope, GAN grid peaks |
| Metadata forensics | 25% | EXIF absence, PNG tEXt chunk AI signatures |

---

## Running locally

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # add your ANTHROPIC_API_KEY
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Or use the start script (starts both):
```bash
./start.sh
```

Open http://localhost:5173

### Reproducing the ML pipeline

```bash
cd backend

# 1. Download dataset
pip install datasets Pillow torch torchvision scikit-learn
python3 data_pipeline/download_dataset.py --output ./raw_data --samples 5000

# 2. Curate and split
python3 data_pipeline/dataset_curator.py ./raw_data ./curated_data

# 3. Fine-tune ViT
python3 training/finetune_vit.py --data ./curated_data --output ./model_output --epochs 5

# 4. Evaluate
python3 training/evaluate.py --data ./curated_data --model ./model_output/best_model.pth
```

---

## Tech stack

- **Backend:** Python, FastAPI, PyTorch, HuggingFace Transformers, scikit-learn, SQLAlchemy
- **Frontend:** React, TypeScript, Vite, TailwindCSS, Recharts
- **AI Agent:** Anthropic Claude (vision + tool use)
- **Dataset:** CIFAKE — Bird & Lotfi, 2023
