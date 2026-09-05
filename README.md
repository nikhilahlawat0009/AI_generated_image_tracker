# AI Generated Image Tracker

Upload a picture and this tells you whether a machine made it, and why it thinks so.

It is a web app with a Python backend, and the part I find interesting is that it refuses to trust any single method. Three separate checks run on every image, each gets a vote, and a Claude vision agent writes up the conclusion in language you would actually say out loud.

## Why three checks instead of one

Any single detector can be fooled. A model that has only ever seen Stable Diffusion images will be confidently wrong the first time you show it something from a newer generator. So rather than one opinion there are three, and they look for completely different things.

**The trained model** takes in the whole frame at once. This is a Vision Transformer, and that choice matters here. AI images tend to go wrong globally rather than in one spot: hands with an extra finger, light arriving from two directions at the same time, a reflection that does not match what is in front of it. Those are relationships across the entire picture, so a model that looks everywhere simultaneously catches them better than one sliding a small window around.

**The frequency check** ignores the picture completely and looks at its mathematical fingerprint. Generators leave behind faint regular patterns that cameras never produce, a kind of graph paper texture that is invisible to you and obvious once the image is transformed.

**The metadata check** just reads the file. Real photographs usually carry a trail: which camera, what shutter speed, when it was taken. Generated images turn up with that trail missing, and a surprising number of them politely embed the name of the tool that made them.

Their votes are weighted into a single answer. When the three disagree strongly the app says it is unsure instead of picking a side, which seemed more honest than a confident coin flip.

## How well it does

Tested on 1,500 images it had never seen before:

| | |
|---|---|
| Correct verdicts | 96.4% |
| AI images caught | 96.5% |
| Real photos wrongly flagged | 28 out of 750 |
| AUC-ROC | 0.9918 |

The false positives are the ones worth caring about. Telling someone their own photograph is fake is a worse failure than missing a generated one, so that number is the one to keep pushing down.

## What it is built from

The training data started as CIFAKE, a published dataset of 60,000 real photographs paired with 60,000 Stable Diffusion equivalents. That was a good start and a narrow one, since it teaches the model exactly one generator. The pipeline in `data_pipeline/` now pulls from several sources and merges them into a single balanced pool of about 40,000 images, keeping a note of where each one came from. Adding a new generator means adding one entry to `registry.py`, not writing new download code.

Training freezes most of the network and fine-tunes only the last two transformer blocks plus the classifier, which is roughly a sixth of the parameters. About 25 minutes on an Apple Silicon laptop. There is a `--resume` flag for continuing from an existing checkpoint when new data arrives, rather than starting again from nothing.

## Running it

Backend:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

Put your Anthropic key in `.env`. Frontend:

```bash
cd frontend
npm install
npm run dev
```

Or `./start.sh` to bring up both, then open http://localhost:5173.

## Rebuilding the model yourself

```bash
cd backend
python3 data_pipeline/download_all.py      # pull every source in registry.py
python3 data_pipeline/merge_datasets.py    # combine into one pool
python3 data_pipeline/dataset_curator.py ./merged_data ./curated_data_v2
python3 training/finetune_vit.py --data ./curated_data_v2 --epochs 5
python3 training/evaluate.py --data ./curated_data_v2 --model ./model_output_v2/best_model.pth
```

The curator deduplicates by hash and splits 70/15/15, then checks that no image ended up in more than one split. Leakage between train and test is the quiet way to end up with a number that looks excellent and means nothing.

## Layout

```
frontend/                React, Vite, Tailwind
backend/
  main.py                FastAPI: upload, analyse, history
  detectors/             the three checks, plus the weighting that combines them
  agents/                the Claude vision agent that writes the report
  data_pipeline/         downloading, merging, curating, splitting
  training/              fine-tuning and evaluation
  db/                    SQLite, for keeping past analyses
```

## Honest limitations

The model has seen a handful of generators, not all of them, and new ones appear constantly. Metadata is trivial to strip or fake, so that check helps and cannot be leaned on. The frequency check gets weaker once an image has been through screenshotting, resizing, or a social media pipeline, all of which scrub exactly the artefacts it hunts for. Treat the verdict as good evidence rather than proof.
