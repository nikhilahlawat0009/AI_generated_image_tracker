"""
Dataset Registry
================

This is the single place to add, remove, or configure dataset sources.
To add a new generator's data in the future:
  1. Add an entry to SOURCES below.
  2. If it needs custom download logic, add a file to downloaders/.
     Otherwise generic_hf.py handles any standard HuggingFace image dataset.
  3. Run download_all.py — everything else is automatic.

SOURCES schema:
  name        : short slug, used as the output folder name (raw_data/<name>/)
  description : what generator/dataset this is
  loader      : which downloader to use ("cifake" or "generic_hf")
  hf_dataset  : HuggingFace dataset ID
  split       : dataset split to stream from (usually "train")
  image_key   : field name for the image in the dataset
  label_key   : field name for the label
  ai_label    : value of label_key that means AI-generated
  real_label  : value of label_key that means real photo
  samples     : max images per class to download from this source

WHY PER-SOURCE SAMPLE CAPS:
  Prevents any single generator from dominating the training distribution.
  If one source has 100k images and another has 2k, the model learns
  that source's artifacts, not the general concept of AI generation.
  Equal caps per source = balanced representation of generators.
"""

SOURCES = [
    {
        "name":        "cifake",
        "description": "CIFAKE — Stable Diffusion v1.4, 32×32 (Bird & Lotfi 2023)",
        "loader":      "cifake",
        "hf_dataset":  "dragonintelligence/CIFAKE-image-dataset",
        "split":       "train",
        "image_key":   "image",
        "label_key":   "label",
        "ai_label":    0,
        "real_label":  1,
        "samples":     5000,
    },
    {
        "name":        "aiornot",
        "description": "AI or Not — mixed generators (SDXL, DALL-E, MJ) vs real photos",
        "loader":      "generic_hf",
        "hf_dataset":  "competitions/aiornot",
        "split":       "train",
        "image_key":   "image",
        "label_key":   "label",
        "ai_label":    1,
        "real_label":  0,
        "samples":     5000,
    },
    {
        "name":        "hemg_aivreal",
        "description": "Hemg AI-Generated vs Real Images — multiple generators vs real photos",
        "loader":      "generic_hf",
        "hf_dataset":  "Hemg/AI-Generated-vs-Real-Images-Datasets",
        "split":       "train",
        "image_key":   "image",
        "label_key":   "label",
        "ai_label":    1,
        "real_label":  0,
        "samples":     5000,
    },
    {
        "name":        "deepfake_real",
        "description": "JamieWithofs Deepfake and real images — deepfake faces vs real photos",
        "loader":      "generic_hf",
        "hf_dataset":  "JamieWithofs/Deepfake-and-real-images-3",
        "split":       "train",
        "image_key":   "image",
        "label_key":   "label",
        "ai_label":    1,
        "real_label":  0,
        "samples":     5000,
    },
]
