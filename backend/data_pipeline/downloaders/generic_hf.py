"""
Generic HuggingFace Downloader
===============================

Handles any HuggingFace image dataset that follows the standard pattern:
  - One field for the image (PIL Image)
  - One field for the label (int or str)
  - Configurable ai_label and real_label values

All configuration comes from registry.py — this file never needs to change
when a new dataset is added. Just add an entry to SOURCES in registry.py.

WHY A GENERIC DOWNLOADER:
  Most image datasets on HuggingFace follow the same schema.
  Writing a bespoke downloader per dataset is unnecessary duplication.
  The only things that vary are: dataset ID, split name, field names,
  and label values — all of which live in the registry entry.
"""

from pathlib import Path


def download(source: dict, output_dir: Path):
    try:
        from datasets import load_dataset
        from PIL import Image
    except ImportError:
        raise ImportError("Run: pip install datasets Pillow")

    ai_dir   = output_dir / "ai_generated"
    real_dir = output_dir / "real"
    ai_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    samples = source["samples"]
    print(f"  Streaming {source['hf_dataset']} (up to {samples} per class)...")

    ds = load_dataset(source["hf_dataset"], streaming=True)
    split = ds.get(source["split"])
    if split is None:
        available = list(ds.keys())
        print(f"  [!] Split '{source['split']}' not found. Available: {available}")
        split = ds[available[0]]
        print(f"  [!] Falling back to '{available[0]}'")

    real_count = ai_count = 0
    skipped = 0

    for example in split:
        if real_count >= samples and ai_count >= samples:
            break

        label = example.get(source["label_key"])
        img   = example.get(source["image_key"])

        if img is None or label is None:
            skipped += 1
            continue

        # Normalise label — some datasets use string labels ("real", "fake")
        # others use ints. Compare as strings to handle both.
        label_str = str(label)
        ai_str    = str(source["ai_label"])
        real_str  = str(source["real_label"])

        if label_str == real_str and real_count < samples:
            # Convert to RGB — some datasets include RGBA or grayscale
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(real_dir / f"real_{real_count:05d}.jpg", quality=95)
            real_count += 1
        elif label_str == ai_str and ai_count < samples:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(ai_dir / f"ai_{ai_count:05d}.jpg", quality=95)
            ai_count += 1

        if (real_count + ai_count) % 500 == 0 and (real_count + ai_count) > 0:
            print(f"    {real_count} real, {ai_count} AI...", end="\r")

    if skipped:
        print(f"  [!] Skipped {skipped} examples with missing image or label")

    print(f"    Done: {real_count} real, {ai_count} AI saved to {output_dir}")
    return real_count, ai_count
