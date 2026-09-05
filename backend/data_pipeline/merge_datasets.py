"""
Merge Dataset Sources
======================

Combines all downloaded sources into a single flat pool:
  merged_data/ai_generated/   ← all AI images from all sources
  merged_data/real/           ← all real images from all sources

Images are renamed to include their source prefix so you always
know where an image came from:
  cifake_ai_00001.jpg
  aiornot_real_00042.jpg

WHY MERGE BEFORE CURATING:
  The curator (dataset_curator.py) does deduplication with MD5 hashing
  across ALL images at once. If we curated per-source and then combined,
  we would miss cross-source duplicates — the same real photo appearing
  in two different datasets would cause data leakage between train and test.

  Example of cross-source leakage:
    CIFAKE real images come from CIFAR-10.
    Some HuggingFace datasets also source from CIFAR-10.
    Without cross-dataset dedup, the same image could be in both
    train (from source A) and test (from source B). AUC-ROC looks great.
    Deploy. Fails on new data. Classic leakage.

Run:
  cd backend
  python data_pipeline/merge_datasets.py --input ./raw_data --output ./merged_data
"""

import argparse
import hashlib
import shutil
from pathlib import Path


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def merge(input_dir: Path, output_dir: Path):
    ai_out   = output_dir / "ai_generated"
    real_out = output_dir / "real"
    ai_out.mkdir(parents=True, exist_ok=True)
    real_out.mkdir(parents=True, exist_ok=True)

    sources = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not sources:
        print(f"[!] No source folders found in {input_dir}")
        return

    print(f"Merging {len(sources)} source(s) from {input_dir}\n")

    seen_hashes = set()
    duplicates  = 0
    total_ai    = 0
    total_real  = 0

    for source_dir in sources:
        name = source_dir.name
        for cls, out_dir, counter_attr in [
            ("ai_generated", ai_out,   "total_ai"),
            ("real",         real_out, "total_real"),
        ]:
            src_folder = source_dir / cls
            if not src_folder.exists():
                continue

            images = sorted(src_folder.glob("*.jpg")) + sorted(src_folder.glob("*.png")) + sorted(src_folder.glob("*.webp"))
            copied = 0

            for img_path in images:
                h = md5(img_path)
                if h in seen_hashes:
                    duplicates += 1
                    continue
                seen_hashes.add(h)

                # Rename to source_class_NNNNN.jpg for traceability
                idx = (total_ai if cls == "ai_generated" else total_real) + copied
                dst = out_dir / f"{name}_{img_path.stem}_{idx:05d}.jpg"
                shutil.copy2(img_path, dst)
                copied += 1

            if cls == "ai_generated":
                total_ai += copied
            else:
                total_real += copied

            print(f"  {name}/{cls}: {copied} images merged")

    print(f"\n{'='*50}")
    print(f"MERGE SUMMARY")
    print(f"{'='*50}")
    print(f"  Total AI images   : {total_ai}")
    print(f"  Total real images : {total_real}")
    print(f"  Cross-source duplicates removed: {duplicates}")
    print(f"  Output: {output_dir}")
    print(f"\nNext step:")
    print(f"  python data_pipeline/dataset_curator.py {output_dir} ./curated_data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Folder containing per-source downloads (raw_data/)")
    parser.add_argument("--output", required=True, help="Output folder for merged pool (merged_data/)")
    args = parser.parse_args()

    merge(Path(args.input), Path(args.output))
