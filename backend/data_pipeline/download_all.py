"""
Download All Dataset Sources
=============================

Runs every source defined in registry.py and saves images to:
  raw_data/<source_name>/ai_generated/
  raw_data/<source_name>/real/

After this, run merge_datasets.py to combine all sources,
then dataset_curator.py to deduplicate and split.

WHY SEPARATE FOLDERS PER SOURCE:
  Keeping sources separate lets you:
  - Re-download one source without touching others
  - Inspect per-source quality before merging
  - Audit which generator an image came from
  - Remove a source later if it turns out to be low quality

ADDING A NEW SOURCE IN THE FUTURE:
  1. Add an entry to registry.py SOURCES list
  2. Run: python download_all.py --sources <name>
     (downloads only that source, skips the rest)
  3. Run merge_datasets.py and dataset_curator.py again

Run:
  cd backend
  python data_pipeline/download_all.py --output ./raw_data
  python data_pipeline/download_all.py --output ./raw_data --sources cifake aiornot
"""

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_pipeline.registry import SOURCES


def get_loader(loader_name: str):
    module = importlib.import_module(f"data_pipeline.downloaders.{loader_name}")
    return module.download


def download_all(output_dir: Path, only: Optional[list] = None):
    sources = SOURCES
    if only:
        sources = [s for s in SOURCES if s["name"] in only]
        missing = set(only) - {s["name"] for s in sources}
        if missing:
            print(f"[!] Unknown sources: {missing}")
            print(f"    Available: {[s['name'] for s in SOURCES]}")

    print(f"\nDownloading {len(sources)} source(s) → {output_dir}\n")
    print(f"{'Source':<16} {'Description'}")
    print("─" * 60)
    for s in sources:
        print(f"  {s['name']:<14} {s['description']}")
    print()

    summary = []
    total_start = time.time()

    for source in sources:
        source_dir = output_dir / source["name"]
        ai_done    = len(list((source_dir / "ai_generated").glob("*.jpg"))) if (source_dir / "ai_generated").exists() else 0
        real_done  = len(list((source_dir / "real").glob("*.jpg")))        if (source_dir / "real").exists()         else 0

        if ai_done >= source["samples"] and real_done >= source["samples"]:
            print(f"[✓] {source['name']} — already downloaded ({real_done} real, {ai_done} AI), skipping")
            summary.append((source["name"], real_done, ai_done, "skipped"))
            continue

        print(f"[→] {source['name']} — {source['description']}")
        t0 = time.time()
        try:
            loader = get_loader(source["loader"])
            real_n, ai_n = loader(source, source_dir)
            elapsed = time.time() - t0
            print(f"    Completed in {elapsed:.0f}s")
            summary.append((source["name"], real_n, ai_n, "ok"))
        except Exception as e:
            print(f"    [!] FAILED: {e}")
            summary.append((source["name"], 0, 0, f"error: {e}"))
        print()

    elapsed_total = time.time() - total_start
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"{'Source':<16} {'Real':>6} {'AI':>6}  Status")
    print("─" * 60)
    total_real = total_ai = 0
    for name, real_n, ai_n, status in summary:
        print(f"  {name:<14} {real_n:>6} {ai_n:>6}  {status}")
        total_real += real_n
        total_ai   += ai_n
    print("─" * 60)
    print(f"  {'TOTAL':<14} {total_real:>6} {total_ai:>6}")
    print(f"\nTotal time: {elapsed_total:.0f}s")
    print(f"\nNext step:")
    print(f"  python data_pipeline/merge_datasets.py --input {output_dir} --output ./merged_data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",  default="./raw_data", help="Root folder for downloaded sources")
    parser.add_argument("--sources", nargs="*",            help="Download only these sources (by name)")
    args = parser.parse_args()

    download_all(Path(args.output), args.sources)
