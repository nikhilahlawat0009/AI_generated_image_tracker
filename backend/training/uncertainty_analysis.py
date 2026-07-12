"""
Uncertainty Analysis & Abstention Layer
========================================

WHAT THIS DOES:
  1. Loads every test image the model got WRONG and shows how confident
     it was — was it a confident mistake or a hesitant one?

  2. Builds an abstention layer — a confidence zone where the model
     says "I don't know" instead of forcing a wrong answer.

  3. Produces enterprise-ready output: AI / Real / Uncertain
     instead of binary AI / Real.

WHY THIS MATTERS FOR ENTERPRISE:
  Organisations integrating this into a workflow need to know:
    - When can the model be trusted to make the call automatically?
    - When should it escalate to a human reviewer?

  A fraud detection system at a bank doesn't reject every transaction
  it's unsure about — it routes uncertain cases to a human analyst.
  Same principle here.

THE CORE CONCEPT — CONFIDENCE CALIBRATION + ABSTENTION:

  Raw model output: a probability score between 0.0 and 1.0
    0.0 = certain it's REAL
    1.0 = certain it's AI
    0.5 = completely uncertain

  The naive approach: if score > 0.5 → AI, else → Real
  The problem: a score of 0.51 is treated the same as 0.99

  The enterprise approach: define two thresholds
    score > HIGH_THRESHOLD  → call it AI (high confidence)
    score < LOW_THRESHOLD   → call it Real (high confidence)
    LOW_THRESHOLD ≤ score ≤ HIGH_THRESHOLD → Uncertain (human review)

  This creates a "dead zone" where the model abstains.
  You tune these thresholds based on your organisation's risk tolerance.

RUN:
  python3 training/uncertainty_analysis.py \
    --model ./model_output/best_model.pth \
    --data  ./curated_data
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224

# ── Abstention thresholds ─────────────────────────────────────────────────────
# These are tunable. Meaning:
#   If AI probability > 0.80 → call it AI
#   If AI probability < 0.20 → call it Real
#   Between 0.20 and 0.80   → Uncertain, send to human review
#
# For a stricter organisation (e.g. legal, journalism):
#   Use 0.90 / 0.10 — only make a call when very confident
#
# For a lenient system (e.g. social media bulk filtering):
#   Use 0.65 / 0.35 — catch more, accept some errors
HIGH_THRESHOLD = 0.80
LOW_THRESHOLD  = 0.20


def load_model(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, 2)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint.get("class_to_idx", {})


@torch.no_grad()
def collect_predictions(model, loader, device, dataset) -> list[dict]:
    """
    Run inference on the full test set and collect per-image results
    including the image path, true label, predicted label, and confidence.
    """
    all_results = []
    img_paths   = [s[0] for s in dataset.samples]  # full path to each image

    batch_start = 0
    for images, labels in tqdm(loader, desc="Analysing"):
        images  = images.to(device)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1).cpu().numpy()
        preds   = outputs.argmax(dim=1).cpu().numpy()
        labels  = labels.numpy()

        for i in range(len(labels)):
            prob_ai  = float(probs[i][0])   # ai_generated = class 0
            prob_real = float(probs[i][1])
            true_label = "ai_generated" if labels[i] == 0 else "real"
            pred_label = "ai_generated" if preds[i] == 0 else "real"
            correct    = (true_label == pred_label)

            # Abstention decision
            if prob_ai >= HIGH_THRESHOLD:
                decision = "AI"
            elif prob_ai <= LOW_THRESHOLD:
                decision = "Real"
            else:
                decision = "Uncertain"

            all_results.append({
                "path":       img_paths[batch_start + i],
                "true_label": true_label,
                "pred_label": pred_label,
                "prob_ai":    round(prob_ai, 4),
                "prob_real":  round(prob_real, 4),
                "correct":    correct,
                "decision":   decision,
                # How far from the decision boundary (0.5)?
                # 0.0 = maximum uncertainty, 0.5 = maximum confidence
                "confidence_margin": round(abs(prob_ai - 0.5), 4),
            })

        batch_start += len(labels)

    return all_results


def analyse_errors(results: list[dict]):
    """
    Break down the wrong predictions by confidence level.
    This answers: was the model hesitant or confident when it was wrong?
    """
    wrong = [r for r in results if not r["correct"]]
    right = [r for r in results if r["correct"]]

    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    print(f"\nTotal test images : {len(results)}")
    print(f"Correct           : {len(right)}  ({100*len(right)/len(results):.1f}%)")
    print(f"Wrong             : {len(wrong)}  ({100*len(wrong)/len(results):.1f}%)")

    if not wrong:
        print("No errors — perfect model!")
        return

    margins = [r["confidence_margin"] for r in wrong]
    probs   = [r["prob_ai"] for r in wrong]

    print(f"\nAmong the {len(wrong)} wrong predictions:")
    print(f"  Average confidence margin : {np.mean(margins):.3f}  "
          f"(0.0 = totally unsure, 0.5 = very confident)")
    print(f"  Median confidence margin  : {np.median(margins):.3f}")

    # Bucket wrong predictions by confidence
    very_confident_wrong = [r for r in wrong if r["confidence_margin"] > 0.3]
    hesitant_wrong       = [r for r in wrong if r["confidence_margin"] <= 0.15]
    borderline_wrong     = [r for r in wrong
                            if 0.15 < r["confidence_margin"] <= 0.3]

    print(f"\n  Confident mistakes  (margin > 0.30) : {len(very_confident_wrong)}")
    print(f"  Borderline mistakes (0.15–0.30)     : {len(borderline_wrong)}")
    print(f"  Hesitant mistakes   (margin < 0.15) : {len(hesitant_wrong)}")

    print("\nWhat this means:")
    if len(hesitant_wrong) > len(very_confident_wrong):
        print("  Most wrong predictions were near the 0.5 boundary — the model")
        print("  was already unsure. Abstention would catch most of these.")
    else:
        print("  Some wrong predictions were confident — harder cases the model")
        print("  genuinely misjudged. These need better training data or features.")

    # Show the 5 most confidently wrong predictions
    worst = sorted(wrong, key=lambda r: r["confidence_margin"], reverse=True)[:5]
    print(f"\nTop 5 most confidently wrong predictions:")
    print(f"  {'True':>14}  {'Predicted':>14}  {'AI prob':>8}  {'Margin':>8}")
    for r in worst:
        print(f"  {r['true_label']:>14}  {r['pred_label']:>14}  "
              f"{r['prob_ai']:>8.3f}  {r['confidence_margin']:>8.3f}")

    # Break down by error type
    missed_ai     = [r for r in wrong if r["true_label"] == "ai_generated"]
    flagged_real  = [r for r in wrong if r["true_label"] == "real"]

    print(f"\nError breakdown:")
    print(f"  False negatives (AI missed, called Real) : {len(missed_ai)}")
    if missed_ai:
        avg_margin = np.mean([r["confidence_margin"] for r in missed_ai])
        print(f"    Average confidence margin: {avg_margin:.3f}")
        print(f"    These are the dangerous ones — AI images that slip through")

    print(f"  False positives (Real flagged as AI)     : {len(flagged_real)}")
    if flagged_real:
        avg_margin = np.mean([r["confidence_margin"] for r in flagged_real])
        print(f"    Average confidence margin: {avg_margin:.3f}")
        print(f"    These cause false alarms — annoying but less dangerous")


def analyse_abstention(results: list[dict], high: float, low: float):
    """
    Show what happens when we add the abstention layer.

    WHAT IS ABSTENTION (for enterprise integration):
      Instead of always outputting AI or Real, we define a confidence
      zone where the model says "I need a human to check this."

      Think of it like a doctor:
        Clear scan → computer gives verdict automatically
        Ambiguous scan → flagged for radiologist review

      The key metric is: how many of the WRONG predictions fall
      in the uncertain zone? If abstention captures most errors,
      it's working.
    """
    auto_ai      = [r for r in results if r["decision"] == "AI"]
    auto_real    = [r for r in results if r["decision"] == "Real"]
    uncertain    = [r for r in results if r["decision"] == "Uncertain"]

    errors_in_uncertain = [r for r in uncertain if not r["correct"]]
    errors_in_auto      = [r for r in results
                           if r["decision"] != "Uncertain" and not r["correct"]]

    print("\n" + "="*60)
    print(f"ABSTENTION ANALYSIS  (thresholds: AI>{high}, Real<{low})")
    print("="*60)

    total = len(results)
    print(f"\nDecision breakdown:")
    print(f"  Auto → AI       : {len(auto_ai):4}  ({100*len(auto_ai)/total:.1f}%)")
    print(f"  Auto → Real     : {len(auto_real):4}  ({100*len(auto_real)/total:.1f}%)")
    print(f"  Uncertain (hold): {len(uncertain):4}  ({100*len(uncertain)/total:.1f}%)")

    # Accuracy on auto-decided cases only
    auto_cases = [r for r in results if r["decision"] != "Uncertain"]
    if auto_cases:
        auto_correct = sum(1 for r in auto_cases if r["correct"])
        auto_acc     = auto_correct / len(auto_cases)
        print(f"\nOn auto-decided cases only:")
        print(f"  Accuracy : {auto_acc:.4f}  ({auto_acc*100:.1f}%)")
        print(f"  This is the accuracy the organisation actually experiences")
        print(f"  on cases the model handles without human review")

    print(f"\nError routing:")
    print(f"  Errors caught by abstention  : {len(errors_in_uncertain)} "
          f"({100*len(errors_in_uncertain)/max(len([r for r in results if not r['correct']]),1):.1f}% of all errors)")
    print(f"  Errors still in auto zone    : {len(errors_in_auto)}")

    print(f"\nOrganisational interpretation:")
    print(f"  {len(uncertain)} images ({100*len(uncertain)/total:.1f}%) go to human review queue")
    print(f"  {len(auto_cases)} images ({100*len(auto_cases)/total:.1f}%) are decided automatically")
    if auto_cases:
        print(f"  Automatic decisions are {auto_acc*100:.1f}% accurate")
        print(f"  Only {len(errors_in_auto)} mistakes reach the end decision — "
              f"the rest are caught by review")

    print(f"\nTuning guidance:")
    print(f"  Tighten thresholds (e.g. 0.90/0.10):")
    print(f"    → More images go to review, fewer mistakes in auto zone")
    print(f"    → Better for high-stakes orgs (legal, journalism, finance)")
    print(f"  Loosen thresholds (e.g. 0.65/0.35):")
    print(f"    → Fewer images go to review, more mistakes slip through")
    print(f"    → Acceptable for bulk filtering at scale (social media)")


def run(model_path: Path, data_dir: Path,
        high: float = HIGH_THRESHOLD, low: float = LOW_THRESHOLD):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model, class_to_idx = load_model(model_path, device)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_ds     = datasets.ImageFolder(data_dir / "test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=32,
                             shuffle=False, num_workers=2)

    results = collect_predictions(model, test_loader, device, test_ds)

    # Analysis 1: What went wrong and how confident was the model?
    analyse_errors(results)

    # Analysis 2: What does adding abstention do to the system?
    analyse_abstention(results, high, low)

    # Save full per-image results for the dashboard
    out_path = model_path.parent / "uncertainty_analysis.json"
    with open(out_path, "w") as f:
        json.dump({
            "thresholds": {"high": high, "low": low},
            "summary": {
                "total": len(results),
                "correct": sum(1 for r in results if r["correct"]),
                "wrong":   sum(1 for r in results if not r["correct"]),
                "uncertain": sum(1 for r in results if r["decision"] == "Uncertain"),
            },
            "per_image": results,
        }, f, indent=2)

    print(f"\nFull per-image results saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data",  required=True)
    parser.add_argument("--high",  type=float, default=HIGH_THRESHOLD,
                        help="Above this → call AI automatically")
    parser.add_argument("--low",   type=float, default=LOW_THRESHOLD,
                        help="Below this → call Real automatically")
    args = parser.parse_args()

    run(Path(args.model), Path(args.data), args.high, args.low)
