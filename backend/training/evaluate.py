"""
Model Evaluation — Precision, Recall, AUC-ROC, Confusion Matrix
================================================================

WHAT THIS DOES:
  Loads the best saved model and the test set, runs inference,
  and produces a full evaluation report with all the metrics that
  matter in ML interviews and production systems.

WHY NOT JUST USE ACCURACY:
  Accuracy = (correct predictions) / (total predictions)

  This is misleading when:
  1. Classes are imbalanced — a model saying "real" always gets
     high accuracy if most images are real
  2. The cost of errors is asymmetric — missing an AI image (false
     negative) might be much worse than a false alarm (false positive)
     depending on your use case

  The four metrics we compute give a complete picture:
    - Precision: quality of positive predictions
    - Recall:    coverage of actual positives
    - F1:        harmonic mean of precision and recall
    - AUC-ROC:   ranking quality independent of threshold

RUN:
  python3 training/evaluate.py \
    --data  ./curated_data \
    --model ./model_output/best_model.pth
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224


def load_model(model_path: Path, device: torch.device) -> tuple:
    """Load saved checkpoint and reconstruct the model."""
    checkpoint = torch.load(model_path, map_location=device)

    # Reconstruct the same architecture we trained
    model = models.vit_b_16(weights=None)   # no pretrained weights — we load ours
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, 2)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, checkpoint.get("class_to_idx", {})


@torch.no_grad()
def run_inference(model, loader, device) -> dict:
    """Run model on all test images and collect predictions + probabilities."""
    all_probs, all_preds, all_labels = [], [], []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.tolist())

    return {
        "probs":  np.array(all_probs),    # shape [N, 2]
        "preds":  np.array(all_preds),    # shape [N]
        "labels": np.array(all_labels),   # shape [N]
    }


def evaluate(data_dir: Path, model_path: Path):
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Load model
    model, class_to_idx = load_model(model_path, device)
    print(f"Class mapping: {class_to_idx}")

    # ai_generated=0, real=1  (alphabetical order from ImageFolder)
    # We define "positive" = ai_generated = class 0
    # So "prob_ai" = probs[:, 0]
    ai_class_idx = class_to_idx.get("ai_generated", 0)

    # Load test set
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_ds     = datasets.ImageFolder(data_dir / "test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    # Run inference
    results = run_inference(model, test_loader, device)
    probs_ai = results["probs"][:, ai_class_idx]  # probability of being AI-generated
    preds    = results["preds"]
    labels   = results["labels"]

    # ── 1. Accuracy ───────────────────────────────────────────────────────────
    accuracy = (preds == labels).mean()

    # ── 2. Confusion Matrix ───────────────────────────────────────────────────
    # Layout:
    #              Predicted Real  Predicted AI
    # Actual Real  [TN]            [FP]
    # Actual AI    [FN]            [TP]
    #
    # TN = True Negative  (real, predicted real)    ← correct
    # FP = False Positive (real, predicted AI)      ← false alarm
    # FN = False Negative (AI, predicted real)      ← missed AI image
    # TP = True Positive  (AI, predicted AI)        ← correct
    #
    # The most dangerous error for our use case is FN — an AI image
    # that slips through and gets labelled as real.
    cm = confusion_matrix(labels, preds)

    # Since ai_generated=0, real=1:
    # positive class = 0 (AI), negative class = 1 (real)
    # sklearn confusion_matrix orders by class index
    # cm[0][0] = predicted 0 when actually 0 = TP (ai correctly detected)
    # cm[0][1] = predicted 1 when actually 0 = FN (ai missed)
    # cm[1][0] = predicted 0 when actually 1 = FP (real flagged as AI)
    # cm[1][1] = predicted 1 when actually 1 = TN (real correctly passed)

    # ── 3. Precision, Recall, F1 ──────────────────────────────────────────────
    # pos_label=ai_class_idx means we treat "AI generated" as the positive class
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, pos_label=ai_class_idx, average="binary"
    )

    # ── 4. AUC-ROC ────────────────────────────────────────────────────────────
    # Converts labels to binary: 1 = AI generated, 0 = real
    binary_labels = (labels == ai_class_idx).astype(int)
    auc_roc = roc_auc_score(binary_labels, probs_ai)

    # ROC curve points (used for plotting)
    fpr, tpr, thresholds_roc = roc_curve(binary_labels, probs_ai)

    # ── 5. Precision-Recall curve ─────────────────────────────────────────────
    # More informative than ROC when classes are imbalanced
    prec_curve, rec_curve, thresholds_pr = precision_recall_curve(
        binary_labels, probs_ai
    )

    # ── 6. Threshold analysis ─────────────────────────────────────────────────
    # The default threshold is 0.5 — but is that the best choice?
    # We sweep all thresholds and find the one that maximises F1.
    # This is called "threshold tuning" and it's a real engineering decision.
    #
    # INTERVIEW ANSWER:
    # "The default 0.5 threshold optimises for balanced precision/recall.
    #  But in production, the right threshold depends on the use case.
    #  If we're flagging images for human review, we want high recall
    #  (don't miss AI images) and can tolerate lower precision (some false alarms).
    #  I found the F1-optimal threshold by sweeping and logged all threshold
    #  performance to let the product team make the final call."

    best_f1, best_thresh = 0.0, 0.5
    threshold_analysis = []

    for thresh in np.arange(0.1, 0.95, 0.05):
        preds_at_thresh = (probs_ai >= thresh).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            binary_labels, preds_at_thresh, average="binary", zero_division=0
        )
        threshold_analysis.append({
            "threshold": round(float(thresh), 2),
            "precision": round(float(p), 4),
            "recall":    round(float(r), 4),
            "f1":        round(float(f), 4),
        })
        if f > best_f1:
            best_f1    = f
            best_thresh = thresh

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("EVALUATION REPORT")
    print("="*55)
    print(f"\nAccuracy  : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"Precision : {precision:.4f}  ({precision*100:.1f}%)")
    print(f"Recall    : {recall:.4f}  ({recall*100:.1f}%)")
    print(f"F1 Score  : {f1:.4f}  ({f1*100:.1f}%)")
    print(f"AUC-ROC   : {auc_roc:.4f}")

    print("\nWhat these mean for AI image detection:")
    print(f"  - Of images flagged as AI, {precision*100:.1f}% actually are AI")
    print(f"  - Of actual AI images, {recall*100:.1f}% are caught")
    print(f"  - {(1-recall)*100:.1f}% of AI images slip through undetected")

    print("\nConfusion Matrix:")
    print("                 Predicted AI    Predicted Real")
    print(f"  Actual AI   :  {cm[0][0]:>10}      {cm[0][1]:>10}")
    print(f"  Actual Real :  {cm[1][0]:>10}      {cm[1][1]:>10}")
    print(f"\n  True Positives  (AI caught)    : {cm[0][0]}")
    print(f"  False Negatives (AI missed)    : {cm[0][1]}  ← most dangerous error")
    print(f"  False Positives (real flagged) : {cm[1][0]}")
    print(f"  True Negatives  (real passed)  : {cm[1][1]}")

    print(f"\nOptimal threshold (max F1): {best_thresh:.2f}  (F1={best_f1:.4f})")
    print("Default threshold is 0.5 — adjust based on use case:")
    print("  Lower threshold → higher recall, lower precision (catch more AI)")
    print("  Higher threshold → higher precision, lower recall (fewer false alarms)")

    # ── Save all results ──────────────────────────────────────────────────────
    output = {
        "accuracy":           round(float(accuracy), 4),
        "precision":          round(float(precision), 4),
        "recall":             round(float(recall), 4),
        "f1":                 round(float(f1), 4),
        "auc_roc":            round(float(auc_roc), 4),
        "optimal_threshold":  round(float(best_thresh), 2),
        "confusion_matrix":   cm.tolist(),
        "threshold_analysis": threshold_analysis,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "pr_curve": {
            "precision": prec_curve.tolist(),
            "recall":    rec_curve.tolist(),
        },
    }

    out_path = model_path.parent / "evaluation_report.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull report saved to: {out_path}")
    print("\nNext step:")
    print("  python3 training/calibrate.py --model ./model_output/best_model.pth \\")
    print("                                --eval  ./model_output/evaluation_report.json")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    evaluate(Path(args.data), Path(args.model))
