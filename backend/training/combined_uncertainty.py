"""
Combined Uncertainty Detection — All 3 Signals
===============================================

SIGNAL 1: Confidence score
  The raw AI probability from a single forward pass.
  Catches obvious uncertainty (score near 0.5).

SIGNAL 2: Entropy
  Measures how "spread out" the probability distribution is.
  entropy([0.99, 0.01]) = very low  → confident
  entropy([0.51, 0.49]) = very high → uncertain
  Formula: H = -sum(p * log(p))
  Does not require a label.

SIGNAL 3: Monte Carlo Dropout (MC Dropout)
  Run the same image through the model 30 times with dropout active.
  Measure the standard deviation across runs.
  High std = model disagrees with itself = genuinely uncertain.
  This catches images from NEW generators the model hasn't seen —
  it's the most powerful signal for real-world deployment.

COMBINED RULE:
  Flag for human review if ANY of the three signals says uncertain.
  This is conservative — maximises recall on the uncertain zone.
  You can tighten it to "ALL three must agree" for stricter auto-decisions.

RUN:
  python3 training/combined_uncertainty.py \
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
MC_RUNS       = 30      # number of stochastic forward passes for MC Dropout

# ── Thresholds (tune per organisation) ────────────────────────────────────────
CONF_HIGH     = 0.80    # Signal 1: AI prob above this → confident AI
CONF_LOW      = 0.20    # Signal 1: AI prob below this → confident Real
ENTROPY_LIMIT = 0.50    # Signal 2: entropy above this → uncertain
                        #   max entropy for 2 classes = ln(2) ≈ 0.693
MC_STD_LIMIT  = 0.10    # Signal 3: std across MC runs above this → uncertain


def load_model(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, 2)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model, checkpoint.get("class_to_idx", {})


def enable_dropout(model: nn.Module):
    """
    Switch dropout layers to TRAIN mode even during inference.

    WHY: By default model.eval() disables dropout — every forward pass
    is identical. We need dropout ACTIVE so each of the 30 MC runs
    randomly drops different neurons, giving different outputs.
    The variance across those 30 outputs measures uncertainty.

    We only activate Dropout layers, not BatchNorm — BatchNorm should
    stay in eval mode to use running statistics, not batch statistics.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def compute_entropy(probs: np.ndarray) -> float:
    """
    Shannon entropy of a probability distribution.
    H = -sum(p * log(p))

    For 2 classes:
      [1.0, 0.0] → H = 0.0    (perfectly certain)
      [0.5, 0.5] → H = 0.693  (maximum uncertainty = ln(2))
      [0.8, 0.2] → H = 0.500  (moderate certainty)

    We use this as a threshold: H > 0.50 means uncertain.
    """
    eps = 1e-9
    return float(-np.sum(probs * np.log(probs + eps)))


@torch.no_grad()
def single_pass(model, images, device) -> np.ndarray:
    """Single deterministic forward pass. Returns probabilities."""
    model.eval()
    outputs = model(images.to(device))
    return torch.softmax(outputs, dim=1).cpu().numpy()


@torch.no_grad()
def mc_dropout_pass(model, images, device, n_runs: int) -> tuple:
    """
    MC Dropout: run n_runs stochastic forward passes.
    Returns mean probabilities and std across runs.
    """
    model.eval()
    enable_dropout(model)   # activate dropout for stochastic sampling

    all_probs = []
    for _ in range(n_runs):
        outputs = model(images.to(device))
        probs   = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)

    all_probs = np.array(all_probs)   # shape: [n_runs, batch, 2]
    mean_probs = all_probs.mean(axis=0)  # shape: [batch, 2]
    std_probs  = all_probs.std(axis=0)   # shape: [batch, 2]
    return mean_probs, std_probs


def analyse(model_path: Path, data_dir: Path):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"MC Dropout runs per image: {MC_RUNS}\n")

    model, class_to_idx = load_model(model_path, device)
    ai_idx = class_to_idx.get("ai_generated", 0)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_ds     = datasets.ImageFolder(data_dir / "test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=16,
                             shuffle=False, num_workers=2)

    results = []
    img_paths = [s[0] for s in test_ds.samples]
    batch_start = 0

    for images, labels in tqdm(test_loader, desc="Running 3-signal analysis"):

        # Signal 1 + 2: single pass for confidence and entropy
        probs_single = single_pass(model, images, device)

        # Signal 3: MC Dropout — 30 stochastic passes
        probs_mean, probs_std = mc_dropout_pass(model, images, device, MC_RUNS)

        for i in range(len(labels)):
            true_label = "ai_generated" if labels[i] == 0 else "real"
            prob_ai    = float(probs_single[i][ai_idx])

            # ── Signal 1: Confidence ──────────────────────────────────────────
            conf_uncertain = (CONF_LOW < prob_ai < CONF_HIGH)

            # ── Signal 2: Entropy ─────────────────────────────────────────────
            entropy = compute_entropy(probs_single[i])
            entropy_uncertain = (entropy > ENTROPY_LIMIT)

            # ── Signal 3: MC Dropout std ──────────────────────────────────────
            mc_std = float(probs_std[i][ai_idx])
            mc_uncertain = (mc_std > MC_STD_LIMIT)

            # ── Combined decision ─────────────────────────────────────────────
            # ANY signal uncertain → send to human review
            any_uncertain = conf_uncertain or entropy_uncertain or mc_uncertain

            # For auto-decided cases: use MC mean probability (more reliable)
            mc_prob_ai = float(probs_mean[i][ai_idx])
            if not any_uncertain:
                auto_pred = "ai_generated" if mc_prob_ai >= 0.5 else "real"
            else:
                auto_pred = None   # deferred to human

            correct_if_auto = (auto_pred == true_label) if auto_pred else None

            results.append({
                "path":              img_paths[batch_start + i],
                "true_label":        true_label,
                "prob_ai_single":    round(prob_ai, 4),
                "entropy":           round(entropy, 4),
                "mc_std":            round(mc_std, 4),
                "mc_prob_ai":        round(mc_prob_ai, 4),
                "signal1_uncertain": conf_uncertain,
                "signal2_uncertain": entropy_uncertain,
                "signal3_uncertain": mc_uncertain,
                "any_uncertain":     any_uncertain,
                "auto_pred":         auto_pred,
                "correct_if_auto":   correct_if_auto,
            })

        batch_start += len(labels)

    # ── Results ───────────────────────────────────────────────────────────────
    total      = len(results)
    to_review  = [r for r in results if r["any_uncertain"]]
    auto_cases = [r for r in results if not r["any_uncertain"]]
    auto_correct   = [r for r in auto_cases if r["correct_if_auto"]]
    auto_wrong     = [r for r in auto_cases if not r["correct_if_auto"]]

    # Which signals caught which errors
    original_wrong = [r for r in results if
                      (("ai_generated" if r["prob_ai_single"] >= 0.5 else "real") != r["true_label"])]

    errors_sent_to_review = [r for r in original_wrong if r["any_uncertain"]]
    errors_still_auto     = [r for r in original_wrong if not r["any_uncertain"]]

    print("\n" + "="*60)
    print("COMBINED 3-SIGNAL UNCERTAINTY RESULTS")
    print("="*60)

    print(f"\nSignal thresholds used:")
    print(f"  Signal 1 (confidence) : flag if {CONF_LOW} < prob_ai < {CONF_HIGH}")
    print(f"  Signal 2 (entropy)    : flag if entropy > {ENTROPY_LIMIT}")
    print(f"  Signal 3 (MC Dropout) : flag if std > {MC_STD_LIMIT} across {MC_RUNS} runs")
    print(f"  Rule                  : ANY signal uncertain → human review")

    print(f"\n{'─'*60}")
    print(f"ROUTING BREAKDOWN  (total: {total} images)")
    print(f"{'─'*60}")
    print(f"  → Human review  : {len(to_review):4}  ({100*len(to_review)/total:.1f}%)")
    print(f"  → Auto-decided  : {len(auto_cases):4}  ({100*len(auto_cases)/total:.1f}%)")

    print(f"\n{'─'*60}")
    print(f"AUTO-DECIDED ACCURACY  ({len(auto_cases)} images)")
    print(f"{'─'*60}")
    if auto_cases:
        acc = len(auto_correct) / len(auto_cases)
        print(f"  Correct : {len(auto_correct)}  ({acc*100:.2f}%)")
        print(f"  Wrong   : {len(auto_wrong)}")
        print(f"\n  ✓ Of the {len(auto_cases)} images the system decided automatically,")
        print(f"    {acc*100:.2f}% were correct")
        if auto_wrong:
            print(f"    {len(auto_wrong)} mistakes still slipped through")
        else:
            print(f"    Zero mistakes slipped through")

    print(f"\n{'─'*60}")
    print(f"ERROR ROUTING  (original 54 wrong predictions)")
    print(f"{'─'*60}")
    print(f"  Sent to human review : {len(errors_sent_to_review)}  "
          f"({100*len(errors_sent_to_review)/len(original_wrong):.1f}% of errors caught)")
    print(f"  Still auto-decided   : {len(errors_still_auto)}  "
          f"({100*len(errors_still_auto)/len(original_wrong):.1f}% of errors slip through)")

    print(f"\n{'─'*60}")
    print(f"SIGNAL BREAKDOWN — what each signal flagged")
    print(f"{'─'*60}")
    s1 = sum(1 for r in results if r["signal1_uncertain"])
    s2 = sum(1 for r in results if r["signal2_uncertain"])
    s3 = sum(1 for r in results if r["signal3_uncertain"])
    print(f"  Signal 1 (confidence) flagged : {s1:4}  ({100*s1/total:.1f}%)")
    print(f"  Signal 2 (entropy)    flagged : {s2:4}  ({100*s2/total:.1f}%)")
    print(f"  Signal 3 (MC Dropout) flagged : {s3:4}  ({100*s3/total:.1f}%)")

    # What each signal uniquely contributed
    only_s3 = [r for r in results
               if r["signal3_uncertain"]
               and not r["signal1_uncertain"]
               and not r["signal2_uncertain"]]
    print(f"\n  Images caught ONLY by MC Dropout (not by signals 1 or 2): {len(only_s3)}")
    print(f"  These are cases where the model LOOKED confident on a single")
    print(f"  pass but was actually unstable — MC Dropout caught them.")

    # Save
    out = {
        "thresholds": {
            "conf_high": CONF_HIGH, "conf_low": CONF_LOW,
            "entropy_limit": ENTROPY_LIMIT, "mc_std_limit": MC_STD_LIMIT,
        },
        "summary": {
            "total": total,
            "sent_to_review": len(to_review),
            "auto_decided": len(auto_cases),
            "auto_correct": len(auto_correct),
            "auto_wrong": len(auto_wrong),
            "auto_accuracy": round(len(auto_correct)/len(auto_cases), 4) if auto_cases else 0,
            "errors_caught_by_review": len(errors_sent_to_review),
            "errors_slipped_through": len(errors_still_auto),
        },
        "per_image": results,
    }
    out_path = model_path.parent / "combined_uncertainty.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data",  required=True)
    args = parser.parse_args()
    analyse(Path(args.model), Path(args.data))
