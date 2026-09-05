"""
Fine-tuning a Vision Transformer (ViT) for AI Image Detection
==============================================================

WHAT THIS DOES:
  Takes a ViT pre-trained on ImageNet and fine-tunes it to classify
  images as "real" or "ai_generated" using our curated dataset.

WHY FINE-TUNING AND NOT TRAINING FROM SCRATCH:
  A ViT trained from scratch needs millions of images and days of GPU time.
  We have ~10k images and a laptop. Fine-tuning works because the pre-trained
  model already understands visual concepts (edges, textures, shapes, objects).
  We only teach it the new skill: "does this image look AI-generated?"

  This is called TRANSFER LEARNING — transferring knowledge from one task
  (ImageNet classification) to another (AI detection).

WHY ViT OVER CNN (interview answer):
  CNNs have a LOCAL inductive bias — convolution filters look at small
  neighbourhoods (3×3 or 5×5 pixels). They're great at textures.

  ViT has NO such bias — every image patch attends to every other patch
  via self-attention. This lets it catch GLOBAL inconsistencies:
  - Anatomically impossible hands that look fine locally
  - Lighting that's locally smooth but globally incoherent
  - Backgrounds with no semantic relationship to the subject

  AI generation artifacts are often global, not local — so ViT wins here.

WHAT WE FREEZE AND WHY:
  We freeze all layers EXCEPT the final classification head.

  Why not fine-tune everything?
  - We have only 9,993 images — not enough to retrain 86M parameters
  - Fine-tuning all layers on small data → catastrophic forgetting
    (the model forgets what it learned from ImageNet)
  - Fine-tuning just the head → fast, stable, works well on small data

  This is a key design decision. We unfreeze the last 2 transformer blocks
  as a middle ground — they contain the most task-specific representations.

RUN (from scratch):
  python3 training/finetune_vit.py --data ./curated_data --epochs 5

RUN (resume from existing checkpoint on new data):
  python3 training/finetune_vit.py --data ./curated_data_v2 --epochs 3 \
    --resume ./model_output/best_model.pth

WHY RESUME INSTEAD OF RETRAIN:
  When we add new data (more generators, higher resolution), we don't throw
  away what the model already learned. We load the existing checkpoint and
  continue training on the expanded dataset.

  This is called CONTINUAL LEARNING or INCREMENTAL FINE-TUNING.
  Benefits:
  - Saves 25+ minutes of training time
  - Preserves knowledge about SD v1.4 artifacts while adding new generator knowledge
  - Lower learning rate when resuming avoids overwriting what's already learned

  The risk to watch for: CATASTROPHIC FORGETTING — if the new data is very
  different from the original, the model may "forget" old patterns. Monitoring
  val accuracy on old test set tells you if this is happening.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


# ─── Hyperparameters ──────────────────────────────────────────────────────────
# WHAT ARE HYPERPARAMETERS:
#   Values you set BEFORE training that control how learning happens.
#   They are NOT learned from data — you choose them.
#   Finding good values is called "hyperparameter tuning".

BATCH_SIZE    = 32    # How many images to process at once.
                      # Larger = faster but needs more RAM.
                      # 32 is a safe default for a laptop.

LEARNING_RATE = 2e-4  # How much to adjust weights per step.
                      # Too high → model overshoots and never converges.
                      # Too low  → training takes forever.
                      # 2e-4 is standard for fine-tuning ViT heads.

NUM_EPOCHS    = 5     # How many times to loop through the full training set.
                      # More epochs → more learning, but risk of overfitting.

IMAGE_SIZE    = 224   # ViT expects 224×224 inputs (ImageNet standard).
                      # Our CIFAKE images are 32×32 — we upscale them.

SEED          = 42    # Fixed seed for reproducibility.


# ─── Data transforms ──────────────────────────────────────────────────────────
# WHAT IS DATA AUGMENTATION:
#   We artificially create more training examples by randomly flipping,
#   cropping, and adjusting images. The model sees the "same" image
#   in slightly different forms — this forces it to learn robust features
#   instead of memorizing exact pixel values.
#
#   Example: if a photo of a dog is flipped horizontally, it's still a dog.
#   But the pixel values are completely different. Augmentation teaches
#   the model this invariance.
#
# WHY NORMALIZE WITH IMAGENET STATS:
#   Our pre-trained ViT was trained on ImageNet with mean=[0.485,0.456,0.406]
#   and std=[0.229,0.224,0.225]. We MUST use the same normalization —
#   the model's internal weights expect inputs in this range.
#   Using different stats would be like giving someone instructions in
#   a different unit system — the numbers mean something different.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),          # augmentation
    transforms.RandomRotation(10),              # augmentation: ±10 degrees
    transforms.ColorJitter(brightness=0.2,      # augmentation: slight color shift
                           contrast=0.2),
    transforms.ToTensor(),                      # PIL image → PyTorch tensor [0,1]
    transforms.Normalize(IMAGENET_MEAN,         # normalize to ImageNet range
                         IMAGENET_STD),
])

eval_transform = transforms.Compose([
    # No augmentation on val/test — we want deterministic evaluation
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ─── Model setup ──────────────────────────────────────────────────────────────

def build_model(num_classes: int = 2) -> nn.Module:
    """
    Load pre-trained ViT-B/16 and replace its classification head.

    ViT-B/16 means:
      B = Base size (86M parameters — there's also Small, Large, Huge)
      16 = patch size of 16×16 pixels

    WHAT WE CHANGE:
      The original model has a head that outputs 1,000 scores (ImageNet classes).
      We replace it with a head that outputs 2 scores (real / ai_generated).
      Everything else stays frozen.

    WHAT IS A CLASSIFICATION HEAD:
      The final layer(s) that map the model's internal representation
      to class scores. It's just a linear layer: y = Wx + b
      where W and b are the only weights we train.
    """
    # weights=DEFAULT loads the ImageNet-pretrained weights
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

    # Step 1: Freeze ALL parameters
    # This means their gradients won't be computed → they won't update
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Unfreeze the last 2 transformer encoder blocks
    # WHY: The later blocks capture higher-level semantic features.
    # Fine-tuning them lets the model adapt its representations to our task.
    # Earlier blocks detect low-level features (edges, colors) — those
    # transfer well and don't need updating.
    for block in model.encoder.layers[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    # Step 3: Replace the classification head
    # The original head: Linear(768 → 1000)
    # Our new head:      Linear(768 → 2)
    # 768 = the ViT-B hidden dimension (each patch becomes a 768-dim vector)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    # The new head is trainable by default (requires_grad=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")

    return model


# ─── Training loop ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device) -> dict:
    """
    One pass through the training data.

    WHAT HAPPENS EACH BATCH:
      1. Forward pass: model predicts class scores for the batch
      2. Loss: measure how wrong the predictions are
      3. Backward pass: compute gradients (how to adjust each weight)
      4. Optimizer step: adjust weights in the direction that reduces loss

    WHAT IS LOSS:
      A single number measuring how wrong the model is.
      CrossEntropyLoss is standard for classification:
        - If model is very confident AND correct → loss near 0
        - If model is very confident AND wrong   → loss is high
        - If model is uncertain                  → loss is moderate

    WHAT IS A GRADIENT:
      The partial derivative of the loss with respect to each weight.
      It tells you: "if I increase this weight slightly, does the loss
      go up or down, and by how much?"
      We move weights in the OPPOSITE direction of the gradient
      (gradient DESCENT) to reduce loss.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()           # clear gradients from previous batch
        outputs = model(images)         # forward pass → shape [batch, 2]
        loss = criterion(outputs, labels)
        loss.backward()                 # compute gradients
        optimizer.step()               # update weights

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    """
    Evaluate on val or test set.

    @torch.no_grad() means we skip gradient computation — we're not
    training, just measuring performance. This makes evaluation 2-3x
    faster and uses less memory.

    We collect all predictions and labels to compute additional metrics
    beyond accuracy — we'll use these for the evaluation dashboard.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)                        # raw logits
        probs   = torch.softmax(outputs, dim=1)        # convert to probabilities

        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())  # prob of class 1 = "ai_generated"

    return {
        "loss":     total_loss / total,
        "accuracy": correct / total,
        "preds":    all_preds,
        "labels":   all_labels,
        "probs":    all_probs,   # we use these for AUC-ROC in the eval step
    }


# ─── Main training script ─────────────────────────────────────────────────────

def train(data_dir: Path, output_dir: Path, epochs: int, resume: Optional[Path] = None):
    torch.manual_seed(SEED)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU — training will be slower (~10 min for 5 epochs)")

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_ds   = datasets.ImageFolder(data_dir / "val",   transform=eval_transform)
    test_ds  = datasets.ImageFolder(data_dir / "test",  transform=eval_transform)

    print(f"Class mapping: {train_ds.class_to_idx}")
    print(f"  Train: {len(train_ds)} images")
    print(f"  Val:   {len(val_ds)} images")
    print(f"  Test:  {len(test_ds)} images\n")

    stats_path = data_dir / "stats.json"
    class_weights = None
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        w_ai   = stats["suggested_class_weight_ai"]
        w_real = stats["suggested_class_weight_real"]
        class_weights = torch.tensor([w_ai, w_real], dtype=torch.float).to(device)
        print(f"Class weights: ai={w_ai}, real={w_real}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # Model — either fresh or resumed from checkpoint
    model = build_model(num_classes=2).to(device)
    prior_history = []
    lr = LEARNING_RATE

    if resume is not None:
        # Load existing checkpoint weights
        # WHY LOWER LR ON RESUME:
        #   The model already converged on the old data. A full learning rate
        #   would overwrite those weights too aggressively — the model would
        #   "forget" what it learned from the original generators.
        #   Using 1/5 of the original LR lets it incorporate new patterns
        #   while preserving old ones. This is the standard practice for
        #   incremental fine-tuning.
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        lr = LEARNING_RATE / 5
        prior_history = checkpoint.get("history", [])
        print(f"Resumed from: {resume}")
        print(f"  Previous best val_acc: {checkpoint.get('val_accuracy', '?'):.3f}")
        print(f"  Learning rate reduced to {lr} (1/5 of original) to avoid catastrophic forgetting\n")

    # Loss function with class weights
    # WHY CrossEntropyLoss: standard for multi-class classification.
    # It measures the difference between predicted probability distribution
    # and the true distribution (one-hot label).
    # The weight parameter makes misclassifying the minority class cost more.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer — AdamW
    # WHY AdamW over plain SGD:
    #   Adam adapts the learning rate per parameter — parameters that haven't
    #   changed much get a larger update, active ones get smaller updates.
    #   The 'W' means weight decay is applied correctly (L2 regularization
    #   that prevents weights from growing too large → prevents overfitting).
    # We only pass TRAINABLE parameters — frozen ones don't need an optimizer.
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    # Learning rate scheduler — cosine annealing
    # WHY: Start with full learning rate, gradually reduce it.
    # High LR early → fast initial learning
    # Low LR later  → fine-grained refinement without overshooting
    # Cosine shape is smoother than step decay.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_acc = 0.0
    best_model_path = output_dir / "best_model.pth"

    print(f"\nTraining for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"Epoch {epoch}/{epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  train loss={train_metrics['loss']:.4f}  acc={train_metrics['accuracy']:.3f}")
        print(f"  val   loss={val_metrics['loss']:.4f}  acc={val_metrics['accuracy']:.3f}  "
              f"({elapsed:.0f}s)")

        # Save best model — based on VALIDATION accuracy, not training accuracy
        # WHY VAL NOT TRAIN: Training accuracy always improves (the model
        # is literally optimizing for it). Val accuracy tells you if the model
        # generalizes to unseen data — that's what actually matters.
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "val_accuracy":   best_val_acc,
                "class_to_idx":   train_ds.class_to_idx,
                "history":        prior_history + history,
                "hyperparams": {
                    "lr":         lr,
                    "batch_size": BATCH_SIZE,
                    "image_size": IMAGE_SIZE,
                    "model":      "vit_b_16",
                    "resumed_from": str(resume) if resume else None,
                },
            }, best_model_path)
            print(f"  ✓ New best model saved (val_acc={best_val_acc:.3f})")

        history.append({
            "epoch":      epoch,
            "train_loss": train_metrics["loss"],
            "train_acc":  train_metrics["accuracy"],
            "val_loss":   val_metrics["loss"],
            "val_acc":    val_metrics["accuracy"],
        })

    # Final evaluation on TEST set
    # We do this ONCE at the very end, using the best model.
    # WHY ONLY ONCE: If you look at test results and then keep tweaking
    # your model, you're fitting to the test set — your numbers are
    # no longer a reliable estimate of real-world performance.
    print("\n=== Final Test Evaluation ===")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Best val acc:  {best_val_acc:.4f}")

    # Save all results for the evaluation dashboard (Step 4)
    results = {
        "history":       prior_history + history,
        "test_accuracy": test_metrics["accuracy"],
        "test_probs":    test_metrics["probs"],
        "test_labels":   test_metrics["labels"],
        "test_preds":    test_metrics["preds"],
        "class_to_idx":  train_ds.class_to_idx,
    }
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nModel saved to:  {best_model_path}")
    print(f"Results saved to: {output_dir / 'training_results.json'}")
    print("\nNext step:")
    print("  python3 training/evaluate.py --data ./curated_data --model ./model_output/best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   type=str, default="./curated_data",
                        help="Path to curated dataset directory")
    parser.add_argument("--output", type=str, default="./model_output",
                        help="Where to save model and results")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to existing checkpoint to resume from (incremental training)")
    args = parser.parse_args()

    train(Path(args.data), Path(args.output), args.epochs,
          resume=Path(args.resume) if args.resume else None)
