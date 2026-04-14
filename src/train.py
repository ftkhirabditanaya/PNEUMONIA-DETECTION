"""
train.py  –  Complete two-phase training pipeline (CPU-optimised).

WHAT CHANGED FROM YOUR ORIGINAL:
  1. Proper validation split (800+ samples instead of 16)
  2. Two-phase training: Phase 1 (classifier only) → Phase 2 (fine-tune)
  3. Correct learning rates per phase (1e-3 → 5e-5)
  4. Full metrics: Accuracy, F1, AUC, Precision, Recall (not just accuracy)
  5. batch_size=16 and num_workers=0 for CPU stability
  6. More epochs (25 Phase1 + 20 Phase2) with patience=7
  7. Correct early stopping on F1 (not accuracy) — better for imbalanced data
  8. Training curve saved as a plot

RUN:
    python src/train.py

EXPECTED RESULTS (CPU, no GPU):
    Phase 1 best val F1   : ~0.87–0.90 (around epoch 15–20)
    Phase 2 best val F1   : ~0.91–0.94 (after fine-tuning)
    Time (CPU):             ~4–8 hours for full run
    Time (CPU, quick test): ~20–40 min for Phase 1 only
"""

import os
import sys
import time
import json
from pathlib import Path

import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for VS Code
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, accuracy_score, confusion_matrix,
)

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import get_dataloaders
from src.models.densenet_attention import DenseNetAttention
from src.losses import FocalLoss


# ── Configuration ─────────────────────────────────────────────────────────────
DATA_ROOT    = "data/raw/chest_xray"   # adjust if your path differs
SAVE_DIR     = Path("models")
PLOT_DIR     = Path("outputs/plots")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CPU-safe batch size (16 = ~1.5 GB RAM; increase to 32 if you have 8+ GB)
BATCH_SIZE   = 16
NUM_WORKERS  = 0       # MUST be 0 on Windows

# Phase 1: Freeze backbone, train classifier only
PHASE1_EPOCHS   = 25
PHASE1_LR       = 1e-3    # high LR is fine because only new layers train
PHASE1_PATIENCE = 7

# Phase 2: Unfreeze denseblock4, fine-tune with tiny LR
PHASE2_EPOCHS   = 20
PHASE2_LR       = 5e-5    # very small — don't destroy pretrained features
PHASE2_PATIENCE = 7

SEED = 42


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(all_labels, all_preds, all_probs):
    """
    Compute full research-grade metric suite.
    Returns a dict with all values, ready for logging and paper tables.
    """
    metrics = {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["auc"] = 0.0   # happens if only one class in batch

    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = tn / max(tn + fp, 1)
        metrics["sensitivity"] = tp / max(tp + fn, 1)  # = recall for PNEUMONIA
    return metrics


# ── Single epoch train ────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, epoch, log_interval=20):
    model.train()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping prevents exploding gradients on deep networks
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item()

        # Collect predictions for train metrics
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if batch_idx % log_interval == 0:
            print(f"  [Batch {batch_idx:3d}/{len(loader)}] loss={loss.item():.4f}")

    avg_loss = running_loss / len(loader)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = avg_loss
    return metrics


# ── Validation / Test ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = avg_loss
    return metrics, all_labels, all_preds, all_probs


# ── Training phase ────────────────────────────────────────────────────────────
def run_phase(
    phase_name, model, train_loader, val_loader, criterion,
    optimizer, scheduler, num_epochs, patience, save_path,
):
    """Generic training loop for one phase."""
    best_f1 = 0.0
    counter = 0
    history = {"train": [], "val": []}

    print(f"\n{'='*60}")
    print(f"  {phase_name}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion)

        elapsed = time.time() - t0

        # Pretty log
        print(f"\n  TRAIN | loss={train_metrics['loss']:.4f} "
              f"acc={train_metrics['accuracy']:.4f} "
              f"f1={train_metrics['f1']:.4f} "
              f"auc={train_metrics['auc']:.4f}")
        print(f"  VAL   | loss={val_metrics['loss']:.4f}  "
              f"acc={val_metrics['accuracy']:.4f} "
              f"f1={val_metrics['f1']:.4f} "
              f"auc={val_metrics['auc']:.4f} "
              f"prec={val_metrics['precision']:.4f} "
              f"rec={val_metrics['recall']:.4f}")
        print(f"  Time  : {elapsed:.1f}s")

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        scheduler.step(val_metrics["f1"])

        # Save best model checkpoint
        val_f1 = val_metrics["f1"]
        if val_f1 > best_f1:
            best_f1 = val_f1
            counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "val_metrics": val_metrics,
                },
                save_path,
            )
            print(f"  ✅ Best model saved (F1={best_f1:.4f})")
        else:
            counter += 1
            print(f"  ⚠  No improvement: {counter}/{patience}")
            if counter >= patience:
                print(f"  🛑 Early stopping at epoch {epoch}")
                break

    print(f"\n  {phase_name} complete. Best val F1: {best_f1:.4f}")
    return history, best_f1


# ── Plot training curves ──────────────────────────────────────────────────────
def plot_history(history_p1, history_p2, save_path):
    """Save training/validation curves for loss, accuracy, F1, AUC."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Combine phases with a visual separator
    def concat(phase, metric):
        p1 = [e[metric] for e in history_p1[phase]]
        p2 = [e[metric] for e in history_p2[phase]]
        return p1 + p2

    epochs_p1 = len(history_p1["train"])
    total = epochs_p1 + len(history_p2["train"])
    x = list(range(1, total + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training History: DenseNet121 + CBAM", fontsize=14, fontweight="bold")

    metrics = [
        ("loss",     "Loss",     "upper right"),
        ("accuracy", "Accuracy", "lower right"),
        ("f1",       "F1-Score", "lower right"),
        ("auc",      "ROC-AUC",  "lower right"),
    ]

    for ax, (key, title, loc) in zip(axes.flat, metrics):
        tr = concat("train", key)
        vl = concat("val",   key)
        ax.plot(x, tr, label="Train", color="#1f77b4", linewidth=2)
        ax.plot(x, vl, label="Val",   color="#ff7f0e", linewidth=2)
        ax.axvline(x=epochs_p1, color="gray", linestyle="--",
                   alpha=0.6, label="Phase 1→2")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(loc=loc)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Plot] Training curves saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Pneumonia Detection — DenseNet121 + CBAM + Focal Loss")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data (this may take 1-2 minutes on first run)...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_split=0.20,
        seed=SEED,
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    # ── Loss function ─────────────────────────────────────────────────────────
    # alpha=0.75: 75% weight on NORMAL (minority class) to reduce false negatives
    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: Train classifier head only (backbone frozen)
    # LR = 1e-3 — high because only the new random layers are updating
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Setup] Building model — Phase 1 (backbone frozen)...")
    model = DenseNetAttention(
        num_classes=2,
        dropout_rate=0.4,
        freeze_backbone=True,   # backbone frozen
    ).to(DEVICE)

    # Only pass parameters that require grad (classifier + CBAM)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Phase 1 trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer_p1 = optim.Adam(trainable_params, lr=PHASE1_LR, weight_decay=1e-4)
    scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p1, mode="max", factor=0.5, patience=3
    )

    history_p1, best_f1_p1 = run_phase(
        phase_name="PHASE 1 — Classifier Training (backbone frozen)",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_p1,
        scheduler=scheduler_p1,
        num_epochs=PHASE1_EPOCHS,
        patience=PHASE1_PATIENCE,
        save_path=SAVE_DIR / "phase1_best.pth",
    )

    # Load best Phase 1 weights before entering Phase 2
    ckpt = torch.load(SAVE_DIR / "phase1_best.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\n[Setup] Phase 1 best weights loaded (F1={best_f1_p1:.4f})")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: Unfreeze denseblock4, fine-tune with very low LR
    # LR = 5e-5 — tiny so we don't destroy the pretrained features
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Setup] Phase 2 — Unfreezing denseblock4 for fine-tuning...")
    model.unfreeze_last_block()

    optimizer_p2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE2_LR,
        weight_decay=1e-4,
    )
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=PHASE2_EPOCHS, eta_min=1e-7
    )

    history_p2, best_f1_p2 = run_phase(
        phase_name="PHASE 2 — Fine-tuning (denseblock4 + CBAM + classifier)",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_p2,
        scheduler=scheduler_p2,
        num_epochs=PHASE2_EPOCHS,
        patience=PHASE2_PATIENCE,
        save_path=SAVE_DIR / "phase2_best.pth",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL EVALUATION on held-out test set
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL TEST SET EVALUATION")
    print("="*60)

    # Load best overall model (Phase 2 usually wins)
    best_path = SAVE_DIR / "phase2_best.pth"
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best model from {best_path}")

    test_metrics, test_labels, test_preds, test_probs = evaluate(
        model, test_loader, criterion
    )

    print(f"\n  Accuracy   : {test_metrics['accuracy']:.4f}")
    print(f"  F1-Score   : {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC    : {test_metrics['auc']:.4f}")
    print(f"  Precision  : {test_metrics['precision']:.4f}")
    print(f"  Recall     : {test_metrics['recall']:.4f}")
    print(f"  Specificity: {test_metrics.get('specificity', 0):.4f}")
    print(f"  Sensitivity: {test_metrics.get('sensitivity', 0):.4f}")

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print(f"\n  Confusion Matrix:")
    print(f"                  Pred NORMAL  Pred PNEUMONIA")
    print(f"  True NORMAL  :     {cm[0,0]:4d}           {cm[0,1]:4d}")
    print(f"  True PNEUMONIA:    {cm[1,0]:4d}           {cm[1,1]:4d}")

    # Check against targets
    acc_target = test_metrics["accuracy"] >= 0.90
    auc_target = test_metrics["auc"] >= 0.95
    print(f"\n  Target accuracy >90%: {'✅ MET' if acc_target else '❌ Not yet'}")
    print(f"  Target AUC     >0.95: {'✅ MET' if auc_target else '❌ Not yet'}")

    # Save results JSON
    results = {
        "phase1_best_f1": best_f1_p1,
        "phase2_best_f1": best_f1_p2,
        "test_metrics": test_metrics,
    }
    results_path = PLOT_DIR / "densenet_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {results_path}")

    # ── Training curves plot ──────────────────────────────────────────────────
    plot_history(
        history_p1, history_p2,
        save_path=PLOT_DIR / "training_history.png",
    )

    print("\n✅ Training complete! Next step: run src/explainability/gradcam.py")


if __name__ == "__main__":
    main()

