"""
TB Training Script -- MedGuard AI
===================================
Memory-conscious ResNet-18 training pipeline for binary TB classification
on the TBX11K dataset using CUDA with mixed precision (AMP).

Features:
  - BCEWithLogitsLoss with inverse-frequency pos_weight for class imbalance
  - Mixed Precision Training (torch.cuda.amp) for RTX 3050 4 GB VRAM
  - Cosine Annealing LR scheduler
  - Best-model checkpoint saving to ml/models/best_model_tb.pth
  - Early stopping with configurable patience
  - Full metrics logging: loss, accuracy, AUROC, sensitivity, specificity
  - Smoke test mode (--smoke-test) for quick validation before full training
  - NEVER overwrites the pneumonia model (best_model.pth)

Usage:
  # Smoke test (1 epoch, 10 batches):
  python -m src.train_tb --smoke-test

  # Full training:
  python -m src.train_tb --epochs 20 --batch-size 32 --lr 5e-5

  # Resume from checkpoint:
  python -m src.train_tb --resume --epochs 20
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

# ── Early path bootstrap ──────────────────────────────────────────────────────
# This block runs BEFORE any project imports, ensuring ml/ root is always on
# sys.path regardless of whether the user runs from:
#   d:\anvation\ml\      -> python -m src.train_tb ...
#   d:\anvation\         -> python ml/src/train_tb.py ...
#   d:\anvation\ml\src\  -> python train_tb.py ...
_THIS_FILE = Path(__file__).resolve()
_ML_ROOT   = _THIS_FILE.parent.parent   # ml/src/train_tb.py -> ml/
for _p in (str(_ML_ROOT), str(_ML_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config import MODELS_DIR, REPORTS_DIR, SEED
from src.tb_dataset import (
    TBX11KDataset,
    TB_CLASS_NAMES,
    get_tb_dataloaders,
    verify_dataset_exists,
)
from src.tb_model import (
    TB_MODEL_PATH,
    TBResNet18,
    build_tb_model,
    load_tb_checkpoint,
    save_tb_checkpoint,
)

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Metric helpers ────────────────────────────────────────────────────────────

def binary_metrics(logits: torch.Tensor, targets: torch.Tensor,
                   threshold: float = 0.5) -> Dict[str, float]:
    """Computes accuracy, sensitivity (recall), specificity for binary classification."""
    probs  = torch.sigmoid(logits).cpu().numpy().flatten()
    labels = targets.cpu().numpy().flatten()
    preds  = (probs >= threshold).astype(int)

    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    acc         = (tp + tn) / max(len(labels), 1)
    sensitivity = tp / max(tp + fn, 1)       # recall for TB class
    specificity = tn / max(tn + fp, 1)       # recall for Non-TB class

    # AUROC — only defined when both classes are present in the sample
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(labels.tolist())) < 2:
            auroc = float("nan")       # too few samples / single class (smoke test)
        else:
            auroc = float(roc_auc_score(labels, probs))
    except Exception:
        auroc = float("nan")

    return {
        "acc":         float(acc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auroc":       auroc,
    }


# ── Training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(
    model:     TBResNet18,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, Dict[str, float]]:
    """Runs one training epoch. Returns (avg_loss, metrics_dict)."""
    model.train()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images  = images.to(device, non_blocking=True)
        targets = labels.float().to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type):
            logits = model(images)
            loss   = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_logits.append(logits.detach().float())
        all_targets.append(targets.detach().float())

    n_batches   = batch_idx + 1
    avg_loss    = total_loss / max(n_batches, 1)
    all_logits  = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics     = binary_metrics(all_logits, all_targets)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(
    model:     TBResNet18,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, Dict[str, float]]:
    """Runs evaluation. Returns (avg_loss, metrics_dict)."""
    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images  = images.to(device, non_blocking=True)
        targets = labels.float().to(device, non_blocking=True).unsqueeze(1)

        with autocast(device_type=device.type):
            logits = model(images)
            loss   = criterion(logits, targets)

        total_loss += loss.item()
        all_logits.append(logits.float())
        all_targets.append(targets.float())

    n_batches   = batch_idx + 1
    avg_loss    = total_loss / max(n_batches, 1)
    all_logits  = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics     = binary_metrics(all_logits, all_targets)
    return avg_loss, metrics


# ── Main training loop ─────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    run_start = datetime.now(timezone.utc)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_tb] Device: {device}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[train_tb] GPU: {gpu_name}  ({gpu_mem:.1f} GB VRAM)")

    # ── Dataset ─────────────────────────────────────────────────────────────
    if not verify_dataset_exists():
        print("\n[ERROR] TBX11K dataset not found. Please run:")
        print("  python ml/download_tbx11k.py")
        sys.exit(1)

    print(f"\n[train_tb] Loading TBX11K dataloaders (batch_size={args.batch_size})...")
    train_loader, val_loader, pos_weight = get_tb_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model      = build_tb_model(pretrained=True, dropout=args.dropout, device=device)
    scaler     = GradScaler(device.type)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler  = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    start_epoch  = 0
    best_auroc   = 0.0
    patience_ctr = 0

    if args.resume and TB_MODEL_PATH.exists():
        print(f"[train_tb] Resuming from {TB_MODEL_PATH}")
        model, meta = load_tb_checkpoint(TB_MODEL_PATH, device)
        # Rebuild optimizer/scheduler with same params
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        start_epoch = meta.get("epoch", 0)
        best_auroc  = meta.get("metrics", {}).get("val_auroc", 0.0)
        criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scaler      = GradScaler(device.type)
        scheduler   = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Smoke-test limits ────────────────────────────────────────────────────
    max_train_batches = 10 if args.smoke_test else None
    max_val_batches   = 5  if args.smoke_test else None
    effective_epochs  = 1  if args.smoke_test else args.epochs

    # ── CSV log ──────────────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts     = run_start.strftime("%Y%m%d_%H%M%S")
    prefix = "smoke_tb" if args.smoke_test else "train_tb"
    log_path = REPORTS_DIR / f"{prefix}_metrics_{ts}.csv"
    csv_fields = [
        "epoch", "train_loss", "val_loss",
        "train_acc", "val_acc",
        "val_sensitivity", "val_specificity", "val_auroc",
        "lr", "elapsed_s"
    ]
    csv_file = open(log_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=csv_fields)
    writer.writeheader()

    print(f"\n{'=' * 65}")
    print(f"{'SMOKE TEST' if args.smoke_test else 'FULL TRAINING'} -- TBX11K TB Binary Classification")
    print(f"Epochs: {effective_epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"AMP: enabled | pos_weight: {pos_weight.item():.3f}")
    print(f"Checkpoint: {TB_MODEL_PATH}")
    print(f"Log: {log_path}")
    print(f"{'=' * 65}\n")

    t0 = time.time()
    for epoch in range(start_epoch, start_epoch + effective_epochs):
        epoch_t0 = time.time()

        train_loss, train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            max_batches=max_train_batches,
        )
        val_loss, val_m = evaluate(
            model, val_loader, criterion, device,
            max_batches=max_val_batches,
        )
        scheduler.step()

        elapsed = time.time() - epoch_t0
        row = {
            "epoch":           epoch + 1,
            "train_loss":      f"{train_loss:.4f}",
            "val_loss":        f"{val_loss:.4f}",
            "train_acc":       f"{train_m['acc']:.4f}",
            "val_acc":         f"{val_m['acc']:.4f}",
            "val_sensitivity": f"{val_m['sensitivity']:.4f}",
            "val_specificity": f"{val_m['specificity']:.4f}",
            "val_auroc":       f"{val_m['auroc']:.4f}",
            "lr":              f"{scheduler.get_last_lr()[0]:.2e}",
            "elapsed_s":       f"{elapsed:.1f}",
        }
        writer.writerow(row)
        csv_file.flush()

        print(
            f"Ep {epoch+1:>3}/{start_epoch + effective_epochs} | "
            f"Loss {train_loss:.4f}/{val_loss:.4f} | "
            f"Acc {train_m['acc']:.3f}/{val_m['acc']:.3f} | "
            f"AUROC {val_m['auroc']:.3f} | "
            f"Sens {val_m['sensitivity']:.3f} | "
            f"Spec {val_m['specificity']:.3f} | "
            f"LR {scheduler.get_last_lr()[0]:.1e} | "
            f"{elapsed:.1f}s"
        )

        # ── Checkpoint best model ────────────────────────────────────────────
        is_best = val_m["auroc"] > best_auroc
        if is_best:
            best_auroc   = val_m["auroc"]
            patience_ctr = 0
            if not args.smoke_test:
                save_tb_checkpoint(
                    model, optimizer, epoch + 1,
                    {
                        "val_loss":        val_loss,
                        "val_acc":         val_m["acc"],
                        "val_auroc":       val_m["auroc"],
                        "val_sensitivity": val_m["sensitivity"],
                        "val_specificity": val_m["specificity"],
                    }
                )
        else:
            patience_ctr += 1
            if not args.smoke_test and patience_ctr >= args.patience:
                print(f"\n[train_tb] Early stopping triggered (patience={args.patience}).")
                break

    csv_file.close()
    total_time = time.time() - t0

    print(f"\n{'=' * 65}")
    print(f"Training complete in {total_time:.1f}s")
    print(f"Best val AUROC: {best_auroc:.4f}")
    if not args.smoke_test:
        print(f"Best model saved: {TB_MODEL_PATH}")
    print(f"Metrics log:     {log_path}")
    print(f"{'=' * 65}")

    if args.smoke_test:
        # Estimate full training time
        epoch_time = total_time
        full_batches_per_epoch = len(train_loader)
        smoke_batches = max_train_batches
        scale = full_batches_per_epoch / smoke_batches
        estimated_epoch_s  = epoch_time * scale
        estimated_total_m  = estimated_epoch_s * args.epochs / 60
        print(f"\n[SMOKE TEST RESULT]")
        print(f"  Smoke epoch time ({smoke_batches} batches): {epoch_time:.1f}s")
        print(f"  Estimated full epoch time (~{full_batches_per_epoch} batches): {estimated_epoch_s:.0f}s")
        print(f"  Estimated total training ({args.epochs} epochs): {estimated_total_m:.0f} min")
        print(f"\nSmoke test passed. Confirm before starting full training.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train TB ResNet-18 classifier on TBX11K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs",       type=int,   default=20,     help="Number of epochs")
    p.add_argument("--batch-size",   type=int,   default=32,     help="Batch size")
    p.add_argument("--lr",           type=float, default=5e-5,   help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4,   help="AdamW weight decay")
    p.add_argument("--dropout",      type=float, default=0.3,    help="Dropout probability in head")
    p.add_argument("--patience",     type=int,   default=5,      help="Early stopping patience")
    p.add_argument("--num-workers",  type=int,   default=0,      help="DataLoader workers")
    p.add_argument("--smoke-test",   action="store_true",        help="Quick 1-epoch, 10-batch smoke test")
    p.add_argument("--resume",       action="store_true",        help="Resume from best_model_tb.pth")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
