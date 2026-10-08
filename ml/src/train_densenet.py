"""
ml/src/train_densenet.py — Memory-Conscious DenseNet-201 Training Pipeline for ChestMNIST 224x224.

Multi-label 14-disease classification on frontal chest radiographs.
Isolates DenseNet-201 artifacts from existing binary ResNet-18 models.
Configured for NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) with mixed precision.
"""
import argparse
import csv
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

# Path bootstrap
_ML_ROOT = Path(__file__).resolve().parent.parent
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from config import (
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
)
from src.chestmnist_dataset import (
    CHESTMNIST_14_LABELS,
    NUM_CHEST_CLASSES,
    ChestMNIST224Dataset,
    compute_multilabel_pos_weights,
    get_chestmnist_transforms,
    load_or_download_chestmnist,
)
from src.densenet_model import DenseNet201Chest14, build_densenet201

# Memory-conscious defaults for 4 GB VRAM
DEFAULT_DENSENET_BATCH_SIZE = 16
DEFAULT_DENSENET_EPOCHS = 10
DEFAULT_DENSENET_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRAD_ACCUM = 2  # Effective batch size = 16 * 2 = 32


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_multilabel(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates multi-label model across all 14 disease classes.
    Computes BCE loss, Macro Mean AUROC, and Mean Average Precision (mAP).
    """
    model.eval()
    total_loss = 0.0
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(targets)
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())

    total_samples = len(loader.dataset)
    avg_loss = total_loss / max(1, total_samples)

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0).numpy()
    probs_cat = torch.sigmoid(logits_cat).numpy()

    # Per-class AUROC and AP
    auroc_list = []
    ap_list = []
    per_class_aurocs = {}

    for c in range(NUM_CHEST_CLASSES):
        y_true = targets_cat[:, c]
        y_pred = probs_cat[:, c]
        class_name = CHESTMNIST_14_LABELS[c]

        # Only compute if at least 1 positive and 1 negative present
        if len(np.unique(y_true)) > 1:
            try:
                c_auroc = float(roc_auc_score(y_true, y_pred))
                auroc_list.append(c_auroc)
                per_class_aurocs[class_name] = round(c_auroc, 4)
            except Exception:
                pass

            try:
                c_ap = float(average_precision_score(y_true, y_pred))
                ap_list.append(c_ap)
            except Exception:
                pass

    mean_auroc = float(np.mean(auroc_list)) if auroc_list else 0.5
    mean_ap = float(np.mean(ap_list)) if ap_list else 0.0

    return {
        "loss": avg_loss,
        "mean_auroc": mean_auroc,
        "mean_ap": mean_ap,
        "per_class_aurocs": per_class_aurocs,
    }


def run_smoke_and_profile(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    num_batches: int = 5,
) -> Dict[str, float]:
    """
    Runs forward, backward, and optimizer steps on a few batches to measure
    per-batch latency, peak VRAM usage, and projected full training time.
    """
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    batch_count = 0

    for inputs, targets in train_loader:
        if batch_count >= num_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_count += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    elapsed = time.perf_counter() - start_time
    sec_per_batch = elapsed / max(1, batch_count)
    sec_per_sample = sec_per_batch / train_loader.batch_size

    # Projected time for 78,468 samples
    total_train_samples = 78468
    total_batches = total_train_samples / train_loader.batch_size
    epoch_sec = total_batches * sec_per_batch
    epoch_min = epoch_sec / 60.0

    return {
        "sec_per_batch": sec_per_batch,
        "sec_per_sample": sec_per_sample,
        "peak_vram_mb": peak_vram_mb,
        "projected_epoch_minutes": epoch_min,
        "projected_10epochs_hours": (epoch_min * 10) / 60.0,
    }


def train_densenet(
    epochs: int = DEFAULT_DENSENET_EPOCHS,
    batch_size: int = DEFAULT_DENSENET_BATCH_SIZE,
    lr: float = DEFAULT_DENSENET_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    grad_accum_steps: int = DEFAULT_GRAD_ACCUM,
    seed: int = SEED,
    smoke: bool = False,
    profile_only: bool = False,
    freeze_early: bool = True,
    checkpoint_out: Path = MODELS_DIR / "densenet201_chestmnist.pth",
) -> Dict:
    """
    Memory-conscious training loop for DenseNet-201 on ChestMNIST 224x224.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("      MEDGUARD AI — DENSENET-201 CHEST-14 MULTI-LABEL PIPELINE")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB VRAM)")
        print(f"CUDA: {torch.version.cuda}")

    # 1. Dataset splits
    data_splits = load_or_download_chestmnist(
        DATA_DIR,
        download_if_missing=(not (smoke or profile_only)),
        smoke_fallback=(smoke or profile_only),
    )
    train_imgs, train_lbls = data_splits["train"]
    val_imgs, val_lbls = data_splits["val"]
    test_imgs, test_lbls = data_splits["test"]

    train_trans, eval_trans = get_chestmnist_transforms()

    train_ds = ChestMNIST224Dataset(train_imgs, train_lbls, transform=train_trans)
    val_ds = ChestMNIST224Dataset(val_imgs, val_lbls, transform=eval_trans)
    test_ds = ChestMNIST224Dataset(test_imgs, test_lbls, transform=eval_trans)

    # 2. Subset selection for smoke test
    if smoke or profile_only:
        print("[DenseNet] Smoke/Profile Mode: using 128 train and 64 val samples.")
        train_ds = Subset(train_ds, indices=list(range(min(128, len(train_ds)))))
        val_ds = Subset(val_ds, indices=list(range(min(64, len(val_ds)))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    # 3. Model construction
    print(f"[DenseNet] Instantiating DenseNet-201 (pretrained ImageNet-1K weights)...")
    model = build_densenet201(pretrained=True, drop_rate=0.2).to(device)

    if freeze_early:
        print("[DenseNet] Memory-conscious transfer learning: Freezing early dense blocks (1-3).")
        model.freeze_early_layers(freeze=True)

    # 4. Loss & Optimizer
    # Positive weights for multi-label class imbalance
    pos_weights = compute_multilabel_pos_weights(train_lbls, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 5. Profile / Smoke step test
    print("\n[DenseNet] Running GPU forward/backward smoke & latency profiling...")
    profile_results = run_smoke_and_profile(model, train_loader, criterion, device, scaler, optimizer, num_batches=4)
    print(f"  Batch size: {batch_size} (effective: {batch_size * grad_accum_steps})")
    print(f"  Latency: {profile_results['sec_per_batch']:.3f} s/batch ({profile_results['sec_per_sample'] * 1000:.1f} ms/image)")
    print(f"  Peak VRAM: {profile_results['peak_vram_mb']:.1f} MB / 4096 MB")
    print(f"  Projected time per full epoch (78,468 images): ~{profile_results['projected_epoch_minutes']:.1f} minutes")
    print(f"  Projected 10 epochs: ~{profile_results['projected_10epochs_hours']:.1f} hours")

    if profile_only:
        print("[DenseNet] Profile complete. Full training deferred until user review.")
        return profile_results

    if smoke:
        epochs = 1
        print(f"[DenseNet] Smoke test running 1 validation epoch on subset...")

    best_val_auroc = 0.0
    best_epoch = -1
    logs = []

    print(f"\n{'Epoch':<6} {'TrainLoss':<11} {'ValLoss':<9} {'MeanAUROC':<11} {'mAP':<8} {'LR':<9}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_count = 0

        optimizer.zero_grad(set_to_none=True)

        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss_accum += loss.item() * len(targets)
            train_count += len(targets)

        scheduler.step()
        avg_train_loss = train_loss_accum / max(1, train_count)

        # Validation
        val_res = evaluate_multilabel(model, val_loader, criterion, device)
        cur_lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:<6d} {avg_train_loss:<11.4f} {val_res['loss']:<9.4f} "
            f"{val_res['mean_auroc']:<11.4f} {val_res['mean_ap']:<8.4f} {cur_lr:<9.2e}"
        )

        logs.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_res["loss"], 4),
            "val_mean_auroc": round(val_res["mean_auroc"], 4),
            "val_mAP": round(val_res["mean_ap"], 4),
            "lr": round(cur_lr, 7),
        })

        if val_res["mean_auroc"] > best_val_auroc:
            best_val_auroc = val_res["mean_auroc"]
            best_epoch = epoch
            # Save checkpoint
            checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_auroc": best_val_auroc,
                "val_mAP": val_res["mean_ap"],
                "per_class_aurocs": val_res["per_class_aurocs"],
                "class_names": CHESTMNIST_14_LABELS,
                "arch": "densenet201",
                "dataset": "ChestMNIST 224x224 (NIH-ChestXray14)",
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }, checkpoint_out)

    print(f"\n[DenseNet] Completed. Best Epoch: {best_epoch} (Val Mean AUROC: {best_val_auroc:.4f})")
    print(f"[DenseNet] Checkpoint saved: {checkpoint_out}")
    return {
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "profile": profile_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DenseNet-201 ChestMNIST Training")
    parser.add_argument("--epochs", type=int, default=DEFAULT_DENSENET_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_DENSENET_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_DENSENET_LR)
    parser.add_argument("--smoke", action="store_true", help="Run quick 1-epoch smoke test on subset")
    parser.add_argument("--profile-only", action="store_true", help="Profile forward/backward and estimate training time without full loop")
    args = parser.parse_args()

    train_densenet(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        smoke=args.smoke,
        profile_only=args.profile_only,
    )
