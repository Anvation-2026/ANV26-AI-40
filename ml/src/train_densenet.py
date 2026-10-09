"""
ml/src/train_densenet.py -- Memory-Conscious DenseNet-201 Training Pipeline for ChestMNIST 224x224.
=================================================================================================
Optimized for 4 GB VRAM (NVIDIA RTX 3050 Laptop GPU):
  - Backbone: Pretrained DenseNet-201 (ImageNet weights)
  - MLP Head: 1920 -> 512 -> 256 -> 14 with ReLU and Dropout(0.3)
  - Stage 1: Freeze backbone, train MLP head (Warmup)
  - Stage 2: Unfreeze denseblock4 & norm5 for fine-tuning
  - Mixed precision training: torch.amp.autocast('cuda') + torch.amp.GradScaler('cuda')
  - Gradient accumulation: Micro-batch 4 * Accum 8 = Effective Batch Size 32
  - Multi-label loss: BCEWithLogitsLoss with inverse-prevalence pos_weights
  - Metrics: Macro and per-class AUROC, AP, sensitivity, specificity, F1
  - Checkpoints: ml/models/densenet201/best_model.pth (never touches ResNet-18)
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

# Path bootstrap
_THIS_FILE = Path(__file__).resolve()
_ML_ROOT = _THIS_FILE.parent.parent
for _p in (str(_ML_ROOT), str(_ML_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import DATA_DIR, MODELS_DIR, REPORTS_DIR, SEED
from src.chestmnist_dataset import (
    CHESTMNIST_14_LABELS,
    NUM_CHEST_CLASSES,
    ChestMNIST224Dataset,
    compute_multilabel_pos_weights,
    get_chestmnist_dataloaders,
    get_chestmnist_transforms,
    load_chestmnist_splits,
)
from src.densenet_model import (
    DENSENET_CHECKPOINT_PATH,
    DENSENET_MODELS_DIR,
    DenseNet201Chest14,
    build_densenet201,
    load_densenet_checkpoint,
    save_densenet_checkpoint,
)
from src.evaluate_densenet import (
    evaluate_predictions,
    save_thresholds_file,
)

# 4 GB VRAM Defaults
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 8       # 4 * 8 = 32 effective batch size
DEFAULT_HEAD_EPOCHS = 5
DEFAULT_FINETUNE_EPOCHS = 5
DEFAULT_LR_HEAD = 1e-3
DEFAULT_LR_FINETUNE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_gpu_smoke_profile(
    model: DenseNet201Chest14,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_batches: int = 5,
) -> Dict[str, Any]:
    """
    Runs forward, backward, and optimizer steps on dummy/representative batches
    to verify CUDA execution, peak VRAM, and latency.
    """
    model.train()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scaler = GradScaler(device.type)
    criterion = nn.BCEWithLogitsLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    dummy_target = (torch.rand(batch_size, NUM_CHEST_CLASSES, device=device) > 0.8).float()

    for _ in range(num_batches):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(dummy_input)
            loss = criterion(outputs, dummy_target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_vram_mb = 0.0

    elapsed = time.perf_counter() - start_time
    sec_per_batch = elapsed / num_batches
    sec_per_sample = sec_per_batch / batch_size

    # Total official ChestMNIST training images: 78,468
    total_samples = 78468
    epoch_sec = (total_samples / batch_size) * sec_per_batch
    epoch_min = epoch_sec / 60.0

    return {
        "sec_per_batch": round(sec_per_batch, 4),
        "ms_per_sample": round(sec_per_sample * 1000, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "est_epoch_minutes": round(epoch_min, 1),
        "est_10epochs_hours": round((epoch_min * 10) / 60.0, 2),
    }


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluates validation loss and multi-label metrics."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(inputs)
                loss = criterion(logits, targets)

            total_loss += loss.item() * len(targets)
            total_samples += len(targets)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())

            if max_batches is not None and (step + 1) >= max_batches:
                break

    avg_loss = total_loss / max(1, total_samples)
    probs_cat = np.concatenate(all_probs, axis=0)
    targets_cat = np.concatenate(all_targets, axis=0)

    eval_results = evaluate_predictions(probs_cat, targets_cat, tune_thresholds=True)
    return avg_loss, eval_results


def train_densenet_pipeline(
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum_steps: int = DEFAULT_GRAD_ACCUM,
    head_epochs: int = DEFAULT_HEAD_EPOCHS,
    finetune_epochs: int = DEFAULT_FINETUNE_EPOCHS,
    lr_head: float = DEFAULT_LR_HEAD,
    lr_finetune: float = DEFAULT_LR_FINETUNE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    smoke_test: bool = False,
    profile_only: bool = False,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    Executes memory-conscious two-stage DenseNet-201 training pipeline.
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_start = datetime.now(timezone.utc)

    print("\n" + "=" * 70)
    print("      MEDGUARD AI -- DENSENET-201 MULTI-LABEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) | CUDA: {torch.version.cuda}")

    # Build model
    model = build_densenet201(pretrained=True, dropout=0.3, activation="relu", device=device)

    # 1. Smoke test / Profiling
    if profile_only or smoke_test:
        print("\n[DenseNet] Running CUDA forward/backward smoke profile...")
        profile = run_gpu_smoke_profile(model, device, batch_size=batch_size, num_batches=5)
        print(f"  Physical batch size: {batch_size} (Grad Accum: {grad_accum_steps} -> Effective: {batch_size * grad_accum_steps})")
        print(f"  Latency: {profile['ms_per_sample']} ms/sample ({profile['sec_per_batch']} s/batch)")
        print(f"  Peak VRAM: {profile['peak_vram_mb']} MB / 4096 MB")
        print(f"  Est. full epoch (78,468 images): ~{profile['est_epoch_minutes']} min")
        print(f"  Est. 10 epochs: ~{profile['est_10epochs_hours']} hours")

        if profile_only:
            print("[DenseNet] Profiling complete.")
            return {"profile": profile}

    # 2. Check dataset availability
    npz_path = DATA_DIR / "chestmnist_224.npz"
    if not npz_path.exists():
        print("\n[ERROR] chestmnist_224.npz not found in ml/data.")
        print("Please run: python ml/download_chestmnist.py")
        sys.exit(1)

    print(f"\n[DenseNet] Loading official ChestMNIST splits from {npz_path}...")
    try:
        train_loader, val_loader, test_loader, pos_weights = get_chestmnist_dataloaders(
            data_dir=DATA_DIR,
            batch_size=batch_size,
            num_workers=0,  # 0 for Windows stability
            device=device,
        )
    except Exception as exc:
        print(f"\n[ERROR] Failed to load dataset: {exc}")
        print("Dataset file may still be downloading. Run 'python ml/download_chestmnist.py' to complete.")
        sys.exit(1)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    scaler = GradScaler(device.type)

    DENSENET_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = run_start.strftime("%Y%m%d_%H%M%S")
    log_csv = REPORTS_DIR / f"train_densenet_metrics_{ts}.csv"

    csv_file = open(log_csv, "w", newline="")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=["stage", "epoch", "train_loss", "val_loss", "macro_auroc", "macro_ap", "macro_f1", "lr", "elapsed_s"],
    )
    writer.writeheader()

    best_macro_auroc = 0.0
    best_thresholds: Dict[str, float] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1: Warmup MLP Classifier Head (Backbone Frozen)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print(f"STAGE 1: Training MLP Head ({head_epochs} epochs, Backbone Frozen)")
    print("-" * 70)
    model.freeze_backbone(True)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=head_epochs, eta_min=lr_head * 0.01)

    for epoch in range(1, head_epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_accum = 0.0
        train_samples = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss_accum += loss.item() * len(targets)
            train_samples += len(targets)

            if smoke_test and step >= 10:
                break

        scheduler.step()
        train_loss = train_loss_accum / max(1, train_samples)
        val_loss, val_eval = validate_epoch(
            model, val_loader, criterion, device, max_batches=(5 if smoke_test else None)
        )
        macro = val_eval["macro"]
        elapsed = time.time() - t0

        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Stage 1 | Ep {epoch:02d}/{head_epochs:02d}] "
            f"TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
            f"Val AUROC: {macro['macro_auroc']:.4f} | Val AP: {macro['macro_ap']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        writer.writerow({
            "stage": "head",
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "macro_auroc": macro["macro_auroc"],
            "macro_ap": macro["macro_ap"],
            "macro_f1": macro["macro_f1"],
            "lr": round(cur_lr, 7),
            "elapsed_s": round(elapsed, 1),
        })
        csv_file.flush()

        if macro["macro_auroc"] > best_macro_auroc:
            best_macro_auroc = macro["macro_auroc"]
            best_thresholds = val_eval["optimal_thresholds"]
            save_densenet_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                stage="head",
                metrics=val_eval,
                thresholds=best_thresholds,
                save_path=DENSENET_CHECKPOINT_PATH,
            )
            save_thresholds_file(best_thresholds, val_eval)

        if smoke_test:
            break

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2: Fine-Tuning denseblock4 & norm5
    # ─────────────────────────────────────────────────────────────────────────
    if finetune_epochs > 0 and not smoke_test:
        print("\n" + "-" * 70)
        print(f"STAGE 2: Fine-Tuning denseblock4 & norm5 ({finetune_epochs} epochs)")
        print("-" * 70)
        model.unfreeze_denseblock4()
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_finetune, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=lr_finetune * 0.01)

        for epoch in range(1, finetune_epochs + 1):
            t0 = time.time()
            model.train()
            train_loss_accum = 0.0
            train_samples = 0
            optimizer.zero_grad(set_to_none=True)

            for step, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    logits = model(inputs)
                    loss = criterion(logits, targets)
                    loss_scaled = loss / grad_accum_steps

                scaler.scale(loss_scaled).backward()

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                train_loss_accum += loss.item() * len(targets)
                train_samples += len(targets)

            scheduler.step()
            train_loss = train_loss_accum / max(1, train_samples)
            val_loss, val_eval = validate_epoch(
                model, val_loader, criterion, device, max_batches=(5 if smoke_test else None)
            )
            macro = val_eval["macro"]
            elapsed = time.time() - t0

            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[Stage 2 | Ep {epoch:02d}/{finetune_epochs:02d}] "
                f"TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
                f"Val AUROC: {macro['macro_auroc']:.4f} | Val AP: {macro['macro_ap']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            writer.writerow({
                "stage": "finetune",
                "epoch": head_epochs + epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "macro_auroc": macro["macro_auroc"],
                "macro_ap": macro["macro_ap"],
                "macro_f1": macro["macro_f1"],
                "lr": round(cur_lr, 7),
                "elapsed_s": round(elapsed, 1),
            })
            csv_file.flush()

            if macro["macro_auroc"] > best_macro_auroc:
                best_macro_auroc = macro["macro_auroc"]
                best_thresholds = val_eval["optimal_thresholds"]
                save_densenet_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=head_epochs + epoch,
                    stage="finetune",
                    metrics=val_eval,
                    thresholds=best_thresholds,
                    save_path=DENSENET_CHECKPOINT_PATH,
                )
                save_thresholds_file(best_thresholds, val_eval)

    csv_file.close()
    print(f"\n[DenseNet] Training completed. Best Val Macro AUROC: {best_macro_auroc:.4f}")
    print(f"[DenseNet] Checkpoint saved: {DENSENET_CHECKPOINT_PATH}")

    # 3. Final untouched test set evaluation (Strictly after model selection)
    if not smoke_test and DENSENET_CHECKPOINT_PATH.exists():
        print("\n" + "=" * 70)
        print("FINAL EVALUATION ON UNTOUCHED TEST SPLIT (22,433 images)")
        print("=" * 70)
        best_model, _ = load_densenet_checkpoint(DENSENET_CHECKPOINT_PATH, device=device)
        test_loss, test_eval = validate_epoch(best_model, test_loader, criterion, device)
        test_report_path = REPORTS_DIR / "densenet_test_evaluation.json"
        with open(test_report_path, "w") as f:
            json.dump({"test_loss": test_loss, **test_eval}, f, indent=2)
        print(f"[DenseNet] Test evaluation saved: {test_report_path}")
        print(f"  Test Macro AUROC: {test_eval['macro']['macro_auroc']:.4f}")
        print(f"  Test Macro AP:    {test_eval['macro']['macro_average_precision']:.4f}")

    return {
        "best_macro_auroc": best_macro_auroc,
        "checkpoint": str(DENSENET_CHECKPOINT_PATH),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DenseNet-201 ChestMNIST Training Pipeline")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--head-epochs", type=int, default=DEFAULT_HEAD_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=DEFAULT_FINETUNE_EPOCHS)
    parser.add_argument("--lr-head", type=float, default=DEFAULT_LR_HEAD)
    parser.add_argument("--lr-finetune", type=float, default=DEFAULT_LR_FINETUNE)
    parser.add_argument("--smoke", action="store_true", help="Run quick 1-epoch smoke test")
    parser.add_argument("--profile-only", action="store_true", help="Profile forward/backward steps without training")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    args = parser.parse_args()

    train_densenet_pipeline(
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        lr_head=args.lr_head,
        lr_finetune=args.lr_finetune,
        smoke_test=args.smoke,
        profile_only=args.profile_only,
        resume=args.resume,
    )
