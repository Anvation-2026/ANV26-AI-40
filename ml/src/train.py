import argparse
import csv
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

# Path bootstrap (adds ml/ root to sys.path)
try:
    from src import _ML_ROOT  # noqa: F401  # python -m src.x from ml/
except ModuleNotFoundError:
    import _pathfix  # noqa: F401  # python x.py from ml/src/

from config import (
    CLASS_NAMES,
    DATASET_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY,
    FALLBACK_BATCH_SIZE,
    MODELS_DIR,
    NUM_WORKERS,
    REPORTS_DIR,
    SEED,
)
from src.dataset import dataset_summary, get_dataloaders, get_split
from src.model import build_model, save_checkpoint
from src.preprocessing import get_eval_transform, get_train_transform


def set_seed(seed: int = SEED) -> None:
    """Sets seeds across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_class_weights(train_labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Computes inverse frequency class weights to balance CrossEntropyLoss:
    w_c = total_samples / (num_classes * count_c)
    """
    flat = train_labels.squeeze()
    count_0 = np.sum(flat == 0)
    count_1 = np.sum(flat == 1)
    total = len(flat)
    w0 = total / (2.0 * max(1, count_0))
    w1 = total / (2.0 * max(1, count_1))
    weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    print(f"[Train] Class counts - Normal: {count_0}, Pneumonia: {count_1}")
    print(f"[Train] Loss weights - Normal: {w0:.4f}, Pneumonia: {w1:.4f}")
    return weights


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Runs evaluation and returns loss, accuracy, AUROC, sensitivity, specificity."""
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
    probs = torch.softmax(logits_cat, dim=1)[:, 1].numpy()
    preds = torch.argmax(logits_cat, dim=1).numpy()

    accuracy = float(np.mean(preds == targets_cat))

    # Confusion matrix elements
    tn = int(np.sum((preds == 0) & (targets_cat == 0)))
    fp = int(np.sum((preds == 1) & (targets_cat == 0)))
    fn = int(np.sum((preds == 0) & (targets_cat == 1)))
    tp = int(np.sum((preds == 1) & (targets_cat == 1)))

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    try:
        auroc = float(roc_auc_score(targets_cat, probs))
    except Exception:
        auroc = 0.5

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "auroc": auroc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def save_training_plots(logs: List[Dict], output_path: Path) -> None:
    """Plots training curves and saves to disk."""
    epochs = [x["epoch"] for x in logs]
    train_losses = [x["train_loss"] for x in logs]
    val_losses = [x["val_loss"] for x in logs]
    val_aurocs = [x["val_auroc"] for x in logs]
    val_accs = [x["val_acc"] for x in logs]
    val_sens = [x["val_sensitivity"] for x in logs]
    val_specs = [x["val_specificity"] for x in logs]

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker="o", color="#e74c3c")
    plt.plot(epochs, val_losses, label="Val Loss", marker="s", color="#3498db")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CrossEntropy Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Metrics plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_aurocs, label="Val AUROC", marker="^", color="#2ecc71")
    plt.plot(epochs, val_accs, label="Val Accuracy", marker="v", color="#9b59b6")
    plt.plot(epochs, val_sens, label="Val Sensitivity", linestyle="--", color="#f39c12")
    plt.plot(epochs, val_specs, label="Val Specificity", linestyle=":", color="#34495e")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_model(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = SEED,
    smoke: bool = False,
    patience: int = DEFAULT_PATIENCE,
) -> Dict:
    """Main training execution function."""
    set_seed(seed)

    # Hardware detection
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("\n" + "=" * 65)
    print("             MEDGUARD AI — TRAINING INITIALIZATION")
    print("=" * 65)
    print(f"Device: {device}")
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        cuda_ver = torch.version.cuda
        print(f"GPU Name: {gpu_name}")
        print(f"CUDA Version: {cuda_ver}")
    else:
        gpu_name = "N/A"
        cuda_ver = "N/A"
        print("GPU: CPU only (CUDA unavailable)")

    # Ensure dataset summary exists
    ds_meta = dataset_summary(DATASET_PATH)

    # Get data loaders
    train_imgs, train_lbls = get_split("train", DATASET_PATH)
    val_imgs, val_lbls = get_split("val", DATASET_PATH)

    class_weights = compute_class_weights(train_lbls, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    dataloaders = get_dataloaders(
        batch_size=batch_size,
        augment=True,
        path=DATASET_PATH,
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Smoke test mode handles smaller subset
    if smoke:
        print("[Train] SMOKE TEST MODE: training on 128 samples, val on 64 samples.")
        train_sub = Subset(train_loader.dataset, indices=list(range(min(128, len(train_loader.dataset)))))
        val_sub = Subset(val_loader.dataset, indices=list(range(min(64, len(val_loader.dataset)))))
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    # Build model (Pretrained ResNet-18)
    print(f"[Train] Building ResNet-18 (ImageNet pretrained weights)...")
    model = build_model(pretrained=True)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_auroc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    epochs_no_improve = 0

    logs: List[Dict] = []
    print("\nStarting Training Loop:")
    print(f"{'Epoch':<6} {'TrainLoss':<11} {'ValLoss':<9} {'ValAcc':<8} {'ValAUROC':<10} {'ValSens':<9} {'ValSpec':<9} {'LR':<9}")
    print("-" * 75)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        train_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss_accum += loss.item() * len(targets)
            train_samples += len(targets)

        scheduler.step()
        avg_train_loss = train_loss_accum / max(1, train_samples)

        # Validation evaluation
        val_metrics = evaluate_loader(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_auroc": round(val_metrics["auroc"], 4),
            "val_sensitivity": round(val_metrics["sensitivity"], 4),
            "val_specificity": round(val_metrics["specificity"], 4),
            "lr": round(current_lr, 7),
        }
        logs.append(epoch_record)

        print(
            f"{epoch:<6d} {epoch_record['train_loss']:<11.4f} {epoch_record['val_loss']:<9.4f} "
            f"{epoch_record['val_acc']:<8.4f} {epoch_record['val_auroc']:<10.4f} "
            f"{epoch_record['val_sensitivity']:<9.4f} {epoch_record['val_specificity']:<9.4f} "
            f"{current_lr:<9.2e}"
        )

        # Checkpoint selection: Best validation AUROC, tie-break lower val loss
        is_best = False
        val_auroc = val_metrics["auroc"]
        val_loss = val_metrics["loss"]

        if val_auroc > best_val_auroc + 1e-4:
            is_best = True
        elif abs(val_auroc - best_val_auroc) <= 1e-4 and val_loss < best_val_loss:
            is_best = True

        if is_best:
            best_val_auroc = val_auroc
            best_val_loss = val_loss
            best_epoch = epoch
            # Store copy of weights in CPU RAM
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience and not smoke:
            print(f"[Train] Early stopping triggered at epoch {epoch} (patience={patience})")
            break

    total_training_sec = time.time() - start_time
    print("-" * 75)
    print(f"Training completed in {total_training_sec:.1f}s. Best Epoch: {best_epoch} (Val AUROC: {best_val_auroc:.4f})")

    # Load best weights into model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Save log CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_csv_path = REPORTS_DIR / "training_log.csv"
    with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
        writer.writeheader()
        writer.writerows(logs)

    # Save training curves plot
    curves_png_path = REPORTS_DIR / "training_curves.png"
    save_training_plots(logs, curves_png_path)

    # Save checkpoint
    training_config = {
        "epochs": epochs,
        "actual_epochs_trained": len(logs),
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "smoke": smoke,
        "device": str(device),
        "gpu_name": gpu_name,
        "cuda_version": cuda_ver,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "class_weights": [float(w) for w in class_weights.cpu().numpy()],
        "training_time_sec": round(total_training_sec, 2),
    }

    best_val_metric = {
        "name": "val_auroc",
        "value": round(float(best_val_auroc), 4),
        "val_loss": round(float(best_val_loss), 4),
    }

    checkpoint_path = MODELS_DIR / "best_model.pth"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        training_config=training_config,
        best_epoch=best_epoch,
        best_val_metric=best_val_metric,
        dataset_meta=ds_meta,
        models_dir=MODELS_DIR,
    )

    return {
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "best_val_loss": best_val_loss,
        "log_csv": str(log_csv_path),
        "curves_png": str(curves_png_path),
        "checkpoint": str(checkpoint_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedGuard AI Training Loop")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE, help="Early stopping patience")
    parser.add_argument("--smoke", action="store_true", help="Run quick 3-epoch smoke test on small subset")
    args = parser.parse_args()

    try:
        train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            smoke=args.smoke,
            patience=args.patience,
        )
    except torch.cuda.OutOfMemoryError as oom_err:
        print(f"[Train] CUDA OOM encountered with batch_size={args.batch_size}. Retrying automatically with {FALLBACK_BATCH_SIZE}...")
        torch.cuda.empty_cache()
        train_model(
            epochs=args.epochs,
            batch_size=FALLBACK_BATCH_SIZE,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            smoke=args.smoke,
            patience=args.patience,
        )
