"""
BI-RADS Classification Training Script
=======================================
Uses INbreast dataset with EfficientNet-B4 transfer learning.

Usage:
    conda run -n mamography python train_birads.py [OPTIONS]

Options (all have defaults):
    --mode        binary | multiclass        (default: binary)
    --img-size    pixel size, e.g. 512       (default: 512)
    --epochs      number of epochs           (default: 40)
    --batch-size  training batch size        (default: 8)
    --lr          initial learning rate      (default: 1e-4)
    --dropout     dropout before classifier  (default: 0.4)
    --val-split   fraction for val set       (default: 0.15)
    --test-split  fraction for test set      (default: 0.15)
    --seed        random seed                (default: 42)
    --output-dir  where to save checkpoints  (default: outputs/birads)
    --no-pretrain disable ImageNet weights   (flag)
    --workers     DataLoader num_workers     (default: 4)
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.inbreast_dataset import INbreastDataset
from src.data.crop_dataset import INbreastCropDataset
from src.data.patient_split import patient_aware_split
from src.models.birads_classifier import BiRadsClassifier
from src.utils.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent
DICOM_DIR = ROOT / "INbreast" / "AllDICOMs"
CSV_PATH  = ROOT / "INbreast" / "INbreast.csv"
XML_DIR   = ROOT / "INbreast" / "AllXML"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights to handle imbalance."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)   # avoid divide-by-zero
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Train / Eval loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_labels, all_preds = [], []

    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", enabled=scaler.is_enabled()):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader.dataset)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    return running_loss / n, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names, mode):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for imgs, labels in tqdm(loader, desc="  eval ", leave=False, unit="batch"):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    n = len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, all_probs,
                              class_names, mode)
    return running_loss / n, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="BI-RADS Classifier Training")
    p.add_argument("--mode",       default="binary",
                   choices=["binary", "multiclass"])
    p.add_argument("--img-size",   type=int,   default=512)
    p.add_argument("--epochs",     type=int,   default=40)
    p.add_argument("--batch-size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--dropout",    type=float, default=0.4)
    p.add_argument("--val-split",  type=float, default=0.15)
    p.add_argument("--test-split", type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--output-dir", default="outputs/birads")
    p.add_argument("--no-pretrain", action="store_true")
    p.add_argument("--workers",    type=int,   default=0)
    p.add_argument("--crop-mode",  action="store_true",
                   help="Train on ROI crops (Step 2) instead of full images")
    p.add_argument("--crop-size",  type=int,   default=224,
                   help="Crop size in pixels when --crop-mode is set")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build full dataset (for index splitting only)
    # ------------------------------------------------------------------
    from collections import Counter

    if args.crop_mode:
        print("Mode: ROI-crop classification (Step 2)")
        full_ds = INbreastCropDataset(
            csv_path=str(CSV_PATH), dicom_dir=str(DICOM_DIR),
            xml_dir=str(XML_DIR), split="train",
            mode=args.mode, crop_size=args.crop_size,
        )
    else:
        print("Mode: full-image classification")
        full_ds = INbreastDataset(
            csv_path=str(CSV_PATH), dicom_dir=str(DICOM_DIR),
            split="train", mode=args.mode, img_size=args.img_size,
        )

    labels = full_ds.get_labels()
    n_total = len(full_ds)
    num_classes = full_ds.num_classes
    class_names = full_ds.class_names
    print(f"Total samples: {n_total} | Classes: {num_classes} | Mode: {args.mode}")
    print(f"Label distribution: {dict(sorted(Counter(labels).items()))}")

    # ------------------------------------------------------------------
    # Patient-aware train / val / test split
    # ------------------------------------------------------------------
    idx_train, idx_val, idx_test = patient_aware_split(
        str(CSV_PATH), full_ds.get_file_names(),
        val_fraction=args.val_split, test_fraction=args.test_split,
        seed=args.seed,
    )
    print(f"Split → train: {len(idx_train)}, val: {len(idx_val)}, test: {len(idx_test)} (patient-aware)")

    # ------------------------------------------------------------------
    # Build split-specific datasets (with correct augmentation)
    # ------------------------------------------------------------------
    def make_ds(split_name, split_indices):
        if args.crop_mode:
            return INbreastCropDataset(
                csv_path=str(CSV_PATH), dicom_dir=str(DICOM_DIR),
                xml_dir=str(XML_DIR), split=split_name,
                mode=args.mode, crop_size=args.crop_size,
                indices=split_indices,
            )
        return INbreastDataset(
            csv_path=str(CSV_PATH), dicom_dir=str(DICOM_DIR),
            split=split_name, mode=args.mode, img_size=args.img_size,
            indices=split_indices,
        )

    train_ds = make_ds("train", idx_train)
    val_ds   = make_ds("val",   idx_val)
    test_ds  = make_ds("test",  idx_test)

    mp_context = "spawn" if args.workers > 0 else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers,
                              pin_memory=args.workers == 0,
                              multiprocessing_context=mp_context,
                              persistent_workers=args.workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=args.workers == 0,
                              multiprocessing_context=mp_context,
                              persistent_workers=args.workers > 0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=args.workers == 0,
                              multiprocessing_context=mp_context,
                              persistent_workers=args.workers > 0)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = BiRadsClassifier(
        num_classes=num_classes,
        pretrained=not args.no_pretrain,
        dropout=args.dropout,
    ).to(device)

    # ------------------------------------------------------------------
    # Class-weighted loss to handle imbalance
    # ------------------------------------------------------------------
    train_labels = train_ds.get_labels()
    class_weights = get_class_weights(train_labels, num_classes).to(device)
    print(f"Class weights: {class_weights.cpu().numpy().round(3)}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # Two-phase: first fine-tune head, then unfreeze all
    # ------------------------------------------------------------------
    # Phase 1: freeze backbone, train head only
    for param in model.backbone.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_auc   = 0.0
    best_epoch = 0
    patience   = 10
    no_improve = 0

    log_path = out_dir / "training_log.csv"
    with open(log_path, "w") as log_f:
        log_f.write("epoch,train_loss,train_acc,val_loss,val_acc,val_auc,val_f1,lr\n")

    unfreeze_epoch = max(5, args.epochs // 6)   # unfreeze after ~5 epochs

    print(f"\nTraining for {args.epochs} epochs (backbone unfreezes at epoch {unfreeze_epoch})\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Phase 2: unfreeze all layers
        if epoch == unfreeze_epoch:
            print(f"[Epoch {epoch}] Unfreezing backbone — using lower LR")
            for param in model.backbone.features.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch,
            )

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, class_names, args.mode,
        )
        scheduler.step()

        val_acc = val_metrics["accuracy"]
        val_auc = val_metrics["auc"]
        val_f1  = val_metrics["f1"]
        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} auc {val_auc:.3f} f1 {val_f1:.3f} | "
            f"lr {lr_now:.2e} | {elapsed:.1f}s"
        )

        with open(log_path, "a") as log_f:
            log_f.write(
                f"{epoch},{train_loss:.5f},{train_acc:.4f},"
                f"{val_loss:.5f},{val_acc:.4f},{val_auc:.4f},{val_f1:.4f},{lr_now:.2e}\n"
            )

        # Save best model
        if val_auc > best_auc:
            best_auc   = val_auc
            best_epoch = epoch
            no_improve = 0
            ckpt_path  = out_dir / "best_model.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_auc":     val_auc,
                "val_acc":     val_acc,
                "args":        vars(args),
            }, ckpt_path)
            print(f"  ✓ New best AUC {best_auc:.4f} — saved to {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # ------------------------------------------------------------------
    # Test evaluation with best checkpoint
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Loading best checkpoint (epoch {best_epoch}, val AUC {best_auc:.4f})")
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, test_metrics = evaluate(
        model, test_loader, criterion, device, class_names, args.mode,
    )
    print(f"\nTest Results:")
    print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  AUC      : {test_metrics['auc']:.4f}")
    print(f"  F1       : {test_metrics['f1']:.4f}")
    print(f"\nClassification Report:\n{test_metrics['report']}")
    print(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    # Save test results
    results_path = out_dir / "test_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Best epoch: {best_epoch} | Val AUC: {best_auc:.4f}\n\n")
        f.write(f"Accuracy : {test_metrics['accuracy']:.4f}\n")
        f.write(f"AUC      : {test_metrics['auc']:.4f}\n")
        f.write(f"F1       : {test_metrics['f1']:.4f}\n\n")
        f.write(f"Classification Report:\n{test_metrics['report']}\n")
        f.write(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
