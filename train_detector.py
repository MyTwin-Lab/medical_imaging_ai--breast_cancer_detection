"""
Step 1 — Lesion Detection Training
====================================
Trains Faster R-CNN (ResNet-50 FPN) on INbreast to detect:
  1=Mass, 2=Calcification, 3=Asymmetry

Usage:
    conda run -n mamography python train_detector.py [OPTIONS]

Options:
    --img-size   resize longer side to this (default: 1024)
    --epochs     number of epochs            (default: 30)
    --batch-size training batch size         (default: 4)
    --lr         initial learning rate       (default: 5e-4)
    --val-split  fraction for validation     (default: 0.2)
    --seed       random seed                 (default: 42)
    --output-dir checkpoint output dir      (default: outputs/detector)
    --no-pretrain disable COCO weights       (flag)
    --workers    DataLoader workers          (default: 4)
"""

import argparse
import random
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.detection_dataset import INbreastDetectionDataset
from src.data.patient_split import patient_aware_split
from src.models.lesion_detector import LesionDetector
from src.data.xml_parser import CLASS_NAMES, NUM_CLASSES


ROOT      = Path(__file__).parent
DICOM_DIR = ROOT / "INbreast" / "AllDICOMs"
XML_DIR   = ROOT / "INbreast" / "AllXML"
CSV_PATH  = ROOT / "INbreast" / "INbreast.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """Custom collate: returns (list_of_images, list_of_targets)."""
    return tuple(zip(*batch))


def compute_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes (xyxy format)."""
    inter_x1 = torch.max(box_a[:, None, 0], box_b[None, :, 0])
    inter_y1 = torch.max(box_a[:, None, 1], box_b[None, :, 1])
    inter_x2 = torch.min(box_a[:, None, 2], box_b[None, :, 2])
    inter_y2 = torch.min(box_a[:, None, 3], box_b[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter   = inter_w * inter_h

    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    union  = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


@torch.no_grad()
def compute_map(predictions: list, targets: list,
                iou_threshold: float = 0.5,
                num_classes: int = NUM_CLASSES) -> dict[str, float]:
    """
    Compute per-class Average Precision at IoU=0.5 and mean AP.
    Returns dict: {"mAP": float, "AP_Mass": float, ...}
    """
    # Collect detections and GTs per class
    det_by_class = defaultdict(list)   # class → [(score, tp_flag)]
    gt_count     = defaultdict(int)    # class → total GT boxes

    for pred, tgt in zip(predictions, targets):
        pred_boxes  = pred["boxes"].cpu()
        pred_labels = pred["labels"].cpu()
        pred_scores = pred["scores"].cpu()
        gt_boxes    = tgt["boxes"].cpu()
        gt_labels   = tgt["labels"].cpu()

        matched_gt = set()

        # Sort detections by score (desc)
        order = pred_scores.argsort(descending=True)
        for di in order:
            cls   = int(pred_labels[di])
            score = float(pred_scores[di])

            # Count GTs for this class
            gt_mask = gt_labels == cls
            gt_cls  = gt_boxes[gt_mask]
            gt_inds = gt_mask.nonzero(as_tuple=True)[0].tolist()

            gt_count[cls] += 0  # ensure key exists

            tp = 0
            if len(gt_cls) > 0:
                iou = compute_iou(pred_boxes[di].unsqueeze(0), gt_cls)[0]
                best_iou, best_j = iou.max(0)
                best_gt_idx = gt_inds[int(best_j)]
                if float(best_iou) >= iou_threshold and best_gt_idx not in matched_gt:
                    tp = 1
                    matched_gt.add(best_gt_idx)

            det_by_class[cls].append((score, tp))

        for cls_idx in gt_labels.tolist():
            gt_count[cls_idx] += 1

    # Compute AP per class
    ap_dict: dict[str, float] = {}
    all_ap = []
    for cls in range(1, num_classes):     # skip background (0)
        dets     = sorted(det_by_class[cls], key=lambda x: -x[0])
        n_gt     = gt_count[cls]
        cls_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"cls{cls}"

        if n_gt == 0:
            ap_dict[f"AP_{cls_name}"] = float("nan")
            continue

        tp_cum = np.cumsum([d[1] for d in dets]).astype(float)
        fp_cum = np.cumsum([1 - d[1] for d in dets]).astype(float)
        precision = tp_cum / (tp_cum + fp_cum + 1e-9)
        recall    = tp_cum / n_gt

        # AP using 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            prec_at_t = precision[recall >= t].max() if (recall >= t).any() else 0.0
            ap += prec_at_t / 11.0

        ap_dict[f"AP_{cls_name}"] = ap
        all_ap.append(ap)

    ap_dict["mAP"] = float(np.mean(all_ap)) if all_ap else 0.0
    return ap_dict


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")

    for images, targets in pbar:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=scaler.is_enabled()):
            loss_dict = model(images, targets)
            loss      = sum(loss_dict.values())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []

    for images, targets in tqdm(loader, desc="  eval ", leave=False, unit="batch"):
        images = [img.to(device) for img in images]
        preds  = model(images)
        all_preds.extend([{k: v.cpu() for k, v in p.items()} for p in preds])
        all_targets.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

    return compute_map(all_preds, all_targets)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Lesion Detector Training")
    p.add_argument("--img-size",    type=int,   default=1024)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch-size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--val-split",   type=float, default=0.2)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--output-dir",  default="outputs/detector")
    p.add_argument("--no-pretrain", action="store_true")
    p.add_argument("--workers",     type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build full dataset for index splitting
    # ------------------------------------------------------------------
    full_ds = INbreastDetectionDataset(
        csv_path=str(CSV_PATH),
        dicom_dir=str(DICOM_DIR),
        xml_dir=str(XML_DIR),
        split="train",
        img_size=args.img_size,
    )
    idx_train, idx_val = patient_aware_split(
        str(CSV_PATH), full_ds.get_file_names(),
        val_fraction=args.val_split, seed=args.seed,
    )
    print(f"Split → train: {len(idx_train)}, val: {len(idx_val)} (patient-aware)")

    # ------------------------------------------------------------------
    # Build split datasets
    # ------------------------------------------------------------------
    train_ds = INbreastDetectionDataset(
        csv_path=str(CSV_PATH), dicom_dir=str(DICOM_DIR),
        xml_dir=str(XML_DIR), split="train",
        img_size=args.img_size, indices=idx_train,
    )
    val_ds = INbreastDetectionDataset(
        csv_path=str(CSV_PATH), dicom_dir=str(DICOM_DIR),
        xml_dir=str(XML_DIR), split="val",
        img_size=args.img_size, indices=idx_val,
    )

    mp_context = "spawn" if args.workers > 0 else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=args.workers == 0,
        collate_fn=collate_fn, multiprocessing_context=mp_context,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=args.workers == 0,
        collate_fn=collate_fn, multiprocessing_context=mp_context,
        persistent_workers=args.workers > 0,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = LesionDetector(
        num_classes=NUM_CLASSES,
        pretrained=not args.no_pretrain,
    ).to(device)

    # ------------------------------------------------------------------
    # Optimizer — SGD with momentum (standard for Faster R-CNN)
    # ------------------------------------------------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=0.9, weight_decay=5e-4,
    )
    # Warmup for 2 epochs, then cosine decay
    warmup_epochs = 2
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_map   = 0.0
    best_epoch = 0
    patience   = 10
    no_improve = 0

    log_path = out_dir / "training_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,mAP,AP_Mass,AP_Calcification,AP_Asymmetry,lr\n")

    print(f"\nTraining for {args.epochs} epochs\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        ap_dict    = evaluate(model, val_loader, device)
        scheduler.step()

        map_val = ap_dict["mAP"]
        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        ap_str = "  ".join(
            f"{k}={v:.3f}" for k, v in ap_dict.items() if k != "mAP"
        )
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss {train_loss:.4f} | mAP {map_val:.4f} | {ap_str} | "
            f"lr {lr_now:.2e} | {elapsed:.1f}s"
        )

        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{train_loss:.5f},{map_val:.4f},"
                f"{ap_dict.get('AP_Mass', float('nan')):.4f},"
                f"{ap_dict.get('AP_Calcification', float('nan')):.4f},"
                f"{ap_dict.get('AP_Asymmetry', float('nan')):.4f},"
                f"{lr_now:.2e}\n"
            )

        if map_val > best_map:
            best_map   = map_val
            best_epoch = epoch
            no_improve = 0
            ckpt_path  = out_dir / "best_detector.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "mAP":         map_val,
                "args":        vars(args),
            }, ckpt_path)
            print(f"  ✓ New best mAP {best_map:.4f} — saved to {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nBest mAP: {best_map:.4f} at epoch {best_epoch}")
    print(f"Checkpoint: {out_dir / 'best_detector.pt'}")


if __name__ == "__main__":
    main()
