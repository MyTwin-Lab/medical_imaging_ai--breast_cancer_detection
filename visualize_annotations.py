"""
Visualize ground truth vs model predictions side-by-side.

Each output is a single PNG with two panels:
  Left  — ground truth boxes + polygon contours (from XML)
  Right — detector predictions (if --detector checkpoint is given)

Usage:
    # Ground truth only, 9 random images
    conda run -n mamography python visualize_annotations.py

    # GT + model predictions
    conda run -n mamography python visualize_annotations.py \\
        --detector outputs/detector/best_detector.pt

    # Specific image
    conda run -n mamography python visualize_annotations.py --file 22678622

    # Filter by lesion type
    conda run -n mamography python visualize_annotations.py --lesion Mass

    # How many random images
    conda run -n mamography python visualize_annotations.py --count 12
"""

import argparse
import csv
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pydicom
import torch

from src.data.xml_parser import build_annotation_map, CLASS_NAMES, NUM_CLASSES

ROOT      = Path(__file__).parent
DICOM_DIR = ROOT / "INbreast" / "AllDICOMs"
XML_DIR   = ROOT / "INbreast" / "AllXML"
CSV_PATH  = ROOT / "INbreast" / "INbreast.csv"
OUT_DIR   = ROOT / "outputs" / "visualizations"

# Colour per class index — BGR for OpenCV, RGB for matplotlib
CLASS_COLOURS_RGB = {
    1: (220,  60,  60),   # Mass          — red
    2: ( 60, 140, 220),   # Calcification — blue
    3: ( 60, 200, 100),   # Asymmetry     — green
}
POLY_ALPHA = 0.22


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_csv_labels(csv_path: str) -> dict[str, str]:
    result = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            result[row["File Name"].strip()] = row["Bi-Rads"].strip()
    return result


def find_dicom(dicom_dir: Path, file_name: str) -> Path | None:
    for f in dicom_dir.iterdir():
        if f.name.startswith(file_name) and f.suffix == ".dcm":
            return f
    return None


def load_dicom_rgb(dcm_path: Path, max_dim: int = 1024) -> tuple[np.ndarray, float]:
    """
    Load DICOM → uint8 RGB (H, W, 3), resized so longest side = max_dim.
    Returns (img_rgb, scale_factor).
    """
    ds    = pydicom.dcmread(str(dcm_path))
    pixel = ds.pixel_array.astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        pixel = pixel.max() - pixel

    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, pydicom.multival.MultiValue) \
             else float(ds.WindowCenter[0])
        ww = float(ds.WindowWidth)  if not isinstance(ds.WindowWidth,  pydicom.multival.MultiValue) \
             else float(ds.WindowWidth[0])
        lo, hi = wc - ww / 2, wc + ww / 2
        pixel  = np.clip(pixel, lo, hi)
        pixel  = (pixel - lo) / (hi - lo) * 255.0
    else:
        mn, mx = pixel.min(), pixel.max()
        if mx > mn:
            pixel = (pixel - mn) / (mx - mn) * 255.0

    img   = pixel.astype(np.uint8)
    h, w  = img.shape
    scale = max_dim / max(h, w)
    img   = cv2.resize(img, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)
    img   = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img, scale


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _put_label(canvas: np.ndarray, text: str, x: int, y: int,
               colour: tuple, font_scale: float = 0.5):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    # Background rectangle
    cv2.rectangle(canvas, (x, y - th - 4), (x + tw + 6, y + 2),
                  colour, -1)
    # Text (dark)
    cv2.putText(canvas, text, (x + 3, y - 1),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (20, 20, 20), 1, cv2.LINE_AA)


def draw_gt(img: np.ndarray, rois: list[dict], scale: float) -> np.ndarray:
    """Draw ground-truth polygons + bounding boxes."""
    canvas  = img.copy().astype(np.float32)
    overlay = img.copy().astype(np.float32)

    for roi in rois:
        cls    = roi["class"]
        colour = CLASS_COLOURS_RGB.get(cls, (180, 180, 180))
        poly   = (roi["poly"] * scale).astype(np.int32)
        b      = [int(v * scale) for v in roi["box"]]   # x1,y1,x2,y2

        # Filled polygon on overlay for soft shading
        cv2.fillPoly(overlay, [poly], colour)
        # Polygon contour
        cv2.polylines(canvas, [poly], isClosed=True, color=colour, thickness=2,
                      lineType=cv2.LINE_AA)
        # Bounding box (dashed look via thick + thin)
        cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]),
                      colour, 2, cv2.LINE_AA)
        # Label
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else "?"
        _put_label(canvas, name, b[0], b[1], colour)

    blended = cv2.addWeighted(overlay, POLY_ALPHA, canvas, 1 - POLY_ALPHA, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_preds(img: np.ndarray, predictions: dict,
               score_thresh: float = 0.4) -> np.ndarray:
    """Draw Faster R-CNN predictions on a copy of img."""
    canvas = img.copy()

    boxes  = predictions["boxes"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()

    keep = scores >= score_thresh
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    if len(boxes) == 0:
        # Write "No detections" centred on canvas
        h, w = canvas.shape[:2]
        text  = f"No detections (thresh={score_thresh})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(canvas, text, ((w - tw) // 2, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 80, 80), 1, cv2.LINE_AA)
        return canvas

    for box, lbl, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        cls    = int(lbl)
        colour = CLASS_COLOURS_RGB.get(cls, (180, 180, 180))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)
        name  = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else "?"
        label = f"{name} {score:.2f}"
        _put_label(canvas, label, x1, y1, colour)

    return canvas


# ---------------------------------------------------------------------------
# Detector inference
# ---------------------------------------------------------------------------
def load_detector(ckpt_path: str, device: torch.device):
    from src.models.lesion_detector import LesionDetector
    model = LesionDetector(num_classes=NUM_CLASSES, pretrained=False)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict(model, img_rgb: np.ndarray, device: torch.device) -> dict:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)
    preds  = model([tensor[0]])
    return {k: v.cpu() for k, v in preds[0].items()}


# ---------------------------------------------------------------------------
# Build sample list
# ---------------------------------------------------------------------------
def build_samples(lesion_filter: str | None) -> list[dict]:
    birads_map = load_csv_labels(str(CSV_PATH))
    ann_map    = build_annotation_map(str(XML_DIR))
    samples    = []

    for file_name, rois in ann_map.items():
        if not rois:
            continue
        if lesion_filter:
            rois = [r for r in rois
                    if CLASS_NAMES[r["class"]].lower() == lesion_filter.lower()]
            if not rois:
                continue
        dcm_path = find_dicom(DICOM_DIR, file_name)
        if dcm_path is None:
            continue
        samples.append({
            "file_name": file_name,
            "dcm_path":  dcm_path,
            "rois":      rois,
            "birads":    birads_map.get(file_name, "?"),
        })
    return samples


# ---------------------------------------------------------------------------
# Save one side-by-side image
# ---------------------------------------------------------------------------
def save_comparison(sample: dict, detector=None, device=None,
                    max_dim: int = 1024, score_thresh: float = 0.4):
    img, scale = load_dicom_rgb(sample["dcm_path"], max_dim)

    left  = draw_gt(img, sample["rois"], scale)

    if detector is not None:
        preds = predict(detector, img, device)
        right = draw_preds(img.copy(), preds, score_thresh)
        right_title = "Model Predictions"
    else:
        right = img.copy()
        h, w  = right.shape[:2]
        cv2.putText(right, "No model loaded", (w // 4, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 1)
        right_title = "Model Predictions (no checkpoint)"

    # Add column headers via matplotlib so we get nice text
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(left)
    axes[0].set_title(
        f"Ground Truth — {sample['file_name']}  |  BI-RADS {sample['birads']}\n"
        f"{len(sample['rois'])} ROI(s): "
        + ", ".join(CLASS_NAMES[r['class']] for r in sample['rois']),
        fontsize=10,
    )
    axes[0].axis("off")

    axes[1].imshow(right)
    axes[1].set_title(right_title, fontsize=10)
    axes[1].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=np.array(c) / 255, label=CLASS_NAMES[i])
        for i, c in CLASS_COLOURS_RGB.items()
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = OUT_DIR / f"{sample['file_name']}_birads{sample['birads']}.png"
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize GT vs model predictions (saved as individual PNGs)"
    )
    p.add_argument("--file",       default=None,
                   help="Single file name stem, e.g. 22678622")
    p.add_argument("--count",      type=int, default=9,
                   help="Number of random images (default 9)")
    p.add_argument("--lesion",     default=None,
                   choices=["Mass", "Calcification", "Asymmetry"],
                   help="Filter by lesion type")
    p.add_argument("--detector",   default=None,
                   help="Path to detector checkpoint (.pt)")
    p.add_argument("--score-thresh", type=float, default=0.4)
    p.add_argument("--img-size",   type=int, default=1024)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = None
    if args.detector:
        print(f"Loading detector from {args.detector} ...")
        detector = load_detector(args.detector, device)

    samples = build_samples(lesion_filter=args.lesion)
    if not samples:
        print("No annotated samples found.")
        return

    if args.file:
        samples = [s for s in samples if s["file_name"] == args.file]
        if not samples:
            print(f"'{args.file}' not found or has no annotations.")
            return
    else:
        random.seed(args.seed)
        samples = random.sample(samples, min(args.count, len(samples)))

    print(f"Saving {len(samples)} image(s) to {OUT_DIR}/\n")
    for s in samples:
        save_comparison(s, detector=detector, device=device,
                        max_dim=args.img_size, score_thresh=args.score_thresh)

    print(f"\nDone. {len(samples)} images saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
