"""
Two-Step Inference Pipeline
============================
Step 1: Detect lesions in a mammogram (Faster R-CNN)
Step 2: Classify each detected ROI with BI-RADS score (EfficientNet-B4)

Usage:
    conda run -n mamography python inference_pipeline.py \\
        --dicom path/to/image.dcm \\
        --detector outputs/detector/best_detector.pt \\
        --classifier outputs/birads/best_model.pt \\
        [--score-thresh 0.4] [--img-size 1024] [--crop-size 224]

Output:
    Prints detected lesions with class, confidence and BI-RADS prediction.
    Saves an annotated image to outputs/inference/<filename>_result.png
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import cv2
import torch
import torchvision.ops as ops
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data.detection_dataset import load_dicom_float
from src.data.crop_dataset import crop_roi, load_dicom_gray
from src.models.lesion_detector import LesionDetector
from src.models.birads_classifier import BiRadsClassifier
from src.data.xml_parser import CLASS_NAMES, NUM_CLASSES
from src.data.inbreast_dataset import CLASS_NAMES_BINARY


# ---------------------------------------------------------------------------
# Transforms (inference only — no augmentation)
# ---------------------------------------------------------------------------
_DET_TRANSFORM = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

_CLS_TRANSFORM = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Colours for drawing (BGR)
_COLOURS = {
    1: (255, 100, 0),    # Mass        — blue
    2: (0, 200, 255),    # Calcification — yellow
    3: (0, 255, 100),    # Asymmetry   — green
}


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------
def load_detector(ckpt_path: str, device: torch.device) -> LesionDetector:
    model = LesionDetector(num_classes=NUM_CLASSES, pretrained=False)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def load_classifier(ckpt_path: str, device: torch.device,
                    num_classes: int = 2) -> BiRadsClassifier:
    model = BiRadsClassifier(num_classes=num_classes, pretrained=False)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_pipeline(
    dcm_path: str,
    detector: LesionDetector,
    classifier: BiRadsClassifier,
    device: torch.device,
    det_img_size: int = 1024,
    crop_size: int = 224,
    score_thresh: float = 0.4,
    nms_iou_thresh: float = 0.3,
) -> list[dict]:
    """
    Run the two-step pipeline on a single DICOM.

    Returns list of dicts:
        {
          "lesion_class":  str,   e.g. "Mass"
          "det_score":     float, detector confidence
          "box_orig":      list,  [x1,y1,x2,y2] in original pixel space
          "birads_class":  str,   e.g. "Malignant (4-6)"
          "birads_probs":  list,  softmax probabilities
        }
    """
    # ---- Step 1: Detect ------------------------------------------------
    img_rgb, scale, _, _ = load_dicom_float(dcm_path, det_img_size)
    det_in = _DET_TRANSFORM(image=img_rgb)["image"].unsqueeze(0).to(device)

    preds = detector([det_in[0]])   # Faster R-CNN expects list of tensors
    pred  = preds[0]

    boxes_det  = pred["boxes"].cpu()
    labels_det = pred["labels"].cpu()
    scores_det = pred["scores"].cpu()

    # Filter by score threshold
    keep = scores_det >= score_thresh
    boxes_det  = boxes_det[keep]
    labels_det = labels_det[keep]
    scores_det = scores_det[keep]

    if len(boxes_det) == 0:
        return []

    # NMS — applied per class so boxes from different classes don't suppress each other
    keep_nms = ops.batched_nms(boxes_det, scores_det, labels_det, nms_iou_thresh)
    boxes_det  = boxes_det[keep_nms]
    labels_det = labels_det[keep_nms]
    scores_det = scores_det[keep_nms]

    # ---- Step 2: Classify each ROI ------------------------------------
    img_gray = load_dicom_gray(dcm_path)    # original resolution

    results = []
    for box, lbl, score in zip(boxes_det, labels_det, scores_det):
        # Scale box back to original coordinates
        box_orig = (box.numpy() / scale).tolist()

        # Crop at original resolution
        crop = crop_roi(img_gray, box_orig, crop_size)
        cls_in = _CLS_TRANSFORM(image=crop)["image"].unsqueeze(0).to(device)

        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = classifier(cls_in)
        probs  = torch.softmax(logits, dim=1)[0].cpu().tolist()
        pred_cls = int(np.argmax(probs))

        results.append({
            "lesion_class":  CLASS_NAMES[int(lbl)] if int(lbl) < len(CLASS_NAMES) else "Unknown",
            "det_score":     float(score),
            "box_scaled":    box.tolist(),      # in det_img_size space (for drawing)
            "box_orig":      box_orig,
            "birads_class":  CLASS_NAMES_BINARY[pred_cls],
            "birads_probs":  probs,
        })

    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def draw_results(img_rgb: np.ndarray, results: list[dict],
                 out_path: str):
    """Draw bounding boxes with lesion class and BI-RADS prediction."""
    vis = img_rgb.copy()
    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r["box_scaled"]]
        lbl_int = list(CLASS_NAMES).index(r["lesion_class"]) \
                  if r["lesion_class"] in CLASS_NAMES else 1
        colour = _COLOURS.get(lbl_int, (200, 200, 200))

        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

        text = (f"{r['lesion_class']} {r['det_score']:.2f} | "
                f"{r['birads_class']} ({max(r['birads_probs']):.2f})")
        cv2.putText(vis, text, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1,
                    cv2.LINE_AA)

    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Annotated image saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Two-step mammography inference")
    root = Path(__file__).parent
    p.add_argument("--dicom",      default=str(next((root / "INbreast" / "AllDICOMs").glob("*.dcm"), "")),
                   help="Path to DICOM file")
    p.add_argument("--detector",   default=str(root / "outputs" / "detector" / "best_detector.pt"),
                   help="Detector checkpoint (.pt)")
    p.add_argument("--classifier", default=str(root / "outputs" / "birads" / "best_model.pt"),
                   help="Classifier checkpoint (.pt)")
    p.add_argument("--test-set",   action="store_true",
                   help="Run on all DICOMs in --dicom-dir and save aggregated results")
    p.add_argument("--dicom-dir",  default=str(root / "INbreast" / "AllDICOMs"),
                   help="Directory of DICOMs (used with --test-set)")
    p.add_argument("--score-thresh", type=float, default=0.4)
    p.add_argument("--nms-iou-thresh", type=float, default=0.3,
                   help="IoU threshold for post-processing NMS (default: 0.3)")
    p.add_argument("--img-size",   type=int,   default=1024)
    p.add_argument("--crop-size",  type=int,   default=224)
    p.add_argument("--num-classes", type=int,  default=2,
                   help="BI-RADS classifier output classes (2=binary)")
    p.add_argument("--output-dir", default="outputs/inference")
    return p.parse_args()


def process_single(dcm_path, detector, classifier, device, args, out_dir):
    """Run pipeline on one DICOM, print summary, save annotated image. Returns summary dict."""
    print(f"\nProcessing {Path(dcm_path).name} ...")
    results = run_pipeline(
        dcm_path=dcm_path,
        detector=detector,
        classifier=classifier,
        device=device,
        det_img_size=args.img_size,
        crop_size=args.crop_size,
        score_thresh=args.score_thresh,
        nms_iou_thresh=args.nms_iou_thresh,
    )

    stem = Path(dcm_path).stem

    if not results:
        print("  No lesions detected above score threshold.")
        return {"file": stem, "num_lesions": 0, "overall": "No detection", "max_malignancy": 0.0}

    print(f"  Found {len(results)} lesion(s):")
    for i, r in enumerate(results, 1):
        probs_str = ", ".join(f"{p:.3f}" for p in r["birads_probs"])
        print(f"    [{i}] {r['lesion_class']} (conf={r['det_score']:.3f}) "
              f"→ {r['birads_class']}  probs=[{probs_str}]")

    malignant_scores = [r["birads_probs"][1] for r in results if len(r["birads_probs"]) > 1]
    overall_score = max(malignant_scores) if malignant_scores else 0.0
    overall_label = CLASS_NAMES_BINARY[1 if overall_score >= 0.5 else 0]
    print(f"  Overall: {overall_label} (max malignancy={overall_score:.3f})")

    img_rgb, _, _, _ = load_dicom_float(dcm_path, args.img_size)
    draw_results(img_rgb, results, str(out_dir / f"{stem}_result.png"))

    return {
        "file": stem,
        "num_lesions": len(results),
        "overall": overall_label,
        "max_malignancy": round(overall_score, 4),
        "lesions": [{"class": r["lesion_class"], "det_score": round(r["det_score"], 4),
                     "birads": r["birads_class"],
                     "probs": [round(p, 4) for p in r["birads_probs"]]} for r in results],
    }


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading detector ...")
    detector   = load_detector(args.detector, device)
    print("Loading classifier ...")
    classifier = load_classifier(args.classifier, device, args.num_classes)

    if args.test_set:
        # ---- Batch mode: run on all DICOMs in dicom_dir --------------------
        dcm_files = sorted(Path(args.dicom_dir).glob("*.dcm"))
        print(f"\nTest-set mode: {len(dcm_files)} DICOMs in {args.dicom_dir}")

        summaries = []
        for dcm_path in dcm_files:
            summary = process_single(str(dcm_path), detector, classifier, device, args, out_dir)
            summaries.append(summary)

        # Save CSV summary
        csv_path = out_dir / "test_set_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "num_lesions", "overall", "max_malignancy"])
            writer.writeheader()
            writer.writerows([{k: s[k] for k in writer.fieldnames} for s in summaries])

        # Save full JSON (includes per-lesion detail)
        json_path = out_dir / "test_set_results.json"
        json_path.write_text(json.dumps(summaries, indent=2))

        # Print aggregate stats
        total = len(summaries)
        detected = sum(1 for s in summaries if s["num_lesions"] > 0)
        malignant = sum(1 for s in summaries if CLASS_NAMES_BINARY[1] in s["overall"])
        print(f"\n{'='*50}")
        print(f"Test set summary ({total} images)")
        print(f"  Lesions detected : {detected}/{total}")
        print(f"  Malignant        : {malignant}/{total}")
        print(f"  CSV saved to     : {csv_path}")
        print(f"  JSON saved to    : {json_path}")

    else:
        # ---- Single image mode --------------------------------------------
        process_single(args.dicom, detector, classifier, device, args, out_dir)


if __name__ == "__main__":
    main()
