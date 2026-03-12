"""
Detection dataset for INbreast — used to train Faster R-CNN.

Each sample returns:
    image  : FloatTensor [3, H, W]  values in [0, 1]
    target : dict with keys
               "boxes"   FloatTensor [N, 4]  (x1,y1,x2,y2)
               "labels"  Int64Tensor [N]
               "image_id" Int64Tensor scalar

Images are resized so the longer side = img_size, then zero-padded to square.
Bounding boxes are scaled accordingly.
"""

import os
import csv
import numpy as np
import cv2
import pydicom
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data.xml_parser import build_annotation_map, parse_xml


# -----------------------------------------------------------------------
# DICOM loading (same preprocessing as classifier but returns float [0,1])
# -----------------------------------------------------------------------
def load_dicom_float(dcm_path: str, img_size: int) -> tuple[np.ndarray, float, int, int]:
    """
    Load DICOM → resized uint8 HxW grayscale, then float RGB [0,1].

    Returns:
        img      : np.ndarray uint8 (img_size, img_size, 3)
        scale    : float  (scale applied to pixel coords)
        pad_h    : int    (rows of zero-padding added at bottom)
        pad_w    : int    (cols of zero-padding added at right)
    """
    ds = pydicom.dcmread(dcm_path)
    pixel = ds.pixel_array.astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        pixel = pixel.max() - pixel

    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, pydicom.multival.MultiValue) \
             else float(ds.WindowCenter[0])
        ww = float(ds.WindowWidth) if not isinstance(ds.WindowWidth, pydicom.multival.MultiValue) \
             else float(ds.WindowWidth[0])
        low, high = wc - ww / 2, wc + ww / 2
        pixel = np.clip(pixel, low, high)
        pixel = (pixel - low) / (high - low) * 255.0
    else:
        pmin, pmax = pixel.min(), pixel.max()
        if pmax > pmin:
            pixel = (pixel - pmin) / (pmax - pmin) * 255.0

    img = pixel.astype(np.uint8)

    h, w = img.shape
    scale = img_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = img_size - new_h
    pad_w = img_size - new_w
    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img, scale, pad_h, pad_w


def scale_boxes(boxes: list[list[float]], scale: float,
                img_size: int) -> np.ndarray:
    """Scale bounding boxes by `scale` and clamp to image bounds."""
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    arr = np.array(boxes, dtype=np.float32) * scale
    arr = np.clip(arr, 0, img_size - 1)
    return arr


# -----------------------------------------------------------------------
# Augmentation for detection
# -----------------------------------------------------------------------
def build_det_transforms(split: str) -> A.Compose:
    bbox_params = A.BboxParams(
        format="pascal_voc",        # [x1, y1, x2, y2]
        label_fields=["labels"],
        min_area=16,                # drop tiny boxes after augmentation
        min_visibility=0.3,
    )
    if split == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=bbox_params)
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=bbox_params)


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------
def _load_csv(csv_path: str) -> dict[str, str]:
    """Return {file_name: birads_label}."""
    result = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            result[row["File Name"].strip()] = row["Bi-Rads"].strip().lower()
    return result


def _find_dicom(dicom_dir: str, file_name: str) -> str | None:
    for fname in os.listdir(dicom_dir):
        if fname.startswith(file_name) and fname.endswith(".dcm"):
            return os.path.join(dicom_dir, fname)
    return None


class INbreastDetectionDataset(Dataset):
    """
    Dataset for lesion detection (Faster R-CNN).

    Samples with no ROI annotations contribute as negatives (empty boxes).
    """

    def __init__(
        self,
        csv_path: str,
        dicom_dir: str,
        xml_dir: str,
        split: str = "train",
        img_size: int = 1024,
        indices: list[int] | None = None,
    ):
        self.dicom_dir = dicom_dir
        self.img_size  = img_size
        self.transform = build_det_transforms(split)

        ann_map = build_annotation_map(xml_dir)
        label_map = _load_csv(csv_path)

        # Build sample list: (dicom_path, rois)
        self.samples = []
        processed_files = set()

        # 1. Samples that have XML annotations
        for file_name, rois in ann_map.items():
            dcm_path = _find_dicom(dicom_dir, file_name)
            if dcm_path is None:
                continue
            processed_files.add(file_name)
            self.samples.append((dcm_path, file_name, rois))

        # 2. DICOMs with no XML → negative samples
        for fname in os.listdir(dicom_dir):
            if not fname.endswith(".dcm"):
                continue
            file_name = fname.split("_")[0]
            if file_name in processed_files:
                continue
            dcm_path = os.path.join(dicom_dir, fname)
            self.samples.append((dcm_path, file_name, []))

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        print(f"[DetectionDataset/{split}] {len(self.samples)} images "
              f"({sum(1 for s in self.samples if s[2])} with ROIs)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        dcm_path, file_name, rois = self.samples[idx]

        img, scale, _, _ = load_dicom_float(dcm_path, self.img_size)

        # Build boxes and labels from ROI list
        raw_boxes  = [r["box"]   for r in rois]
        raw_labels = [r["class"] for r in rois]

        scaled_boxes = scale_boxes(raw_boxes, scale, self.img_size)

        # Albumentations augmentation
        if len(scaled_boxes) > 0:
            aug = self.transform(
                image=img,
                bboxes=scaled_boxes.tolist(),
                labels=raw_labels,
            )
            boxes  = aug["bboxes"]
            labels = aug["labels"]
        else:
            aug    = self.transform(image=img, bboxes=[], labels=[])
            boxes  = []
            labels = []

        # Build target dict expected by torchvision Faster R-CNN
        if boxes:
            boxes_t  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return aug["image"], target

    def get_file_names(self) -> list[str]:
        return [s[1] for s in self.samples]
