"""
Crop-based dataset for Step-2 BI-RADS classification.

For each image that has XML ROI annotations, we crop each polygon bounding box
(with padding) and label the crop with the image-level BI-RADS score from CSV.

Images with no ROI annotations are skipped here — they are handled at the
image level by the standard INbreastDataset.
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

from src.data.xml_parser import build_annotation_map
from src.data.inbreast_dataset import (
    BIRADS_BINARY, BIRADS_TO_IDX, CLASS_NAMES_BINARY, CLASS_NAMES_MULTI,
)


PADDING_FACTOR = 0.20   # add 20% of box size as context padding


def load_dicom_gray(dcm_path: str) -> np.ndarray:
    """Load DICOM → uint8 grayscale (original resolution)."""
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
        mn, mx = pixel.min(), pixel.max()
        if mx > mn:
            pixel = (pixel - mn) / (mx - mn) * 255.0

    return pixel.astype(np.uint8)


def crop_roi(img_gray: np.ndarray, box: list[float],
             crop_size: int = 224) -> np.ndarray:
    """
    Crop a bounding box from a grayscale image with padding, resize to
    crop_size × crop_size, apply CLAHE, convert to 3-channel RGB uint8.
    """
    h, w = img_gray.shape
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * PADDING_FACTOR
    pad_y = bh * PADDING_FACTOR

    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w, int(x2 + pad_x))
    cy2 = min(h, int(y2 + pad_y))

    crop = img_gray[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        crop = img_gray[:crop_size, :crop_size]

    crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    crop  = clahe.apply(crop)
    crop  = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    return crop


def build_transforms(split: str) -> A.Compose:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.4),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


def _load_csv(csv_path: str) -> dict[str, str]:
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


class INbreastCropDataset(Dataset):
    """
    Each sample is a single ROI crop from a mammogram, labeled with the
    image-level BI-RADS score.

    One mammogram with N ROIs → N training samples.
    """

    def __init__(
        self,
        csv_path: str,
        dicom_dir: str,
        xml_dir: str,
        split: str = "train",
        mode: str = "binary",       # "binary" | "multiclass"
        crop_size: int = 224,
        indices: list[int] | None = None,
    ):
        self.crop_size = crop_size
        self.mode      = mode
        self.transform = build_transforms(split)

        label_map  = BIRADS_BINARY if mode == "binary" else BIRADS_TO_IDX
        birads_map = _load_csv(csv_path)
        ann_map    = build_annotation_map(xml_dir)

        # Build sample list: (dicom_path, box, label)
        self.samples: list[tuple[str, list[float], int]] = []
        skipped = 0

        for file_name, rois in ann_map.items():
            if not rois:          # no polygon annotations
                continue
            birads = birads_map.get(file_name, "").lower()
            if birads not in label_map:
                skipped += 1
                continue
            dcm_path = _find_dicom(dicom_dir, file_name)
            if dcm_path is None:
                skipped += 1
                continue
            label = label_map[birads]
            for roi in rois:
                self.samples.append((dcm_path, roi["box"], label, file_name))

        if skipped:
            print(f"[CropDataset] Skipped {skipped} images (no DICOM or unknown label)")

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        self.num_classes = 2 if mode == "binary" else len(BIRADS_TO_IDX)
        self.class_names = CLASS_NAMES_BINARY if mode == "binary" else CLASS_NAMES_MULTI
        print(f"[CropDataset/{split}] {len(self.samples)} ROI crops")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        dcm_path, box, label, _ = self.samples[idx]

        # Load full image at original resolution then crop
        img_gray = load_dicom_gray(dcm_path)
        crop     = crop_roi(img_gray, box, self.crop_size)

        aug    = self.transform(image=crop)
        tensor = aug["image"]
        return tensor, torch.tensor(label, dtype=torch.long)

    def get_labels(self) -> list[int]:
        return [s[2] for s in self.samples]

    def get_file_names(self) -> list[str]:
        return [s[3] for s in self.samples]
