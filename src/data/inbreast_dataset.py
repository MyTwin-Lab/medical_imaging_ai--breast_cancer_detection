"""
INbreast Dataset loader for BI-RADS classification.

BI-RADS label mapping:
  Binary mode  : 1,2,3 → 0 (benign/negative), 4a,4b,4c,5,6 → 1 (malignant/positive)
  Multiclass   : 1→0, 2→1, 3→2, 4a→3, 4b→4, 4c→5, 5→6, 6→7
"""

import os
import csv
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


# BI-RADS raw label → integer index (for multiclass)
BIRADS_TO_IDX = {
    "1": 0, "2": 1, "3": 2,
    "4a": 3, "4b": 4, "4c": 5,
    "5": 6, "6": 7,
}

# BI-RADS raw label → binary (0 = benign, 1 = malignant)
BIRADS_BINARY = {
    "1": 0, "2": 0, "3": 0,
    "4a": 1, "4b": 1, "4c": 1,
    "5": 1, "6": 1,
}

CLASS_NAMES_BINARY = ["Benign (1-3)", "Malignant (4-6)"]
CLASS_NAMES_MULTI = ["BIRADS-1", "BIRADS-2", "BIRADS-3",
                     "BIRADS-4a", "BIRADS-4b", "BIRADS-4c",
                     "BIRADS-5", "BIRADS-6"]


def load_csv(csv_path: str) -> list[dict]:
    """Parse INbreast.csv and return list of records."""
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            birads = row["Bi-Rads"].strip().lower()
            file_name = row["File Name"].strip()
            if birads and file_name:
                records.append({"file_name": file_name, "birads": birads})
    return records


def find_dicom_path(dicom_dir: str, file_name: str) -> str | None:
    """Find DICOM file whose name starts with file_name."""
    for fname in os.listdir(dicom_dir):
        if fname.startswith(file_name) and fname.endswith(".dcm"):
            return os.path.join(dicom_dir, fname)
    return None


def load_dicom_as_uint8(dcm_path: str, target_size: int = 512) -> np.ndarray:
    """
    Load a DICOM mammogram and return a uint8 HxWx3 numpy array.

    Steps:
      1. Read pixel array
      2. Apply VOI LUT / window if available
      3. Normalize to [0, 255]
      4. Resize
      5. Apply CLAHE for local contrast enhancement
      6. Convert to 3-channel
    """
    ds = pydicom.dcmread(dcm_path)
    pixel_array = ds.pixel_array.astype(np.float32)

    # Photometric interpretation: MONOCHROME1 means 0=white; invert it
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array

    # Window/level normalization if tags are present
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, pydicom.multival.MultiValue) \
            else float(ds.WindowCenter[0])
        ww = float(ds.WindowWidth) if not isinstance(ds.WindowWidth, pydicom.multival.MultiValue) \
            else float(ds.WindowWidth[0])
        low = wc - ww / 2
        high = wc + ww / 2
        pixel_array = np.clip(pixel_array, low, high)
        pixel_array = (pixel_array - low) / (high - low) * 255.0
    else:
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax > pmin:
            pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0

    img = pixel_array.astype(np.uint8)

    # Resize (preserve aspect ratio by fitting into a square)
    h, w = img.shape
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                             cv2.BORDER_CONSTANT, value=0)

    # CLAHE for contrast enhancement (standard in mammography preprocessing)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Convert grayscale → 3-channel RGB (for pretrained ImageNet models)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def build_transforms(split: str, img_size: int = 512) -> A.Compose:
    """Return albumentations transform pipeline."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15,
                                       contrast_limit=0.15, p=0.4),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


class INbreastDataset(Dataset):
    """PyTorch Dataset for INbreast mammography BI-RADS classification."""

    def __init__(
        self,
        csv_path: str,
        dicom_dir: str,
        split: str = "train",          # "train" | "val" | "test"
        mode: str = "binary",          # "binary" | "multiclass"
        img_size: int = 512,
        indices: list[int] | None = None,   # subset indices (for fold splits)
    ):
        self.dicom_dir = dicom_dir
        self.mode = mode
        self.img_size = img_size
        self.transform = build_transforms(split, img_size)

        label_map = BIRADS_BINARY if mode == "binary" else BIRADS_TO_IDX

        raw_records = load_csv(csv_path)
        self.samples = []  # list of (dicom_path, label)

        skipped = 0
        for rec in raw_records:
            birads = rec["birads"]
            if birads not in label_map:
                skipped += 1
                continue
            dcm_path = find_dicom_path(dicom_dir, rec["file_name"])
            if dcm_path is None:
                skipped += 1
                continue
            self.samples.append((dcm_path, label_map[birads], rec["file_name"]))

        if skipped:
            print(f"[Dataset] Skipped {skipped} records (missing file or unknown label)")

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        self.num_classes = 2 if mode == "binary" else len(BIRADS_TO_IDX)
        self.class_names = CLASS_NAMES_BINARY if mode == "binary" else CLASS_NAMES_MULTI

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dcm_path, label, _ = self.samples[idx]
        img = load_dicom_as_uint8(dcm_path, self.img_size)
        augmented = self.transform(image=img)
        tensor = augmented["image"]
        return tensor, torch.tensor(label, dtype=torch.long)

    def get_labels(self) -> list[int]:
        return [s[1] for s in self.samples]

    def get_file_names(self) -> list[str]:
        return [s[2] for s in self.samples]
