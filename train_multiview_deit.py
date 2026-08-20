

from __future__ import annotations

import modal
# from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MINUTES = 60
VOLUME_NAME = "mammo-multiview"
VOLUME_PATH = Path("/root/data")
DATA_NAME = "CBIS-DDSM"
DATA_DIR = VOLUME_PATH / f"dataset/mammo/{DATA_NAME}"
CONFIG = {
    "BATCH_SIZE": 8,  
    "NUM_EPOCHS": 30,
    "NUM_CLASSES": 1,
    "EARLY_STOPPING_PATIENCE": 5, 
    "TARGET_SIZE": 224,
    "LEARNING_RATE_BACKBONE": 1e-5,
    "LEARNING_RATE_HEAD": 5e-4,
    "WEIGHT_DECAY_BACKBONE": 1e-4,
    "WEIGHT_DECAY_HEAD": 1e-2,
    "GLOBAL_DROPOUT": 0.3, # Increased to prevent multi-view overfitting
    "SCHEDULER_ETA_MIN": 1e-6,
    "SMOOTHING": 0.1,
    "MODEL_NAME": 'deit_small_patch16_224',
    "AUGMENTATION_TRAIN": {
        "Resize": 224,
        "RandomRotation_degrees": 30,
        "RandomHorizontalFlip_p": 0.5,
        "RandomVerticalFlip_p": 0.5,
        "ColorJitter_brightness": 0.2,
        "ColorJitter_contrast": 0.2,
        "Normalize_mean": [0.485, 0.456, 0.406],
        "Normalize_std": [0.229, 0.224, 0.225]
    }
}



image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install(["git", "libglib2.0-0", "libgl1"])
         .pip_install(["numpy<2.0.0", "torch==2.2.2", "torchvision==0.17.2", 
             "opencv-python-headless>=4.1.2","scipy","pandas","scikit-learn",
             "tensorboard", "pydicom", "matplotlib", "tqdm", "timm", "grad-cam",
             "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "gdcm"]))


app = modal.App(image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)



with image.imports():
    import numpy as np
    from PIL import Image

    import torch



def normalize_label(pathology: str) -> str:
    text = (pathology or "").strip().upper()
    if text in {"BENIGN", "BENIGN_WITHOUT_CALLBACK"}:
        return "benign"
    if text in {"MALIGNANT"}:
        return "malignant"
    return text.lower()


def extract_uid(file_path: str) -> str:
    parts = [p for p in file_path.strip().replace("\\", "/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Unexpected DDSM path format: {file_path}")
    return parts[-2]


def load_metadata(metadata_csv: str) -> Dict[str, Tuple[str, int]]:
    import csv
    lookup = {}
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            series_uid = row["SeriesInstanceUID"].strip()
            file_location = row["SeriesDescription"].strip()
            num_images = int(row["SeriesNumber"])
            lookup[series_uid] = (file_location, num_images)
    return lookup


def resolve_dicom_path(base_dir: str, relative_path: str) -> str:
    rel = relative_path.strip().replace("\\", "/")
    filename = Path(rel).name
    matches = list(Path(base_dir).rglob(filename))
    dcm_matches = [m for m in matches if m.suffix.lower() == ".dcm"]
    return str(dcm_matches[0]) if dcm_matches else None


def build_ddsm_subjects(base_dir: str, 
                        dicom_csv: str, 
                        stage: str, 
                        require_both_views: bool = False) -> List[Dict[str, Any]]:
    from collections import defaultdict
    import pandas as pd

    df = pd.read_csv(dicom_csv)
    subject_dict = defaultdict(lambda: {
        "patient_id": None,
        "side": None,
        "pathology": "benign",
        "abnormality_type": None,
        "views": {},
    })

    selected_opt = (df['SeriesDescription'].str.contains('full', na=False) & df['PatientID'].str.contains(stage, na=False))
    subset = df.loc[selected_opt].copy() #+ "/" + df.loc[selected_opt, "file_path"].str.split("dicom/", expand=True)[1]

    for _, row in subset.iterrows():
        patient_id = str(row["PatientID"]).strip()
        
        side = str(row.get("Laterality", "")).strip().upper()
        view = str(row.get("PatientOrientation", "")).strip().upper()
        # pathology = str(row["Pathology"]).strip().upper()
        pathology = "BENIGN"
        file_path = str(row["file_path"]).strip()

        if view not in {"CC", "MLO"}:
            continue

        if not file_path:
            continue

        key = (patient_id, side)
        if key not in subject_dict:
            subject_dict[key]["patient_id"] = patient_id
            subject_dict[key]["side"] = side
            subject_dict[key]["pathology"] = pathology
            # subject_dict[key]["abnormality_type"] = abnormality_type
            subject_dict[key]["file_path"] = file_path

        # print(key)
        
        subject_dict[key]["views"][view] = {
            "image": str(Path(base_dir) / file_path),
            "mask": None,
            "crop": None,
        }

    subjects = []
    for entry in subject_dict.values():
        views = entry["views"]
        # print(views.keys(), "CC" in list(views.keys()))
        if require_both_views:
            if "CC" in list(views.keys()) and "MLO" in list(views.keys()):
                subjects.append(entry)
        else:
            subjects.append(entry)

    print(len(subjects))
    return subjects


def crop_breast(image: Image.Image) -> Image.Image:
    """Removes text artifacts and background by cropping to the largest contour."""
    import cv2
    img_np = np.array(image.convert('L'))
    _, thresh = cv2.threshold(img_np, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add a small 2% padding around the bounding box
    pad_x = int(w * 0.02)
    pad_y = int(h * 0.02)
    
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_np.shape[1], x + w + pad_x)
    y2 = min(img_np.shape[0], y + h + pad_y)
    
    return image.crop((x1, y1, x2, y2))


def build_transform(split: str):
    from torchvision import transforms as T
    aug = CONFIG["AUGMENTATION_TRAIN"]
    if split == "train":
        return T.Compose([
            # T.ToPILImage(),
            # T.RandomResizedCrop(TARGET_SIZE, scale=(0.8, 1.0)),
            T.Resize((aug["Resize"], aug["Resize"])), 
            T.RandomRotation(degrees=aug["RandomRotation_degrees"]),
            T.RandomHorizontalFlip(p=aug["RandomHorizontalFlip_p"]),
            T.RandomVerticalFlip(p=aug["RandomVerticalFlip_p"]),
            # T.ColorJitter(brightness=aug["ColorJitter_brightness"], contrast=aug["ColorJitter_contrast"]),
            # T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            # T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            T.ToTensor(),
            T.Normalize(mean=aug["Normalize_mean"], std=aug["Normalize_std"]),
        ])
    else:
        return T.Compose([
            # T.ToPILImage(),
            T.Resize((aug["Resize"], aug["Resize"])),
            T.ToTensor(),
            T.Normalize(mean=aug["Normalize_mean"], std=aug["Normalize_std"]),
        ])


def inspect_subjects(dicom_csv: str, stage: str = None, 
                     patient_col: str = "PatientID", 
                     side_col: str = "Laterality", 
                     view_col: str = "PatientOrientation", 
                     file_col: str = "file_path", 
                     desc_col: str = "SeriesDescription") -> List[Dict[str, Any]]:
    import pandas as pd
    df = pd.read_csv(dicom_csv, low_memory=False)
    if stage:
        stage_mask = df[patient_col].astype(str).str.contains(stage, na=False)
        df = df.loc[stage_mask]

    groups = defaultdict(lambda: {"patient_id": None, "side": None, "views": set(), "rows": []})
    for _, row in df.iterrows():
        patient_id = str(row[patient_col]).strip()
        side = str(row[side_col]).strip().upper()
        view = str(row[view_col]).strip().upper()
        file_path_val = str(row[file_col]).strip()
        desc_val = str(row[desc_col]).strip()

        if not patient_id or not side or not view or not file_path_val:
            continue

        group_key = (patient_id, side)
        groups[group_key]["patient_id"] = patient_id
        groups[group_key]["side"] = side
        groups[group_key]["views"].add(view)
        groups[group_key]["rows"].append(row.to_dict())

    summary = []
    for (patient_id, side), group in groups.items():
        summary.append({
            "patient_id": patient_id,
            "side": side,
            "views": list(group["views"]),
            "has_cc": "CC" in group["views"],
            "has_mlo": "MLO" in group["views"],
            "num_rows": len(group["rows"]),
        })
    
    return sorted(summary, key=lambda x: (x["patient_id"], x["side"]))


def print_summary(summary: List[Dict[str, Any]]):
    total = len(summary)
    both = sum(1 for s in summary if s["has_cc"] and s["has_mlo"])
    only_cc = sum(1 for s in summary if s["has_cc"] and not s["has_mlo"])
    only_mlo = sum(1 for s in summary if not s["has_cc"] and s["has_mlo"])
    print(f"Total subjects: {total}")
    print(f"Subjects with both CC and MLO: {both}")
    print(f"Subjects with only CC: {only_cc}")
    print(f"Subjects with only MLO: {only_mlo}")


def pad_collate_fn(batch):
    """
    Custom collate function to handle variable number of views per patient.
    Pads single-view examinations with a blank tensor so the batch can be stacked.
    """
    images_list, labels_list, views_list = zip(*batch)
    
    max_views = max([imgs.shape[0] for imgs in images_list])
    
    padded_images = []
    masks = []
    
    for imgs in images_list:
        num_views, C, H, W = imgs.shape
        
        if num_views < max_views:
            padding = torch.zeros((max_views - num_views, C, H, W))
            padded_img = torch.cat([imgs, padding], dim=0)
            
            # Create a mask (1 for real image, 0 for padded blank image)
            mask = torch.tensor([1] * num_views + [0] * (max_views - num_views))
        else:
            padded_img = imgs
            mask = torch.tensor([1] * max_views)
            
        padded_images.append(padded_img)
        masks.append(mask)
    
    batch_images = torch.stack(padded_images, dim=0) # Shape: (B, max_views, C, H, W)
    batch_labels = torch.stack(labels_list, dim=0)   # Shape: (B,)
    batch_masks = torch.stack(masks, dim=0)          # Shape: (B, max_views)
    
    return batch_images, batch_labels, batch_masks, views_list



def compute_metrics(labels, preds, probs):
    from sklearn.metrics import (classification_report, 
                                 confusion_matrix, 
                                 roc_auc_score)

    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    tn, fp, fn, tp = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
    class_1 = report.get("1", report.get("1",{}))
    macro_avg = report.get("macro avg", {})
    weighted_avg = report.get("weighted avg", {})
    try:
        auc = roc_auc_score(labels, probs)
    except Exception as e:
        print(f"Warning: Unable to compute ROC AUC score: {e}")
        auc = 0.0
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "f1_score": class_1.get("f1-score", 0.0),
        "recall": class_1.get("recall", 0.0),
        "precision": class_1.get("precision", 0.0),
        "accuracy": class_1.get("accuracy", 0.0),
        "macro_f1": macro_avg.get("f1-score", 0.0),
        "macro_recall": macro_avg.get("recall", 0.0),
        "macro_precision": macro_avg.get("precision", 0.0),
        "weighted_f1": weighted_avg.get("f1-score", 0.0),
        "weighted_recall": weighted_avg.get("recall", 0.0),
        "weighted_precision": weighted_avg.get("precision", 0.0),
        "roc_auc": auc
    }


def show_cam_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import cv2
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def save_confusion_matrix_plot(metrics, output_path, class_names=("Benign", "Malignant")):
    import matplotlib.pyplot as plt

    conf_matrix = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(conf_matrix.shape[1]),
        yticks=np.arange(conf_matrix.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    thresh = conf_matrix.max() / 2.0 if conf_matrix.size else 0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], "d"),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# @app.function(secrets=[modal.Secret.from_name("kaggle-secret")], volumes={str(VOLUME_PATH): volume}, timeout=30 * MINUTES)
# def download_dataset():
#     import os
#     import shutil
#     import kagglehub
#     # import subprocess
#     # import zipfile
#     # from pathlib import Path
#     print(os.environ["KAGGLE_KEY"])
#     # DATASET_URL = "https://www.dropbox.com/s/your_dataset_link/ddsm_dataset.zip?dl=1"
#     # dataset_zip_path = Path("/root/data/ddsm_dataset.zip")
#     # dataset_dir = Path("/root/data/dataset/mammo/DDSM")

#     # # Create the dataset directory if it doesn't exist
#     # dataset_dir.mkdir(parents=True, exist_ok=True)

#     # # Download the dataset zip file
#     # subprocess.run(["wget", "-O", str(dataset_zip_path), DATASET_URL], check=True)

#     # # Extract the zip file
#     # with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
#     #     zip_ref.extractall(dataset_dir)

#     # # Clean up the zip file
#     # dataset_zip_path.unlink()
    
#     os.makedirs(DATA_DIR, exist_ok=True)

#     # Important: keep path out of dataset_download()
#     dataset_path = kagglehub.dataset_download(
#         "awsaf49/cbis-ddsm-breast-cancer-image-dataset"
#     )

#     print("Downloaded to:", dataset_path)

#     # Move the downloaded folder into your persistent volume
#     target_dir = DATA_DIR
#     if Path(target_dir).exists():
#         shutil.rmtree(target_dir)
#     shutil.move(dataset_path, target_dir)


def reshape_transform(tensor, height=14, width=14):
    # Reshape the tensor from (batch_size, num_patches, embedding_dim) to (batch_size, embedding_dim, height, width)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)  # Change to (batch_size, embedding_dim, height, width)
    return result



@app.function(gpu="A10", volumes={str(VOLUME_PATH): volume}, timeout=12 * 30 * MINUTES)
def run_training():
    import cv2
    import pandas as pd
    from tqdm import tqdm

    import timm
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils.data as data
    from torch.utils.tensorboard import SummaryWriter
    # from torchvision import models
    from torchvision.utils import make_grid, save_image

    # from pytorch_grad_cam import GradCAM
    # from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    class MultiViewDataset(data.Dataset):
        def __init__(self, csv_paths, transform=None):

            if isinstance(csv_paths, str):
                csv_paths = [csv_paths]

            dfs = [pd.read_csv(path) for path in csv_paths]
            self.df = pd.concat(dfs, ignore_index=True)
            self.transform = transform
            self.view_indices = {"L_CC": 0, "R_CC": 1, "L_MLO": 2, "R_MLO": 3}            
            grouped = self.df.groupby(['patient_id','side'])
            
            self.examinations = []
            for (patient, side), group in grouped:
                exam_data = {"patient_id": patient, "side": side, "label": group['label'].max()}
                for _, row in group.iterrows():
                    view_key = f"{side}_{row['view']}"
                    # exam_data[view_key] = {
                    #     "path": row['image_path'],
                    #     "label": row['label']
                    # }
                    if view_key not in exam_data:
                        exam_data[view_key] = {
                            "path": row['image_path'],
                            "label": row['label']
                        }
                self.examinations.append(exam_data)

        def __len__(self):
            return len(self.examinations)
        
        def __getitem__(self, idx):
            exam = self.examinations[idx]

            view_tensor = torch.zeros((4, 3, CONFIG["TARGET_SIZE"], CONFIG["TARGET_SIZE"]))  # 4 views, 3 channels, H, W
            view_mask = torch.zeros(4, dtype=torch.bool)  # To track which views are present

            side = exam['side']

            def load_view(view_key):
                if view_key in exam:
                    raw_path = str(exam[view_key]['path'])
                # if exam.get(view_key):
                    img = Image.open(f"{VOLUME_PATH}/dataset/mammo/{raw_path}").convert('RGB')
                    img = crop_breast(img)
                    if self.transform:
                        img = self.transform(img)
                    view_tensor[self.view_indices[view_key]] = img
                    view_mask[self.view_indices[view_key]] = True

            # if exam.get('L_CC'):
            load_view('L_CC')
            # if exam.get('L_MLO'):
            load_view('R_CC')
            # if exam.get('R_CC'):
            load_view('L_MLO')
            # if exam.get('R_MLO'):
            load_view('R_MLO')
                # view_mask[0] = True

            # left_malignant = True if (exam.get('L_CC') and exam['L_CC']['label'] == 1) or (exam.get('L_MLO') and exam['L_MLO']['label'] == 1) else False
            # right_malignant = True if (exam.get('R_CC') and exam['R_CC']['label'] == 1) or (exam.get('R_MLO') and exam['R_MLO']['label'] == 1) else False
            # is_malignant = 1.0 if left_malignant or right_malignant else 0.0
            targets = torch.tensor([float(exam["label"])], dtype=torch.float32)
            return view_tensor, view_mask, targets


    class MultiViewDeiT(nn.Module):
        def __init__(self, num_classes=1, model_name=CONFIG["MODEL_NAME"]):
            super(MultiViewDeiT, self).__init__()
            
            # Local Transformer (Pre-trained DeiT)
            self.local_deit = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
            embed_dim = self.local_deit.embed_dim
            
            # Learnable view positional embeddings (L-CC, L-MLO, R-CC, R-MLO)
            self.view_embed = nn.Parameter(torch.zeros(1, 4, 1, embed_dim))

            # Global CLS token
            self.global_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

            # init with variance scaling (truncated normal) for better convergence
            nn.init.trunc_normal_(self.view_embed, std=0.02)
            nn.init.trunc_normal_(self.global_cls_token, std=0.02)

            # Global Transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=8, 
                dim_feedforward=2048, 
                dropout=CONFIG["GLOBAL_DROPOUT"], # default 0.1
                batch_first=True
            )
            # Global Transformer blocks to jointly learn patch relationships across views
            self.global_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            
            # Final Multi-Label Classifier
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, num_classes)
            )

        def forward_local(self, x):
            # Pass image tensor directly through the timm DeiT
            return self.local_deit(x)

        def forward(self, x, mask):
            """
            x shape: (B, 4, 3, 224, 224)
            mask shape: (B, 4) boolean 
            """
            B, V, C, H, W = x.shape
            x_reshaped = x.view(B * V, C, H, W)
            
            # Local feature extraction
            local_features = self.forward_local(x_reshaped) 
            _, seq_len, embed_dim = local_features.shape
            
            # Reshape back to views and apply learnable view identifiers
            local_features = local_features.view(B, V, seq_len, embed_dim)
            local_features = local_features + self.view_embed
            
            # Flatten to single global token sequence 
            global_seq = local_features.reshape(B, V * seq_len, embed_dim)
            
            # Append global classification token
            global_cls = self.global_cls_token.expand(B, -1, -1)
            global_seq = torch.cat([global_cls, global_seq], dim=1) 
            
            # Build Attention Mask for missing views (True blocks attention)
            padding_mask = ~mask # Invert so missing views = True (pad)
            padding_mask = padding_mask.unsqueeze(-1).expand(B, V, seq_len).reshape(B, V * seq_len)
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device) # Never mask the CLS token
            global_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
            
            # Global feature extraction
            global_out = self.global_transformer(global_seq, src_key_padding_mask=global_padding_mask)
            
            # Final classification via the global CLS token's output
            cls_out = global_out[:, 0]
            logits = self.classifier(cls_out)
            
            return logits


    class CamWrapper(nn.Module):
        def __init__(self, model):
            super(CamWrapper, self).__init__()
            self.model = model

        def forward(self, x):
            x_unsqueezed = x.unsqueeze(0)
            dummy_mask = torch.ones(1, x.size(0), dtype=torch.bool).to(x.device)
            
            out = self.model(x_unsqueezed, dummy_mask) # Output shape: [1]
            
            # Expand the output to shape [4] so Grad-CAM can compute loss for all 4 images
            return out.expand(x.size(0),-1)
        

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_dir = VOLUME_PATH / "logs" / DATA_NAME / "multiview_deit"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))


    # print_summary(inspect_subjects(dicom_csv=dicom_path, stage="Mass-Test"))
    train_paths = [str(DATA_DIR / "csv/mass_train_manifest.csv"), str(DATA_DIR / "csv/calc_train_manifest.csv")]
    train_dataset = MultiViewDataset(csv_paths=train_paths, transform=build_transform(split="train"))
    train_dataloader = data.DataLoader(
        train_dataset, 
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True, 
        # collate_fn=pad_collate_fn
    )
    num_pos = sum([1 for exam in train_dataset.examinations if exam['label'] == 1])
    num_neg = len(train_dataset.examinations) - num_pos
    print(f"Training dataset: {len(train_dataset)} samples, {num_pos} positive, {num_neg} negative")
    imbalance_ratio = num_neg / num_pos if num_pos > 0 else 1.0
    test_paths = [str(DATA_DIR / "csv/mass_test_manifest.csv"), str(DATA_DIR / "csv/calc_test_manifest.csv")]
    test_dataset = MultiViewDataset(csv_paths=test_paths, transform=build_transform(split="test"))
    test_dataloader = data.DataLoader(
        test_dataset, 
        batch_size=CONFIG["BATCH_SIZE"], 
        shuffle=False, 
        # collate_fn=pad_collate_fn
    )

    model = MultiViewDeiT(model_name=CONFIG["MODEL_NAME"]).to(device)
    safe_ratio = min(imbalance_ratio, 5.0)
    pos_weight = torch.tensor([safe_ratio], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # optimizer = optim.Adam(model.parameters(), lr=1e-5)
    backbone_params = []
    new_params = []
    for name, param in model.named_parameters():
        if 'local_deit' in name:
            backbone_params.append(param)
        else:
            new_params.append(param)
    # optimizer = optim.Adam([
    #     {'params': backbone_params, 'lr': 1e-5},
    #     {'params': new_params, 'lr': 1e-4}
    # ])
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': CONFIG["LEARNING_RATE_BACKBONE"], 'weight_decay': CONFIG["WEIGHT_DECAY_BACKBONE"]},
        {'params': new_params, 'lr': CONFIG["LEARNING_RATE_HEAD"], 'weight_decay': CONFIG["WEIGHT_DECAY_HEAD"]}
    ])    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["NUM_EPOCHS"], eta_min=1e-6)
    best_val_auc = 0.0
    epochs_no_improve = 0
    metrics_history = []
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    model.train()
    for epoch in range(CONFIG["NUM_EPOCHS"]):

        total_loss = 0
        model.train()
        for _, (images, masks, labels) in enumerate(tqdm(train_dataloader, desc="Training")):
            # images shape: (B, max_views, C, H, W)
            # masks shape:  (B, max_views) -> 1 for real image, 0 for padding
            
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device) #mask to handle variable views
            
            optimizer.zero_grad()
            
            outputs = model(images, masks)
            # print(outputs.shape, labels.shape)
            smoothed_labels = labels.float() * (1 - CONFIG["SMOOTHING"]) + 0.5 * CONFIG["SMOOTHING"]
            loss = criterion(outputs.view_as(labels), smoothed_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            
            total_loss += loss.item()
        
        scheduler.step()
        train_loss_avg = total_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0
        all_val_labels, all_val_preds, all_val_probs = [], [], []
        best_samples, worst_samples = [], []

        with torch.no_grad():
            for images, masks, labels in tqdm(test_dataloader, desc="Validation"):
                images, masks, labels = images.to(device), masks.to(device), labels.to(device)
                outputs = model(images, masks)
                
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                probs_np, labels_np = probs.cpu().numpy(), labels.cpu().numpy()
                all_val_labels.extend(labels_np)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_probs.extend(probs_np)

                for j in range(images.size(0)):
                    error = abs(probs_np[j][0] - labels_np[j][0])
                    item = {
                        "error": error,
                        "image": images[j].cpu(),
                        "mask": masks[j].cpu(),
                        "label": int(labels_np[j][0]),
                        "pred": int(preds[j].item()),
                        "prob": float(probs_np[j][0])
                    }
                    best_samples.append(item)
                    best_samples.sort(key=lambda x: x["error"])
                    best_samples = best_samples[:5]
                    
                    worst_samples.append(item)
                    worst_samples.sort(key=lambda x: x["error"], reverse=True)
                    worst_samples = worst_samples[:5]

        val_loss_avg = total_val_loss / len(test_dataloader)
        metrics = compute_metrics(np.array(all_val_labels), np.array(all_val_preds), np.array(all_val_probs))
        val_acc = (np.array(all_val_labels) == np.array(all_val_preds)).mean()
        val_auc = metrics["roc_auc"]
        
        metrics_history.append({
            "Epoch": epoch + 1, 
            "Train_Loss": train_loss_avg, 
            "Val_Loss": val_loss_avg,
            "Val_Acc": val_acc, 
            "Val_AUC": val_auc, 
            "F1_Score": metrics["f1_score"],
            "TP": metrics["tp"],
            "FP": metrics["fp"],
            "TN": metrics["tn"],
            "FN": metrics["fn"],
            "Macro_F1": metrics["macro_f1"],
            "Macro_Recall": metrics["macro_recall"],
            "Macro_Precision": metrics["macro_precision"],         
        })
        pd.DataFrame(metrics_history).to_csv(log_dir / "metrics.csv", index=False)
        
        writer.add_scalar("Loss/Train", train_loss_avg, epoch)
        writer.add_scalar("Loss/Val", val_loss_avg, epoch)
        writer.add_scalar("Metrics/Acc", val_acc, epoch)
        writer.add_scalar("Metrics/AUC", val_auc, epoch)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        # Checkpointing & Native 5D Grad-CAM Image Generation
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0 
            
            torch.save(model.state_dict(), VOLUME_PATH / "models" / "best_multiview_deit.pth")
            print(f" Saved new best model with AUC: {best_val_auc:.4f}")
            
            img_save_dir = log_dir / "best_worst_predictions"
            img_save_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup Native Hooks for 5D Tensor
            target_layer = model.local_deit.blocks[-1].norm1 
            class HookManager:
                def __init__(self):
                    self.activations, self.gradients, self.f_hook, self.b_hook = [], [], None, None
                def register(self, layer):
                    self.f_hook = layer.register_forward_hook(lambda m, i, o: self.activations.append(o))
                    self.b_hook = layer.register_full_backward_hook(lambda m, gi, go: self.gradients.append(go[0]))
                def clear(self):
                    self.activations.clear(); self.gradients.clear()
                def remove(self):
                    if self.f_hook: self.f_hook.remove()
                    if self.b_hook: self.b_hook.remove()

            hooks = HookManager()
            hooks.register(target_layer)
            
            def save_native_cams(sample_list, prefix):
                for idx, sample in enumerate(sample_list):
                    valid_indices = torch.where(sample["mask"])[0]
                    if len(valid_indices) == 0: continue            
                    
                    img_tensor_display = sample["image"][sample["mask"]].to(device) * std + mean
                    img_tensor_display = torch.clamp(img_tensor_display, 0, 1)

                    try:
                        hooks.clear()
                        cam_input = sample["image"].unsqueeze(0).to(device) 
                        cam_mask = sample["mask"].unsqueeze(0).to(device)
                        
                        model.zero_grad()
                        output = model(cam_input, cam_mask)
                        output.backward()
                        
                        activations = hooks.activations[0] 
                        gradients = hooks.gradients[0]     
                        
                        combined_grids = []
                        for i_valid, actual_view_idx in enumerate(valid_indices):
                            act = activations[actual_view_idx, 1:, :].transpose(0, 1).view(384, 14, 14)
                            grad = gradients[actual_view_idx, 1:, :].transpose(0, 1).view(384, 14, 14)
                            
                            weights = grad.mean(dim=(1, 2), keepdim=True)
                            cam = torch.nn.functional.relu((weights * act).sum(dim=0))
                            
                            cam_min, cam_max = cam.min(), cam.max()
                            if cam_max > cam_min:
                                cam = (cam - cam_min) / (cam_max - cam_min)
                            
                            cam_resized = cv2.resize(cam.cpu().detach().numpy(), (224, 224))
                            
                            img_np = img_tensor_display[i_valid].cpu().numpy().transpose(1, 2, 0)
                            visualization = show_cam_on_image(img_np, cam_resized)
                            
                            orig_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
                            heatmap_tensor = torch.from_numpy(visualization).permute(2, 0, 1).float() / 255.0
                            combined_grids.extend([orig_tensor, heatmap_tensor])
                        
                        final_grid = make_grid(combined_grids, nrow=2, padding=2)
                        filename = f"{prefix}_{idx+1}_GT_{sample['label']}_Pred_{sample['pred']}_Prob_{sample['prob']:.3f}.png"
                        save_image(final_grid, img_save_dir / filename)
                    except Exception as e:
                        print(f"Native CAM failed for {prefix} {idx}: {e}")

            save_native_cams(best_samples, "Best")
            save_native_cams(worst_samples, "Worst")
            hooks.remove()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CONFIG["EARLY_STOPPING_PATIENCE"]:
                print(f"\nEarly stopping triggered! Best Val AUC: {best_val_auc:.4f}")
                break


@app.local_entrypoint()
def main():
    # download_dataset.remote()
    run_training.remote()

