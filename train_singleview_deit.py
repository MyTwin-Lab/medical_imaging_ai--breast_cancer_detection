from __future__ import annotations
import json
from pyexpat import model
from sched import scheduler
from shlex import split

import modal
from pathlib import Path
# from typing import Any, Dict, List, Tuple

MINUTES = 60
VOLUME_NAME = "mammo-multiview"
VOLUME_PATH = Path("/root/data")
DATA_NAME = "CBIS-DDSM"
DATA_DIR = VOLUME_PATH / f"dataset/mammo/{DATA_NAME}"
CONFIG = {
    "BATCH_SIZE": 64,
    "NUM_EPOCHS": 30,
    "EARLY_STOPPING_PATIENCE": 5,
    "NUM_CLASSES": 1,
    "TARGET_SIZE": 224,
    "LEARNING_RATE": 1e-5,
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

    import torch



def build_transform(split: str):
    from torchvision import transforms as T
    aug = CONFIG["AUGMENTATION_TRAIN"]
    if split == "train":
        return T.Compose([
            T.Resize((aug["Resize"], aug["Resize"])), 
            T.RandomRotation(degrees=aug["RandomRotation_degrees"]),
            T.RandomHorizontalFlip(p=aug["RandomHorizontalFlip_p"]),
            T.RandomVerticalFlip(p=aug["RandomVerticalFlip_p"]),
            T.ColorJitter(brightness=aug["ColorJitter_brightness"], contrast=aug["ColorJitter_contrast"]),
            T.ToTensor(),
            T.Normalize(mean=aug["Normalize_mean"], std=aug["Normalize_std"]),
        ])
    else:
        return T.Compose([
            T.Resize((aug["Resize"], aug["Resize"])),
            T.ToTensor(),
            T.Normalize(mean=aug["Normalize_mean"], std=aug["Normalize_std"]),
        ])


def compute_metrics(labels, preds, probs):
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(labels, preds).tolist()
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.0
    return {
        "confusion_matrix": conf_matrix,
        "f1_score": report.get("1.0", report.get("1", {})).get("f1-score", 0.0),
        "roc_auc": auc
    }


def reshape_transform(tensor, height=14, width=14):
    # Reshape DeiT sequence tokens back to a 2D spatial grid for Grad-CAM
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


@app.function(gpu="A10", volumes={str(VOLUME_PATH): volume}, timeout=6 * 30 * MINUTES)
def run_training():
    import json
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm
    
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils.data as data
    from torch.utils.tensorboard import SummaryWriter
    from torchvision.utils import make_grid, save_image
    
    import timm
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    
    

    class SingleViewDataset(data.Dataset):
        def __init__(self, csv_paths, transform=None):
            if isinstance(csv_paths, str):
                csv_paths = [csv_paths]
            dfs = [pd.read_csv(path) for path in csv_paths]
            self.df = pd.concat(dfs, ignore_index=True)
            self.df = self.df[self.df['image_path'].notna()].reset_index(drop=True)
            self.transform = transform

        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = f"{Path(VOLUME_PATH)}/dataset/mammo/{row['image_path']}"
            
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
                
            label = torch.tensor([float(row['label'])], dtype=torch.float32)
            return img, label

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = VOLUME_PATH / "logs" / DATA_NAME / "singleview_deit"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    with open(log_dir / "training_config.json", "w") as f:
        json.dump(CONFIG, f, indent=4)    

    train_paths = [str(DATA_DIR / "csv/mass_train_manifest.csv"), str(DATA_DIR / "csv/calc_train_manifest.csv")]
    train_dataset = SingleViewDataset(csv_paths=train_paths, transform=build_transform(split="train"))
    train_dataloader = data.DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    
    num_pos = sum(train_dataset.df['label'] == 1)
    num_neg = sum(train_dataset.df['label'] == 0)
    imbalance_ratio = num_neg / num_pos if num_pos > 0 else 1.0

    test_paths = [str(DATA_DIR / "csv/mass_test_manifest.csv"), str(DATA_DIR / "csv/calc_test_manifest.csv")]
    test_dataset = SingleViewDataset(csv_paths=test_paths, transform=build_transform(split="test"))
    test_dataloader = data.DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)


    model = timm.create_model(CONFIG['MODEL_NAME'], pretrained=True, num_classes=CONFIG['NUM_CLASSES']).to(device)
    
    pos_weight = torch.tensor([imbalance_ratio], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['NUM_EPOCHS'], eta_min=1e-6)

    max_samples = 12
    smoothing = CONFIG["SMOOTHING"]
    best_val_auc = 0.0
    
    # Tracking list for the CSV logger
    metrics_history = []

    for epoch in range(CONFIG["NUM_EPOCHS"]):
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']} Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            smoothed_labels = labels * (1 - smoothing) + 0.5 * smoothing
            loss = criterion(outputs, smoothed_labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        train_loss_avg = total_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0
        all_val_labels, all_val_preds, all_val_probs = [], [], []
        samples = []
        
        best_samples = []
        worst_samples = []

        target_layers = [model.blocks[-1].norm1]
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

        with torch.no_grad():
            for images, labels in tqdm(test_dataloader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                probs_np = probs.cpu().numpy()
                labels_np = labels.cpu().numpy()

                all_val_labels.extend(labels_np)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_probs.extend(probs_np)

                for j in range(images.size(0)):
                    error = abs(probs_np[j][0] - labels_np[j][0])
                    item = {
                        "error": error,
                        "image": images[j].cpu(),
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

                if len(samples) < max_samples:
                    for i in range(images.size(0)):
                        if len(samples) < max_samples:
                            samples.append({
                                "image": images[i],
                                "label": int(labels[i].item()),
                                "pred": int(preds[i].item()),
                                "prob": float(probs[i].item())
                            })

        val_loss_avg = total_val_loss / len(test_dataloader)
        y_true = np.array(all_val_labels)
        y_pred = np.array(all_val_preds)
        y_prob = np.array(all_val_probs)
        
        metrics = compute_metrics(y_true, y_pred, y_prob)
        val_acc = (y_true == y_pred).mean()
        val_auc = metrics["roc_auc"]
        
        epoch_metrics = {
            "Epoch": epoch + 1,
            "Train_Loss": train_loss_avg,
            "Val_Loss": val_loss_avg,
            "Val_Acc": val_acc,
            "Val_AUC": val_auc,
            "F1_Score": metrics["f1_score"]
        }
        metrics_history.append(epoch_metrics)
        pd.DataFrame(metrics_history).to_csv(log_dir / "metrics.csv", index=False)
        
        writer.add_scalar("Loss/Train", train_loss_avg, epoch)
        writer.add_scalar("Loss/Val", val_loss_avg, epoch)
        writer.add_scalar("Metrics/Acc", val_acc, epoch)
        writer.add_scalar("Metrics/AUC", val_auc, epoch)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        # Checkpointing & Early Stopping Logic
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0  # Reset patience counter
            
            checkpoint_path = VOLUME_PATH / "models"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path / "best_singleview_deit.pth")
            print(f"--> Saved new best model with AUC: {best_val_auc:.4f}")
            
            img_save_dir = log_dir / "best_worst_predictions"
            img_save_dir.mkdir(parents=True, exist_ok=True)
            
            def save_cam_samples(sample_list, prefix):
                with torch.enable_grad():
                    for idx, sample in enumerate(sample_list):
                        img_tensor = sample["image"].unsqueeze(0).to(device)
                        display_img = img_tensor[0] * std + mean
                        display_img = torch.clamp(display_img, 0, 1).cpu().numpy().transpose(1, 2, 0)
                        orig_tensor = torch.from_numpy(display_img).permute(2, 0, 1).float()

                        try:
                            targets = [ClassifierOutputTarget(0)]
                            grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
                            visualization = show_cam_on_image(display_img, grayscale_cam, use_rgb=True)
                            heatmap_tensor = torch.from_numpy(visualization).permute(2, 0, 1).float() / 255.0
                            final_grid = make_grid([orig_tensor, heatmap_tensor], nrow=2, padding=2)
                        except Exception as e:
                            final_grid = orig_tensor
                            
                        filename = f"{prefix}_{idx+1}_GT_{sample['label']}_Pred_{sample['pred']}_Prob_{sample['prob']:.3f}.png"
                        save_image(final_grid, img_save_dir / filename)
            
            save_cam_samples(best_samples, "Best")
            save_cam_samples(worst_samples, "Worst")
        else:
            epochs_no_improve += 1
            print(f"Early stopping counter: {epochs_no_improve} out of {CONFIG['EARLY_STOPPING_PATIENCE']}")

        # Standard TensorBoard Images
        with torch.enable_grad(): 
            for i, sample in enumerate(samples):
                img_tensor = sample["image"].unsqueeze(0)
                display_img = img_tensor[0] * std + mean
                display_img = torch.clamp(display_img, 0, 1).cpu().numpy().transpose(1, 2, 0)
                orig_tensor = torch.from_numpy(display_img).permute(2, 0, 1).float()

                try:
                    targets = [ClassifierOutputTarget(0)]
                    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
                    visualization = show_cam_on_image(display_img, grayscale_cam, use_rgb=True)
                    heatmap_tensor = torch.from_numpy(visualization).permute(2, 0, 1).float() / 255.0
                    final_grid = make_grid([orig_tensor, heatmap_tensor], nrow=2, padding=2)
                except Exception as e:
                    final_grid = orig_tensor 

                title = f"SingleView/Sample_{i}_Label_{sample['label']}_Pred_{sample['pred']}_Prob_{sample['prob']:.3f}"
                writer.add_image(title, final_grid, epoch)
                
        # Trigger Early Stopping
        if epochs_no_improve >= CONFIG["EARLY_STOPPING_PATIENCE"]:
            print(f"\nEarly stopping triggered after {epoch+1} epochs! Best Val AUC: {best_val_auc:.4f}")
            break


@app.local_entrypoint()
def main():
    run_training.remote()
    print("DONE!!")
