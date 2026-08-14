
import re
import pandas as pd

from pathlib import Path
from typing import List, Dict, Any


import re

def parse_dicom_path(path_str: str) -> pd.Series:
    """Extracts metadata purely from the DICOM image_path string."""
    path_upper = str(path_str).upper()
    
    # Patient ID (e.g., P_00017)
    p_match = re.search(r"P[_-]?\d+", path_upper)
    patient_id = f"P_{p_match.group(0).replace('P', '').replace('_', '')}" if p_match else None
    
    # Side
    side = "L" if "LEFT" in path_upper else "R" if "RIGHT" in path_upper else None
    
    # View
    view = "CC" if "_CC" in path_upper or "/CC" in path_upper else "MLO" if "_MLO" in path_upper or "/MLO" in path_upper else None
    
    return pd.Series({'patient_id': patient_id, 'side': side, 'view': view})


def build_manifest(dicom_csv: str, case_csv: str, stage: str = "Mass-Test"):
    print("Loading CSVs...")
    dicom_df = pd.read_csv(dicom_csv, low_memory=False)
    case_df = pd.read_csv(case_csv, low_memory=False)

    # Strict filter for full mammograms and the specific stage
    is_full_mammo = dicom_df['SeriesDescription'].astype(str).str.strip().str.lower() == "cropped images"
    is_stage = dicom_df['PatientID'].astype(str).str.contains(stage, case=False, na=False)
    
    dicom_filtered = dicom_df[is_full_mammo & is_stage].copy()
    
    if dicom_filtered.empty:
        print(f"Warning: No full mammograms found for stage {stage}.")
        return pd.DataFrame()

    # Parse patient, side, and view directly from the image_path
    meta_df = dicom_filtered['PatientID'].apply(parse_dicom_path)
    dicom_filtered = pd.concat([dicom_filtered, meta_df], axis=1)
    print(meta_df.head())

    def extract_p(p):
        m = re.search(r"P[_-]?\d+", str(p).upper())
        return f"P_{m.group(0).replace('P', '').replace('_', '')}" if m else None

    case_df['patient_id'] = case_df['patient_id'].apply(extract_p)
    case_df['side'] = case_df['left or right breast'].apply(lambda x: "L" if "LEFT" in str(x).upper() else "R")
    case_df['view'] = case_df['image view'].apply(lambda x: "CC" if "CC" in str(x).upper() else "MLO")
    case_df['label'] = case_df['pathology'].apply(lambda x: 1 if "MALIGNANT" in str(x).upper() else 0)
    #print(case_df[['patient_id', 'side', 'view', 'label']].head())
    # Handle multiple masses per breast (if any is malignant, the whole breast view is labeled malignant)
    case_breast_level = case_df.groupby(['patient_id', 'side', 'view']).agg({
        'label': 'max'
    }).reset_index()
    
    case_breast_level['pathology'] = case_breast_level['label'].apply(lambda x: "MALIGNANT" if x == 1 else "BENIGN")

    print("Merging Datasets...")
    # Inner join the DICOM file paths with the breast-level pathology labels
    manifest_df = pd.merge(
        dicom_filtered[['patient_id', 'side', 'view', 'image_path']], 
        case_breast_level, 
        on=['patient_id', 'side', 'view'], 
        how='inner'
    )

    return manifest_df


if __name__ == "__main__":
    dicom_csv_path = "dicom_info.csv"
    case_csv_path = "mass_case_description_train_set.csv"
    base_dir = "dataset/mammo"
    stage = "Mass-Training"
    data = build_manifest(dicom_csv=dicom_csv_path, case_csv=case_csv_path, stage=stage)
    df = pd.DataFrame(data)
    print(f"Built manifest with {len(df)} rows.")
    print(df.loc[0])
    df.to_csv("./mass_train_manifest.csv", index=False)
