"""
Patient-aware train/val/test split for INbreast.

Since Patient IDs are anonymized in the public release, we reconstruct
patient groups using two observations from the dataset:
  1. All images from the same patient share the same acquisition date.
  2. Within a date, images from the same patient have consecutive file-name
     IDs with small gaps (~19-28). A gap > 200 in file-name IDs indicates
     a new patient.

Splitting at the patient level (rather than the image level) prevents
the same patient's CC and MLO views from appearing in both train and val,
which would inflate validation metrics.
"""

import csv
import random
from collections import defaultdict

import numpy as np


def _build_patient_groups(csv_path: str) -> list[list[str]]:
    """Return a list of patient groups, each a list of file_name strings."""
    date_to_files: dict[str, list[str]] = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            file_name = row["File Name"].strip()
            date = row["Acquisition date"].strip()
            if file_name and date:
                date_to_files[date].append(file_name)

    GAP_THRESHOLD = 200
    groups: list[list[str]] = []

    for files in date_to_files.values():
        files_sorted = sorted(files, key=lambda x: int(x))
        current: list[str] = [files_sorted[0]]

        for i in range(1, len(files_sorted)):
            if int(files_sorted[i]) - int(files_sorted[i - 1]) < GAP_THRESHOLD:
                current.append(files_sorted[i])
            else:
                groups.append(current)
                current = [files_sorted[i]]
        groups.append(current)

    return groups


def patient_aware_split(
    csv_path: str,
    file_names: list[str],
    val_fraction: float,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> tuple[list[int], ...]:
    """
    Split dataset indices so that all images from the same patient end up
    in the same partition.

    Args:
        csv_path:      Path to INbreast.csv
        file_names:    file_name string for each dataset sample (in order)
        val_fraction:  fraction of patients for validation
        test_fraction: fraction of patients for test (0 = no test set)
        seed:          random seed

    Returns:
        (idx_train, idx_val) or (idx_train, idx_val, idx_test)
    """
    all_groups = _build_patient_groups(csv_path)

    file_name_set = set(file_names)
    fn_to_idx: dict[str, int] = {fn: i for i, fn in enumerate(file_names)}

    # Keep only patients that have at least one image in this dataset
    relevant: list[list[str]] = [
        [fn for fn in g if fn in file_name_set]
        for g in all_groups
    ]
    relevant = [g for g in relevant if g]

    rng = random.Random(seed)
    rng.shuffle(relevant)

    n = len(relevant)
    n_test = int(np.round(test_fraction * n))
    n_val  = int(np.round(val_fraction * n))

    train_groups = relevant[: n - n_val - n_test]
    val_groups   = relevant[n - n_val - n_test : n - n_test]
    test_groups  = relevant[n - n_test :]

    def to_indices(gs: list[list[str]]) -> list[int]:
        return sorted(fn_to_idx[fn] for g in gs for fn in g)

    idx_train = to_indices(train_groups)
    idx_val   = to_indices(val_groups)

    if test_fraction > 0:
        return idx_train, idx_val, to_indices(test_groups)
    return idx_train, idx_val
