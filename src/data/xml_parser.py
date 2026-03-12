"""
INbreast XML annotation parser.

Extracts polygon ROIs from OsiriX plist XML files and converts them
to axis-aligned bounding boxes in pixel coordinates.

Lesion class mapping (simplified to 3 classes):
  1 = Mass         (Mass, Spiculated Region, Distortion)
  2 = Calcification (Calcification cluster, Cluster)
  3 = Asymmetry
"""

import os
import plistlib
import numpy as np

# -----------------------------------------------------------------------
# Class mapping — variants / typos present in the dataset are normalised
# -----------------------------------------------------------------------
_NAME_TO_CLASS = {
    "mass":               1,
    "spiculated region":  1,
    "espiculated region": 1,
    "spiculated region":  1,
    "distortion":         1,
    "calcification":      2,
    "calcifications":     2,
    "cluster":            2,
    "asymmetry":          3,
    "assymetry":          3,
}

CLASS_NAMES = ["__background__", "Mass", "Calcification", "Asymmetry"]
NUM_CLASSES  = len(CLASS_NAMES)   # 4  (including background)


def _parse_point(s: str) -> tuple[float, float]:
    """Parse '(x, y, z)' → (x, y)."""
    s = s.strip("() ")
    parts = s.split(",")
    return float(parts[0]), float(parts[1])


def parse_xml(xml_path: str) -> list[dict]:
    """
    Parse one INbreast XML annotation file.

    Returns a list of dicts, one per valid polygon ROI:
        {
          "name":  str,          # original ROI name
          "class": int,          # 1=Mass, 2=Calcification, 3=Asymmetry
          "box":   [x1,y1,x2,y2] # axis-aligned bounding box (pixel coords)
          "poly":  np.ndarray    # polygon points, shape (N, 2), xy
        }

    Single-point annotations (individual calcification dots) are skipped
    because they cannot form a meaningful bounding box for detection.
    """
    with open(xml_path, "rb") as f:
        data = plistlib.load(f)

    images = data.get("Images", [])
    if not images:
        return []

    rois_raw = images[0].get("ROIs", [])
    results = []

    for roi in rois_raw:
        n_pts = roi.get("NumberOfPoints", 0)
        if n_pts < 2:          # skip single-point markers
            continue

        name = roi.get("Name", "").strip()
        cls  = _NAME_TO_CLASS.get(name.lower())
        if cls is None:        # unknown annotation type — skip
            continue

        points_raw = roi.get("Point_px", [])
        if len(points_raw) < 2:
            continue

        pts = np.array([_parse_point(p) for p in points_raw], dtype=np.float32)  # (N, 2)

        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()

        # Degenerate box guard
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue

        results.append({
            "name":  name,
            "class": cls,
            "box":   [x1, y1, x2, y2],
            "poly":  pts,
        })

    return results


def build_annotation_map(xml_dir: str) -> dict[str, list[dict]]:
    """
    Build a dict mapping file_name (stem without extension) → list of ROI dicts.

    Example key: "22678622"  (matches INbreast.csv File Name column)
    """
    ann_map = {}
    for fname in os.listdir(xml_dir):
        if not fname.endswith(".xml"):
            continue
        stem = os.path.splitext(fname)[0]   # "22678622"
        xml_path = os.path.join(xml_dir, fname)
        rois = parse_xml(xml_path)
        ann_map[stem] = rois                # may be empty list
    return ann_map
