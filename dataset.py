# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 13:23:57 2025

Utilities to load data paths for:
  - Segmentation: image–mask pairs
  - Classification: image–label pairs

Supports your current JSON layout:
{
  "train": { "<filename>": ..., ... },
  "val":   { ... },              # optional
  "test":  { ... },              # optional
  "config": {
    "emidec_data_dir": "...",
    "imperial_data_dir": "...",
    // (optional) "classification_labels": { "<filename>": <int_label>, ... }
    // or provide labels externally via a callback
  }
}

Provides builders for:
  - TorchIO Subjects (3D seg)
  - MONAI dicts (2D/3D seg) and vanilla tuples (classification)

@author: Henry
"""

import os
import json
import torch
import torchio as tio
from typing import Dict, List, Tuple, Callable, Optional, Iterable, Any

# ---------------------------
# JSON I/O
# ---------------------------

def load_json_config(json_path: str) -> Dict:
    """Load your dataset_config_train_test.json."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def get_split_filenames(data: Dict, split: str = "train") -> List[str]:
    """
    Return the list of filenames for a given split. If split not present,
    raises a KeyError to make issues explicit.
    """
    if split not in data:
        raise KeyError(f"Split '{split}' not found in JSON. Available: {list(k for k in data.keys() if k != 'config')}")
    return list(data[split].keys())


# ---------------------------
# Path building (segmentation)
# ---------------------------

def build_segmentation_paths(
    data: Dict,
    split: str = "train",
) -> Tuple[List[str], List[str]]:
    """
    Build (images_list, masks_list) for segmentation from your JSON.
    Uses your current logic:
        - .nii.gz -> EMIDEC (emidec_train_images / emidec_train_labels)
        - .nii    -> Imperial (self_annotated_images / self_annotated_merged_labels)
    """
    emidec_dir = data["config"]["emidec_data_dir"]
    imperial_dir = data["config"]["imperial_data_dir"]

    images_emidec, masks_emidec = [], []
    images_imperial, masks_imperial = [], []

    filenames = get_split_filenames(data, split)
    for fname in filenames:
        if fname.endswith(".nii.gz"):
            img = os.path.join(emidec_dir, "emidec_train_images", fname)
            msk = os.path.join(emidec_dir, "emidec_train_labels", fname)
            images_emidec.append(img)
            masks_emidec.append(msk)
        elif fname.endswith(".nii"):
            img = os.path.join(imperial_dir, "self_annotated_images", fname)
            msk = os.path.join(imperial_dir, "self_annotated_merged_labels", fname)
            images_imperial.append(img)
            masks_imperial.append(msk)
        else:
            # Unknown extension: skip or raise. Here we raise to surface data issues.
            raise ValueError(f"Unrecognized file extension for '{fname}'. Expected .nii.gz or .nii")

    images = images_emidec + images_imperial
    masks  = masks_emidec  + masks_imperial
    return images, masks


# ---------------------------
# Path building (classification)
# ---------------------------

def build_classification_paths(
    data: Dict,
    split: str = "train",
    label_getter: Optional[Callable[[str, Dict], int]] = None,
) -> List[Tuple[str, int]]:
    """
    Build (image_path, class_id) pairs for classification.

    Label sources supported:
      1) If JSON has data["config"]["classification_labels"] as {filename: int_label},
         it will be used automatically (unless you pass a custom label_getter).
      2) Or pass a `label_getter(filename, data_dict) -> int` callback to compute labels
         (e.g., from filename patterns or another file).

    Returns:
      List of tuples: [(image_path, class_id), ...]
    """
    emidec_dir = data["config"]["emidec_data_dir"]
    imperial_dir = data["config"]["imperial_data_dir"]
    filenames = get_split_filenames(data, split)

    # Optional built-in mapping from JSON
    json_labels = data.get("config", {}).get("classification_labels", {})

    pairs: List[Tuple[str, int]] = []

    for fname in filenames:
        # Build image path (no mask for classification)
        if fname.endswith(".nii.gz"):
            img = os.path.join(emidec_dir, "emidec_train_images", fname)
        elif fname.endswith(".nii"):
            img = os.path.join(imperial_dir, "self_annotated_images", fname)
        else:
            raise ValueError(f"Unrecognized file extension for '{fname}'. Expected .nii.gz or .nii")

        # Resolve label
        if label_getter is not None:
            cls = int(label_getter(fname, data))
        else:
            if fname not in json_labels:
                raise KeyError(
                    f"No classification label for '{fname}'. "
                    "Provide label_getter or add to config.classification_labels."
                )
            cls = int(json_labels[fname])

        pairs.append((img, cls))

    return pairs


# ---------------------------
# Sanity checks
# ---------------------------

def check_files_exist(paths: Iterable[str], raise_on_missing: bool = True) -> List[str]:
    """
    Check which files exist. Returns list of missing files.
    """
    missing = [p for p in paths if not os.path.exists(p)]
    if missing and raise_on_missing:
        raise FileNotFoundError(f"Missing files ({len(missing)}):\n" + "\n".join(missing))
    return missing


# ---------------------------
# Builders for datasets
# ---------------------------

# 1) TorchIO Subjects (3D segmentation)
def make_torchio_subjects(images: List[str], masks: List[str]):
    """
    Return a list of TorchIO Subjects for 3D segmentation.
    Each subject has keys: 'image' (ScalarImage) and 'label' (LabelMap).
    """
    import torchio as tio
    assert len(images) == len(masks), "images and masks must have same length"
    subjects = []
    for img, msk in zip(images, masks):
        subjects.append(tio.Subject(
            image=tio.ScalarImage(img),
            label=tio.LabelMap(msk),
        ))
    return subjects

# ---------------------------
# Example label_getter (optional)
# ---------------------------

def example_label_getter_from_filename(fname: str, data: Dict) -> int:
    """
    Example callback for build_classification_paths:
    - Return 1 if filename contains 'scar', else 0.
    Replace with your actual logic or a CSV lookup.
    """
    name = fname.lower()
    return 1 if "scar" in name else 0

# ---------------------------
# Data Loader (segmentation)
# ---------------------------

def collate_tio_subjects(batch):
    """
    Collate a list of tio.Subject into a dict compatible with your trainer:
      batch["image"][tio.DATA] -> (B,C,H,W,D) float32
      batch["label"][tio.DATA] -> (B,1,H,W,D) uint8   (if present)
    Any extra fields are ignored (keeps batching simple & robust).
    """
    imgs = []
    lbls = []
    has_label = ("label" in batch[0]) and (batch[0]["label"] is not None)

    for s in batch:
        img = s["image"][tio.DATA]           # (C,H,W,D)
        # enforce dtypes
        img = img.to(torch.float32)
        imgs.append(img)

        if has_label:
            y = s["label"][tio.DATA]         # (1,H,W,D) or (C,H,W,D) if multi-class
            y = y.to(torch.uint8)            # keep binary labels compact (trainer calls .float() later)
            lbls.append(y)

    out = {"image": {tio.DATA: torch.stack(imgs, dim=0)}}
    if has_label:
        out["label"] = {tio.DATA: torch.stack(lbls, dim=0)}
    else:
        out["label"] = None
    return out

def make_loaders(
    train_indices: List[int],
    val_indices: List[int],
    base_data: Any,
    train_transform,
    val_transform,
    train_batch_size: int = 2,
    val_batch_size: int = 1,
    num_workers: int = 8,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train/val DataLoaders for either:
      • TorchIO: list[ t io.Subject ]  -> SubjectsDataset

    Args:
        base_data: list of Subjects OR list of dicts ({"image": path, "label": path})
    """
    # Slice the split
    train_data = [base_data[i] for i in train_indices]
    val_data   = [base_data[i] for i in val_indices]

    # TorchIO Subjects
    if len(train_data) > 0 and isinstance(train_data[0], tio.Subject):
        train_ds = tio.SubjectsDataset(train_data, transform=train_transform)
        val_ds   = tio.SubjectsDataset(val_data,   transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_tio_subjects,
        )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_tio_subjects,
        )
    return train_loader, val_loader