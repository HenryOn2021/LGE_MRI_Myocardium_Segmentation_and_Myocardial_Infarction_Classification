# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 20:40:36 2025

2D Classification (DenseNet/ResNet) with patient-wise CV + Captum CAMs + Dice vs. infarct masks
+ Ensemble figures across 5 folds
-----------------------------------------------------------------------------------------------

What this script does
---------------------
1) Converts the pipeline to **2D slice classification** while performing **patient-wise splits**
   (no slice leakage across train/val/test). EMIDEC (.nii.gz) and Imperial (.nii) are supported.

2) Per dataset: **stratified 80/20 hold-out on PATIENTS**, then combine the 80% portions (EMIDEC+Imperial)
   into one pool for development. On this combined pool we run **patient-wise stratified 5-fold CV**.
   Training/validation are on slices expanded from the patients in each fold.

3) Model zoo: **DenseNet (2D)** via MONAI and **ResNet (2D)** via torchvision.
   User chooses `--model_name` among: 
   ["densenet121","densenet169","densenet201","densenet264",
    "resnet18","resnet34","resnet50","resnet101","resnet152"]

4) Inference: evaluates each fold model **separately** (NO soft voting) on slice-level;
   reports **mean ± std** metrics across folds for: accuracy, specificity, sensitivity,
   weighted precision, weighted F1, ROC AUC. Also saves per-fold ROC/PR curves and
   their **ensemble (mean ± SD)** curves and confusion matrix.

5) **Captum**: Generates **GradCAM** and **GuidedGradCAM** (from the last Conv2d layer) for:
   (a) **ground-truth positive** slices and (b) **predicted positive** slices.
   For each fold we save CAMs under `cams_fold{k}/{cams_gtpos|cams_predpos}/{gradcam|guided_gradcam}/`.
   We also build **ensemble CAMs** (pixelwise mean & std across folds) under `cams_ensemble/`.

6) **Dice vs infarct masks**: Provide paired infarct mask directories for EMIDEC & Imperial
   via `--infarct_dir_emidec` and `--infarct_dir_imperial`. During inference we compute the
   **Dice score** between the **thresholded mean GradCAM** (threshold `--cam_thresh`) and the
   corresponding infarct mask slice. We save the **Top-10 Dice** figures (two subplots):
   LEFT = image + infarct mask overlay (alpha=0.6), RIGHT = thresholded GradCAM heatmap with dice.

Usage (example)
---------------
python lge_2d_classification_with_cams.py \
  --images_dir_emidec "/path/to/emidec_images" \
  --images_dir_imperial "/path/to/imperial_images" \
  --labels_csv_emidec "/path/to/emidec_labels.csv" \
  --labels_csv_imperial "/path/to/imperial_labels.csv" \
  --infarct_dir_emidec "/path/to/emidec_infarct_masks" \
  --infarct_dir_imperial "/path/to/imperial_infarct_masks" \
  --out_dir "./outputs_lge_2d" \
  --model_name "resnet50" \
  --spatial_size 224 224 \
  --batch_size 32 \
  --max_epochs 100 \
  --patience 20 \
  --cam_thresh 0.5 \
  --enable_cams 1 \
  --mode full

@author: Henry
"""

import os, glob
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re
import heapq
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

from monai.config import print_config
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    Resize,
    NormalizeIntensity,
    RandFlip,
    RandRotate90,
    RandBiasField,
    RandAdjustContrast,
    RandGaussianNoise,
    EnsureType,
    LoadImage,
)
from monai.networks.nets import DenseNet121, DenseNet169, DenseNet201, DenseNet264
from monai.networks.nets import resnet18, resnet34, resnet50, resnet101, resnet152
from monai.utils import set_determinism

# Captum for CAMs
from captum.attr import LayerGradCam, GuidedGradCam

# QC
import random, sys
from collections import defaultdict


# -----------------------------
# Utility & Reproducibility
# -----------------------------

def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Data IO & patient tokenisation
# -----------------------------

CANDIDATE_NAME_COLS = ["filename", "file", "case", "id", "name", "image_name"]

def _detect_name_col(df: pd.DataFrame) -> str:
    # Return the first non-'label' column that looks like a filename/case id.
    for c in df.columns:
        lc = str(c).lower()
        if lc == "label":
            continue
        if lc in CANDIDATE_NAME_COLS:
            return c
    # Fallback: take the first non-label column
    for c in df.columns:
        if str(c).lower() != "label":
            return c
    raise ValueError("Could not find a filename/case column in the labels CSV.")


def _basename_no_ext(p: Path) -> str:
    name = p.name
    lname = name.lower()
    if lname.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if lname.endswith(".nii"):
        return name[: -len(".nii")]
    return p.stem


def _norm_token(x: str) -> str:
    """Lowercase, strip extensions (.nii/.nii.gz), and common suffixes like _seg/_mask."""
    x = str(x).strip().lower()
    x = x.replace(".nii.gz", "").replace(".nii", "")
    x = re.sub(r"(_seg|_mask)$", "", x)
    return x


def _case_token(x: str) -> str:
    """
    Patient/case token: normalize, then strip a trailing '_<digits>' (slice index).
    Examples:
      'AA42_20190730_Late_Gad_LVSA_27.nii'  -> 'aa42_20190730_late_gad_lvsa'
      'Case_N006_18.nii.gz'                 -> 'case_n006'
    """
    t = _norm_token(x)
    t = re.sub(r"_(\d+)$", "", t)
    return t


def link_images_to_labels(
    images_dir: Path,
    labels_csv: Path,
    pattern: str,
    strict: bool = True,
) -> Tuple[List[str], List[int], List[str], List[str]]:
    """
    Build (image_path, label, patient_id) by matching the case/filename from the CSV
    to the image filename (extension-agnostic, robust).

    Returns:
        image_paths, labels, patient_ids, unmatched_image_paths
    """
    images = sorted(list(images_dir.rglob(pattern)))
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in {images_dir} matching {pattern}.")
    df = pd.read_csv(labels_csv)
    # Drop index-like unnamed columns
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:")]
    if "label" not in [c.lower() for c in df.columns]:
        raise ValueError(f'"label" column not found in {labels_csv}')
    # Resolve columns
    label_col = [c for c in df.columns if str(c).lower() == "label"][0]
    name_col = _detect_name_col(df)

    # Normalise CSV names
    df["_name_norm"] = df[name_col].apply(_norm_token)
    name_to_label: Dict[str, int] = {str(row["_name_norm"]): int(row[label_col]) for _, row in df.iterrows()}

    matched_paths: List[str] = []
    matched_labels: List[int] = []
    patient_ids: List[str] = []

    # Strategy: exact on robust token, else two-way substring (longest)
    for img_path in images:
        base_full = _norm_token(_basename_no_ext(img_path))  # KEEP slice index for label matching
        label: Optional[int] = None
        if base_full in name_to_label:
            label = name_to_label[base_full]
            pid = _case_token(base_full)  # patient id: strip trailing _<digits>
        else:
            cands = []
            for nm, lb in name_to_label.items():
                if nm and (nm in base_full or base_full in nm):
                    cands.append((nm, lb))
            if cands:
                nm, lb = max(cands, key=lambda t: len(t[0]))
                label = lb
                pid = _case_token(nm)
        if label is not None:
            matched_paths.append(str(img_path))
            matched_labels.append(label)
            patient_ids.append(pid)

    # Find unmatched images (no label in CSV)
    unmatched_images = [str(p) for p in images if str(p) not in set(matched_paths)]

    if strict and len(unmatched_images) > 0:
        examples = "\n".join(f"  - {Path(u).name}" for u in unmatched_images[:20])
        raise RuntimeError(
            f"{len(unmatched_images)} image(s) in {images_dir} have no label match in {labels_csv}.\n"
            f"Examples:\n{examples}\n"
            "Tips: ensure the CSV filename column is detected correctly "
            "(e.g., 'image_name'), and that tokens align after normalisation "
            "(extensions removed; '_seg'/'_mask' stripped)."
        )
    elif len(unmatched_images) > 0:
        print(f"[WARN] {len(unmatched_images)} image(s) in {images_dir} had no label match. Proceeding because strict=False.")

    return matched_paths, matched_labels, patient_ids, unmatched_images


# -----------------------------
# 2D Dataset (slice-level, but patient-wise split)
# -----------------------------

class SliceList:
    """
    Helper to expand a list of patient volumes into a list of (path, slice_idx).
    For 2D-per-file datasets, each file contributes a single slice (idx=0).
    For 3D volumes, contributes D slices (idx=0..D-1).
    """
    def __init__(self, paths: List[str]):
        self.paths = paths
        self._slices: List[Tuple[str, int]] = []
        self._built = False

    def build(self) -> None:
        if self._built:
            return
        loader = LoadImage(image_only=True)
        for p in self.paths:
            vol = loader(p)  # [H,W]
            if vol.ndim == 2:
                d = 1
            else:
                d = int(vol.shape[-1])
            for s in range(d):
                self._slices.append((p, s))
        self._built = True

    def __len__(self) -> int:
        if not self._built:
            self.build()
        return len(self._slices)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        if not self._built:
            self.build()
        return self._slices[idx]


class LGE2DDataset(Dataset):
    """
    2D slice classification dataset. Slices along the last axis (D).
    Patient-wise split is enforced externally; labels are looked up per-slice **by path**.
    """
    def __init__(self, slice_list: SliceList, label_map: Dict[str, int], transforms: Compose,
                 path_to_pid: Optional[Dict[str, str]] = None):
        self.slice_list = slice_list
        self.label_map = label_map          # NOW: path -> slice_label (0/1)
        self.transforms = transforms
        self.path_to_pid = path_to_pid      # still used for bookkeeping, CAM grouping, etc.
        self.loader = LoadImage(image_only=True)

    def __len__(self):
        return len(self.slice_list)

    def __getitem__(self, idx: int):
        path, sidx = self.slice_list[idx]
        vol = self.loader(path)  # [H,W] or [H,W,D]
        img2d = vol[..., sidx] if getattr(vol, "ndim", 2) == 3 else vol

        img = self.transforms(img2d.astype(np.float32))  # -> [1,H,W]

        # label by PATH (true slice/file label from CSV)
        spath = str(path)
        if spath not in self.label_map:
            raise KeyError(f"Missing slice label for path: {spath}")
        lb = int(self.label_map[spath])

        return img, lb, path, int(sidx)


def make_2d_transforms(spatial_size_hw: Tuple[int, int]) -> Tuple[Compose, Compose]:
    """
    Training and validation/test transforms for 2D.
    """
    train_tf = Compose([
        EnsureChannelFirst(channel_dim="no_channel"),
        Resize(spatial_size=spatial_size_hw),
        NormalizeIntensity(nonzero=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandFlip(spatial_axis=1, prob=0.5),
        RandRotate90(prob=0.3, spatial_axes=(0,1)),
        RandBiasField(prob = 0.2, coeff_range = (0, 0.1)),
        RandAdjustContrast(prob=0.3, gamma=(0.7, 1.3)),
        RandGaussianNoise(prob=0.15, mean=0.0, std=0.05),
        EnsureType(),
    ])
    eval_tf = Compose([
        EnsureChannelFirst(channel_dim="no_channel"),
        Resize(spatial_size=spatial_size_hw),
        NormalizeIntensity(nonzero=True),
        EnsureType(),
    ])
    return train_tf, eval_tf


# -----------------------------
# Models (2D DenseNet + 2D ResNet)
# -----------------------------

def get_last_conv2d(module: nn.Module) -> nn.Module:
    for m in reversed(list(module.modules())):
        if isinstance(m, nn.Conv2d):
            return m
    raise RuntimeError("No Conv2d layer found for CAM.")


def get_model_2d(model_name: str, out_channels: int = 2) -> Tuple[nn.Module, str]:
    """
    Returns a 2D model (MONAI) with 1 input channel and `out_channels` classes.
    `model_name` must be one of:
      DenseNet:  densenet121, densenet169, densenet201, densenet264
      ResNet:    resnet18, resnet34, resnet50, resnet101, resnet152
    The second return value "last_conv_auto" is a tag; CAM code auto-finds the last Conv2d.
    """
    mn = model_name.lower()

    densenet_builders = {
        "densenet121": lambda: DenseNet121(spatial_dims=2, in_channels=1, out_channels=out_channels),
        "densenet169": lambda: DenseNet169(spatial_dims=2, in_channels=1, out_channels=out_channels),
        "densenet201": lambda: DenseNet201(spatial_dims=2, in_channels=1, out_channels=out_channels),
        "densenet264": lambda: DenseNet264(spatial_dims=2, in_channels=1, out_channels=out_channels),
    }

    resnet_builders = {
        "resnet18":  lambda: resnet18(spatial_dims=2, n_input_channels=1, num_classes=out_channels),
        "resnet34":  lambda: resnet34(spatial_dims=2, n_input_channels=1, num_classes=out_channels),
        "resnet50":  lambda: resnet50(spatial_dims=2, n_input_channels=1, num_classes=out_channels),
        "resnet101": lambda: resnet101(spatial_dims=2, n_input_channels=1, num_classes=out_channels),
        "resnet152": lambda: resnet152(spatial_dims=2, n_input_channels=1, num_classes=out_channels),
    }

    if mn in densenet_builders:
        return densenet_builders[mn](), "last_conv_auto"
    elif mn in resnet_builders:
        return resnet_builders[mn](), "last_conv_auto"
    else:
        valid = list(densenet_builders.keys()) + list(resnet_builders.keys())
        raise ValueError(f"Unknown model_name={model_name}. Choose one of: {', '.join(valid)}")


# -----------------------------
# Training / Validation (per fold) — 2D
# -----------------------------

def train_one_fold_2d(
    fold_idx: int,
    train_slice_list: SliceList,
    val_slice_list: SliceList,
    train_tf: Compose,
    eval_tf: Compose,
    args,
    device: torch.device,
    out_dir: Path,
    path_to_pid: Dict[str, str],
    path_label_map: Dict[str, int]
) -> Path:
    """
    Train one fold (2D), save best checkpoint (by val AUC), and return its path.
    """
    fold_dir = out_dir / f"fold_{fold_idx}"
    ensure_out_dir(fold_dir)
    ckpt_path = fold_dir / "best_model.pth"
    log_csv = fold_dir / "train_log.csv"

    ds_train = LGE2DDataset(train_slice_list, label_map=path_label_map, transforms=train_tf, path_to_pid=path_to_pid)
    ds_val   = LGE2DDataset(val_slice_list,   label_map=path_label_map, transforms=eval_tf,  path_to_pid=path_to_pid)

    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0)
    )
    dl_val = DataLoader(
        ds_val, batch_size=max(1, args.batch_size//2), shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0)
    )

    model, _ = get_model_2d(args.model_name, out_channels=2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_auc = -np.inf
    best_state = None
    no_improve = 0

    log_rows = []
    print(f"[Fold {fold_idx}] Train slices={len(ds_train)} | Val slices={len(ds_val)}")

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, labels, _, _ in dl_train:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * imgs.size(0)

        epoch_train_loss = running_loss / len(ds_train)

        # Validation
        model.eval()
        all_probs = []
        all_true  = []
        with torch.no_grad():
            for imgs, labels, _, _ in dl_val:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)[:, 1]  # p(class=1)
                all_probs.append(probs.detach().cpu().numpy())
                all_true.append(labels.detach().cpu().numpy())
        y_true = np.concatenate(all_true) if len(all_true) else np.array([])
        y_prob = np.concatenate(all_probs) if len(all_probs) else np.array([])

        if len(y_true) > 0:
            try:
                val_auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                val_auc = np.nan

            y_pred = (y_prob >= 0.5).astype(int)
            val_acc = accuracy_score(y_true, y_pred)
            val_f1w = f1_score(y_true, y_pred, average="weighted")
            val_prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            val_rec_w  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        else:
            val_auc = np.nan
            val_acc = val_f1w = val_prec_w = val_rec_w = np.nan

        log_rows.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_auc": float(val_auc) if not np.isnan(val_auc) else None,
            "val_acc": val_acc,
            "val_f1_weighted": val_f1w,
            "val_precision_weighted": val_prec_w,
            "val_recall_weighted": val_rec_w,
            "lr": optimizer.param_groups[0]["lr"],
        })

        score = val_auc if not np.isnan(val_auc) else (val_acc if not np.isnan(val_acc) else -np.inf)
        improved = score > best_auc
        if improved:
            best_auc = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
            torch.save(best_state, ckpt_path)
        else:
            no_improve += 1

        print(f"[Fold {fold_idx:02d}][Epoch {epoch:03d}] "
              f"loss={epoch_train_loss:.4f} | AUC={val_auc:.4f} | "
              f"ACC={val_acc:.4f} | F1w={val_f1w:.4f} | "
              f"no_improve={no_improve}/{args.patience}")

        if no_improve >= args.patience:
            print(f"[Fold {fold_idx}] Early stopping at epoch {epoch}.")
            break

    pd.DataFrame(log_rows).to_csv(log_csv, index=False)

    if not ckpt_path.exists() and best_state is not None:
        torch.save(best_state, ckpt_path)

    return ckpt_path

# -----------------------------
# Metrics & plotting (2D)
# -----------------------------

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    if y_true.size == 0:
        return {k: np.nan for k in ["accuracy","specificity","sensitivity",
                                    "precision_weighted","f1_weighted","roc_auc"]}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "specificity": specificity,
        "sensitivity": sensitivity,
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = np.nan
    return metrics


from matplotlib.ticker import PercentFormatter

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_png: Path,
    out_npy: Optional[Path] = None,
) -> None:
    """
    Plot a 2x2 confusion matrix using matplotlib only.
    Each cell displays:
        "<count>\n<percent of TOTAL>"
    Colors encode percent-of-total (0..1). Saves the percent matrix to out_npy,
    and (if out_npy is provided) also saves a counts matrix next to it with the
    suffix "_counts.npy" for ensemble use.
    """
    if y_true.size == 0:
        return

    # binarize predictions at 0.5
    y_pred = (y_prob >= 0.5).astype(int)

    # raw counts (fixed order [[TN, FP],[FN, TP]])
    cf_counts = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int)
    total = int(cf_counts.sum())
    denom = max(total, 1)
    cf_pct = cf_counts.astype(float) / denom  # 0..1, % of TOTAL

    # --- save numeric arrays ---
    if out_npy is not None:
        np.save(out_npy, cf_pct)  # percent-of-total
        # also save counts next to it (for ensemble mean±SD counts)
        counts_path = Path(str(out_npy).replace(".npy", "_counts.npy"))
        np.save(counts_path, cf_counts)

    # --- draw ---
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cf_pct, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)

    # axis labels & ticks
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title("Confusion Matrix")

    # annotations: count + percent in each cell
    for i in range(2):
        for j in range(2):
            count = cf_counts[i, j]
            pct   = cf_pct[i, j] * 100.0
            # choose contrasting text color for readability
            color = "white" if cf_pct[i, j] > 0.5 else "black"
            ax.text(j, i, f"{count}\n{pct:.2f}%", ha="center", va="center",
                    fontsize=12, color=color, fontweight="bold")

    # colorbar with percent formatter
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    cbar.ax.set_ylabel("% of total", rotation=90)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_png: Path) -> None:
    if y_true.size == 0:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    sns.set_theme(context="talk", style="whitegrid")
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, drawstyle="steps-post", label=f"ROC (AUC={auc:.3f})", linewidth=2.0)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, out_png: Path) -> None:
    if y_true.size == 0:
        return
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = np.trapz(precision[::-1], recall[::-1])
    sns.set_theme(context="talk", style="whitegrid")
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, drawstyle="steps-post", label=f"PR Curve (AUC≈{ap:.3f})", linewidth=2.0)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    

def remove_old_cm_files(base_out: Path, model_name: str):
    """
    Remove stale confusion-matrix numpy files from previous runs
    so the ensemble step only sees fresh percent-of-total CMs.
    """
    patterns = [
        str(base_out / model_name / "test_*" / "figures" / "cm_*fold*.npy"),
        str(base_out / model_name / "ensemble" / "figures" / "confusion_matrix_mean.png"),
    ]
    removed = 0
    for pat in patterns:
        for f in glob.glob(pat):
            try:
                os.remove(f)
                removed += 1
                print(f"[CLEAN] removed {f}")
            except FileNotFoundError:
                pass
    if removed == 0:
        print("[CLEAN] no old CM files to remove")
        

# -----------------------------
# Captum CAMs & Dice vs. infarct masks
# -----------------------------

def _last_conv_layer(module: nn.Module) -> nn.Module:
    return get_last_conv2d(module)


def _cam_to_numpy(cam: torch.Tensor) -> np.ndarray:
    cam = cam.detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    if cam.ndim == 3:  # [C,H,W]? reduce channel
        cam = cam.mean(axis=0)
    return cam


def dice_coeff(bin_a: np.ndarray, bin_b: np.ndarray, eps: float = 1e-6) -> float:
    inter = float((bin_a & bin_b).sum())
    s = float(bin_a.sum() + bin_b.sum())
    return (2.0 * inter) / (s + eps) if s > 0 else 0.0


def _find_mask_by_same_filename(infarct_root: Optional[Path], img_path: str) -> Optional[Path]:
    """
    Pair image and mask by identical base filename (ignoring extension).
    E.g., 'AA42_xxx_22.nii' ↔ 'AA42_xxx_22.nii.gz'
    """
    if infarct_root is None or not infarct_root.exists():
        return None
    img_base = _norm_token(_basename_no_ext(Path(img_path)))
    for mpath in infarct_root.rglob("*.nii*"):
        if _norm_token(_basename_no_ext(mpath)) == img_base:
            return mpath
    return None


def generate_and_save_cams(
    model: nn.Module,
    imgs: torch.Tensor,            # [N,1,H,W]
    y_true: np.ndarray,            # [N]
    y_prob: np.ndarray,            # [N] probs for class=1
    paths: List[str],
    sidxs: List[int],
    out_dir: Path,
    device: torch.device,
    enable_cams: bool,
    infarct_dir_map: Dict[str, Optional[Path]],  # dataset_name -> dir or None
    cam_thresh: float,
    spatial_size_hw: Tuple[int, int],
    topk: int = 10
) -> None:
    """
    Generate CAMs for GT-positive and Pred-positive slices.
    Save *only* Top-K slices (by Dice) as (1) PNG figure and (2) .npy arrays (GradCAM & GuidedGradCAM).
    """
    if not enable_cams:
        return

    model.eval()
    layer = _last_conv_layer(model)
    gradcam = LayerGradCam(model, layer)
    guided  = GuidedGradCam(model, layer)

    cams_root = out_dir
    gt_dir    = cams_root / "cams_gtpos"
    pr_dir    = cams_root / "cams_predpos"
    logs_dir  = cams_root / "logs"
    for d in [gt_dir / "gradcam", gt_dir / "guided_gradcam",
              pr_dir / "gradcam", pr_dir / "guided_gradcam", logs_dir]:
        ensure_out_dir(d)

    missing_masks = []
    imgs = imgs.to(device)

    # Identify GT-positive and Pred-positive indices
    y_pred = (y_prob >= 0.5).astype(int)
    idx_gtpos  = np.where(y_true == 1)[0].tolist()
    idx_prpos  = np.where(y_pred == 1)[0].tolist()

    dice_candidates = []

    def _norm01(a: np.ndarray) -> np.ndarray:
        a = a - a.min()
        rng = a.max() - a.min()
        return (a / rng) if rng > 1e-8 else a

    # process a single slice index into a candidate buffer (no saving yet)
    def _process(idx: int, subdir: Path):
        x = imgs[idx:idx+1]  # [1,1,H,W]
        path = paths[idx]
        sidx = sidxs[idx]
        base_full = _norm_token(_basename_no_ext(Path(path)))  # keep slice suffix
        img2d = imgs[idx, 0].detach().cpu().numpy()

        # forward once to set gradients’ graph
        model.zero_grad(set_to_none=True)
        _ = model(x)
        target = 1  # positive class

        # GradCAM
        gcam = gradcam.attribute(x, target=target)
        gcam = torch.nn.functional.interpolate(gcam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        gcam_np = _norm01(_cam_to_numpy(gcam.squeeze(0)))

        # GuidedGradCAM
        ggcam = guided.attribute(x, target=target)
        ggcam = torch.nn.functional.interpolate(ggcam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        ggcam_np = _norm01(_cam_to_numpy(ggcam.squeeze(0)))

        # infarct mask (pair by identical filename, any extension)
        dataset_name = "emidec" if str(path).lower().endswith(".nii.gz") else "imperial"
        infarct_root = infarct_dir_map.get(dataset_name, None)
        mask_path = _find_mask_by_same_filename(infarct_root, path)
        if infarct_root is not None and mask_path is None:
            missing_masks.append(base_full)
            return  # cannot compute Dice, skip candidate

        if mask_path is None:
            return

        vol = LoadImage(image_only=True)(str(mask_path))
        if vol.ndim == 3:
            mslice = vol[..., sidx] if sidx < vol.shape[-1] else vol[..., vol.shape[-1]//2]
        else:
            mslice = vol
        from skimage.transform import resize as imresize
        mslice_r = imresize(mslice.astype(float), spatial_size_hw, order=0, preserve_range=True, anti_aliasing=False)
        bin_mask = (mslice_r > 0.5).astype(np.uint8)

        # strict binary map for Dice (visualization remains RAW heatmap)
        gbin = (gcam_np > cam_thresh).astype(np.uint8)
        d = dice_coeff(bin_mask.astype(bool), gbin.astype(bool))

        # where arrays would be saved if selected
        g_save = subdir / "gradcam" / f"{base_full}.npy"
        gg_save = subdir / "guided_gradcam" / f"{base_full}.npy"

        dice_candidates.append({
            "dice": float(d),
            "base": base_full,
            "img2d": img2d,
            "mask": bin_mask.astype(np.uint8),
            "gcam": gcam_np.astype(np.float32),         # RAW [0..1]
            "ggcam": ggcam_np.astype(np.float32),       # RAW [0..1]
            "g_path": g_save,
            "gg_path": gg_save,
        })

    # Build candidates for both GT+ and Pred+
    for idx in idx_gtpos:
        _process(idx, gt_dir)
    for idx in idx_prpos:
        _process(idx, pr_dir)

    if missing_masks:
        with open(logs_dir / "missing_infarct_masks.txt", "w") as f:
            for b in sorted(set(missing_masks)):
                f.write(b + "\n")

    # Save only Top-K (arrays + figure)
    K = max(1, int(topk))
    top = sorted(dice_candidates, key=lambda x: x["dice"], reverse=True)[:K]

    fig_dir = cams_root / "dice_top10"
    ensure_out_dir(fig_dir)

    for ent in top:
        base_full = ent["base"]
        img2d     = ent["img2d"]
        bin_mask  = ent["mask"]
        gcam_np   = ent["gcam"]
        ggcam_np  = ent["ggcam"]

        # 1) arrays (.npy)
        ensure_out_dir(ent["g_path"].parent)
        ensure_out_dir(ent["gg_path"].parent)
        np.save(ent["g_path"],  gcam_np)
        np.save(ent["gg_path"], ggcam_np)

        # 2) figure (RAW heatmap; Dice computed from binary map only)
        d = ent["dice"]
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img2d, cmap="gray"); ax1.imshow(bin_mask, alpha=0.3, cmap="Reds")
        ax1.set_title("Image + GT mask"); ax1.axis("off")
        ax2 = fig.add_subplot(1, 2, 2)
        im = ax2.imshow(gcam_np, cmap="rainbow", vmin=0.0, vmax=1.0)
        ax2.set_title("Grad-CAM")
        # Legend in top-right with Dice value (use an empty handle so only the text shows)
        legend_lbl = f"Dice = {d:.3f}"
        ax2.plot([], [], ' ', label=legend_lbl)
        leg = ax2.legend(loc='upper right', frameon=True)
        for lh in leg.legendHandles:
            lh.set_visible(False)
        ax2.axis("off"); #cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        #cbar.set_label("Intensity", rotation=90)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{base_full}_dice{d:.3f}.png", dpi=200)
        plt.close(fig)


# -----------------------------
# Ensemble CAMs & Top-10 Dice (mean across folds)
# -----------------------------

def build_ensemble_cams_and_top10(
    out_dir: Path,
    imgs: torch.Tensor,                  # [N,1,H,W] CPU
    paths: List[str],
    sidxs: List[int],
    infarct_dir_map: Dict[str, Optional[Path]],
    cam_thresh: float,
    spatial_size_hw: Tuple[int, int],
    topk: int = 10
) -> None:
    """
    Aggregate per-fold CAM .npy into mean/std under cams_ensemble/, compute Dice using mean GradCAM,
    and save arrays + PNGs *only for Top-K* slices by Dice.
    """
    cams_ensemble = out_dir / "cams_ensemble"
    for sub in ["cams_gtpos/gradcam", "cams_gtpos/guided_gradcam",
                "cams_predpos/gradcam", "cams_predpos/guided_gradcam",
                "dice_top10", "logs"]:
        ensure_out_dir(cams_ensemble / sub)

    # Map key -> index to recover image & slice
    idx_map = {}
    for i, (p, s) in enumerate(zip(paths, sidxs)):
        base_full = _norm_token(_basename_no_ext(Path(p)))
        idx_map[base_full] = i

    # Load per-fold arrays (note: now only Top-K per fold exist)
    groups: Dict[Tuple[str, str], Dict[str, List[np.ndarray]]] = {}
    fold_dirs = sorted(out_dir.glob("cams_fold*"))
    for fd in fold_dirs:
        for split in ["cams_gtpos", "cams_predpos"]:
            for kind in ["gradcam", "guided_gradcam"]:
                src = fd / split / kind
                if not src.exists():
                    continue
                for f in src.glob("*.npy"):
                    key = f.stem  # base_full
                    arr = np.load(f)
                    groups.setdefault((split, kind), {}).setdefault(key, []).append(arr)

    # Compute means/stds but DO NOT SAVE yet; we select Top-K first
    agg_store: Dict[Tuple[str, str], Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for (split, kind), dct in groups.items():
        for key, stacks in dct.items():
            arr = np.stack(stacks, axis=0)  # [F,H,W]
            mean_cam = arr.mean(axis=0)
            std_cam  = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean_cam)
            agg_store.setdefault((split, kind), {})[key] = (mean_cam.astype(np.float32), std_cam.astype(np.float32))

    # Build candidates (Dice computed from mean GradCAM only)
    ensemble_candidates = []
    missing_masks = []
    for split in ["cams_gtpos", "cams_predpos"]:
        # only keys common to gradcam means
        keys = list(agg_store.get((split, "gradcam"), {}).keys())
        for key in keys:
            if key not in idx_map:
                continue
            i = idx_map[key]
            img2d = imgs[i, 0].numpy()
            path  = paths[i]
            sidx  = sidxs[i]

            dataset_name = "emidec" if str(path).lower().endswith(".nii.gz") else "imperial"
            infarct_root = infarct_dir_map.get(dataset_name, None)
            mask_path = _find_mask_by_same_filename(infarct_root, path)
            if infarct_root is not None and mask_path is None:
                missing_masks.append(key)
                continue
            if mask_path is None:
                continue

            vol = LoadImage(image_only=True)(str(mask_path))
            mslice = vol[..., sidx] if (vol.ndim == 3 and sidx < vol.shape[-1]) else (vol[..., vol.shape[-1]//2] if vol.ndim == 3 else vol)
            from skimage.transform import resize as imresize
            mslice_r = imresize(mslice.astype(float), spatial_size_hw, order=0, preserve_range=True, anti_aliasing=False)
            bin_mask = (mslice_r > 0.5).astype(np.uint8)

            mean_cam, std_cam = agg_store[(split, "gradcam")][key]
            gbin = (mean_cam > cam_thresh).astype(np.uint8)
            d = dice_coeff(bin_mask.astype(bool), gbin.astype(bool))

            # stash also guided if available
            guided_pair = agg_store.get((split, "guided_gradcam"), {}).get(key, (None, None))

            ensemble_candidates.append({
                "dice": float(d),
                "split": split,
                "key": key,
                "img2d": img2d,
                "mask": bin_mask,
                "grad_mean": mean_cam,
                "grad_std": std_cam,
                "guid_mean": guided_pair[0],
                "guid_std": guided_pair[1],
            })

    # --- Top-K selection and saving (arrays + figure) ---
    K = max(1, int(topk))
    top = sorted(ensemble_candidates, key=lambda x: x["dice"], reverse=True)[:K]

    out_top = cams_ensemble / "dice_top10"
    ensure_out_dir(out_top)

    for ent in top:
        split = ent["split"]; key = ent["key"]
        img2d = ent["img2d"]; bin_mask = ent["mask"]
        grad_mean = ent["grad_mean"]; grad_std = ent["grad_std"]
        guid_mean = ent["guid_mean"]; guid_std = ent["guid_std"]

        # 1) save arrays ONLY for Top-K
        g_dir = cams_ensemble / split / "gradcam"
        ensure_out_dir(g_dir)
        np.save(g_dir / f"{key}_mean.npy", grad_mean)
        np.save(g_dir / f"{key}_std.npy",  grad_std)
        if guid_mean is not None and guid_std is not None:
            gg_dir = cams_ensemble / split / "guided_gradcam"
            ensure_out_dir(gg_dir)
            np.save(gg_dir / f"{key}_mean.npy", guid_mean)
            np.save(gg_dir / f"{key}_std.npy",  guid_std)

        # 2) figure (RAW mean Grad-CAM)
        d = ent["dice"]
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img2d, cmap="gray"); ax1.imshow(bin_mask, alpha=0.3, cmap="Reds")
        ax1.set_title("Image + GT mask"); ax1.axis("off")
        ax2 = fig.add_subplot(1, 2, 2)
        im = ax2.imshow(grad_mean, cmap="rainbow", vmin=0.0, vmax=1.0)
        ax2.set_title("Grad-CAM")
        legend_lbl = f"Dice = {d:.3f}"
        ax2.plot([], [], ' ', label=legend_lbl)
        leg = ax2.legend(loc='upper right', frameon=True)
        for lh in leg.legendHandles:
            lh.set_visible(False)
        ax2.axis("off"); #cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        #cbar.set_label("Intensity", rotation=90)
        fig.tight_layout()
        fig.savefig(out_top / f"{split}__{key}_dice{d:.3f}.png", dpi=200)
        plt.close(fig)

    if missing_masks:
        with open(cams_ensemble / "logs" / "missing_infarct_masks_ensemble.txt", "w") as f:
            for b in sorted(set(missing_masks)):
                f.write(b + "\n")


# -----------------------------
# Inference on Hold-out Test (per fold, no soft voting) — 2D
# -----------------------------

def evaluate_fold_models_2d(
    ckpt_paths: List[Path],
    model_name: str,
    test_slice_list: SliceList,
    eval_tf: Compose,
    device: torch.device,
    out_dir: Path,
    enable_cams: bool,
    infarct_dir_map: Dict[str, Optional[Path]],
    cam_thresh: float,
    spatial_size_hw: Tuple[int, int],
    path_to_pid: Dict[str, str],
    path_label_map: Dict[str, int],
    topk_dice_figs: int,
    save_per_fold: bool
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate each fold checkpoint separately (NO soft voting).
    Saves per-fold figures + classification reports, plus mean ROC/PR with ±1 SD.
    Also generates CAMs and Dice vs masks per fold, and builds **ensemble figures**:
      - confusion_matrix_mean.png (mean ± SD of normalized CMs)
      - copies of roc_curve_mean.png & pr_curve_mean.png into ensemble/figures/
      - cams_ensemble/ (mean & std heatmaps) and dice_top10 figures.
    Returns: {"mean": {...}, "std": {...}} across folds.
    """
    ensure_out_dir(out_dir)
    fig_dir = out_dir / "figures"
    ensure_out_dir(fig_dir)
    ensemble_dir = out_dir / "ensemble"
    ensemble_fig_dir = ensemble_dir / "figures"
    ensure_out_dir(ensemble_fig_dir)
    logs_dir = out_dir / "logs"
    ensure_out_dir(logs_dir)

    ds_test = LGE2DDataset(test_slice_list, label_map=path_label_map, transforms=eval_tf, path_to_pid=path_to_pid)
    dl_test = DataLoader(ds_test, batch_size=64, shuffle=False, num_workers=0,
                         pin_memory=torch.cuda.is_available())

    models = []
    for ck in ckpt_paths:
        model, _ = get_model_2d(model_name, out_channels=2)
        state = torch.load(ck, map_location=device)
        model.load_state_dict(state)
        model = model.to(device).eval()
        models.append(model)

    # Accumulate per-model slice probabilities (also retain tensors for CAM generation)
    all_imgs, all_y, all_paths, all_sidxs = [], [], [], []
    all_probs_models: List[List[float]] = [[] for _ in range(len(models))]

    with torch.no_grad():
        for imgs, labels, paths, sidxs in dl_test:
            # store for cams later
            all_imgs.append(imgs.clone())
            all_y.append(labels.clone())
            all_paths.extend(list(paths))
            all_sidxs.extend([int(s) for s in sidxs])

            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            for mi, m in enumerate(models):
                logits = m(imgs)
                probs = torch.softmax(logits, dim=1)[:, 1]  # p(class=1)
                all_probs_models[mi].extend(probs.detach().cpu().tolist())

    y_true = torch.cat(all_y, dim=0).cpu().numpy().astype(int) if len(all_y) else np.array([])
    y_probs_models = [np.asarray(p, dtype=float) for p in all_probs_models]
    imgs_cat = torch.cat(all_imgs, dim=0).cpu() if len(all_imgs) else torch.zeros(0,1,*spatial_size_hw)  # [N,1,H,W]

    # Metrics & figures per model
    per_model_metrics = []
    cm_pct_list = []      # list of 2x2 percent-of-total matrices (per fold)
    cm_counts_list = []   # list of 2x2 counts (per fold)
    for mi, y_prob in enumerate(y_probs_models, start=1):
        m = compute_metrics(y_true, y_prob)
        m["model_idx"] = mi
        per_model_metrics.append(m)
    
        vals, cnts = np.unique(y_true, return_counts=True)
        print(f"[CM] Fold {mi}: class counts = {dict(zip(vals.tolist(), cnts.tolist()))}, N={len(y_true)}")
    
        # Build confusion matrix (counts and percent-of-total) IN MEMORY
        y_pred = (y_prob >= 0.5).astype(int)
        cf_counts = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int)
        total = int(cf_counts.sum())
        cf_pct = cf_counts.astype(float) / (total if total > 0 else 1)
        cm_counts_list.append(cf_counts)
        cm_pct_list.append(cf_pct)
    
        # --- per-fold SAVES (skip when save_per_fold=False) ---
        if save_per_fold:
            plot_confusion_matrix(
                y_true, y_prob,
                out_png = fig_dir / f"confusion_matrix_fold{mi}.png",
                out_npy = fig_dir / f"cm_pct_fold{mi}.npy",
            )
            try:
                plot_roc_curve(y_true, y_prob, fig_dir / f"roc_curve_fold{mi}.png")
            except Exception:
                pass
            plot_pr_curve(y_true, y_prob, fig_dir / f"pr_curve_fold{mi}.png")
    
            # CAMs per fold
            generate_and_save_cams(
                model=models[mi-1],
                imgs=imgs_cat,
                y_true=y_true,
                y_prob=y_probs_models[mi-1],
                paths=all_paths,
                sidxs=all_sidxs,
                out_dir=out_dir / f"cams_fold{mi}",
                device=device,
                enable_cams=enable_cams,
                infarct_dir_map=infarct_dir_map,
                cam_thresh=cam_thresh,
                spatial_size_hw=spatial_size_hw,
                topk=topk_dice_figs
            )
    
            # Classification report per fold
            if y_true.size > 0:
                report = classification_report(y_true, y_pred, digits=4, target_names=["Class 0","Class 1"])
                with open(out_dir / f"classification_report_fold{mi}.txt", "w") as f:
                    f.write(report)

    # Save per-model metrics table
    df_per_model = pd.DataFrame(per_model_metrics).set_index("model_idx").sort_index()
    df_per_model.to_csv(out_dir / "metrics_per_model.csv", index=True)

    # Aggregate mean and std across models
    metric_keys = ["accuracy","specificity","sensitivity","precision_weighted","f1_weighted","roc_auc"]
    mean_metrics = {k: float(np.nanmean(df_per_model[k].values)) for k in metric_keys if k in df_per_model.columns}
    std_metrics  = {k: float(np.nanstd(df_per_model[k].values, ddof=1)) for k in metric_keys if k in df_per_model.columns}

    # Mean ROC/PR across models (interpolated)
    try:
        if y_true.size > 0:
            fpr_grid = np.linspace(0, 1, 101)
            tpr_models = []
            for y_prob in y_probs_models:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                ufpr, idx = np.unique(fpr, return_index=True)
                tpr = tpr[idx]
                tpr_interp = np.interp(fpr_grid, ufpr, tpr, left=0, right=1)
                tpr_models.append(tpr_interp)
            tpr_models = np.array(tpr_models)
            tpr_mean = tpr_models.mean(axis=0)
            tpr_std  = tpr_models.std(axis=0, ddof=1)

            sns.set_theme(context="talk", style="whitegrid")
            plt.figure(figsize=(7,6))
            plt.plot(fpr_grid, tpr_mean, label=f"Mean ROC (AUC={mean_metrics.get('roc_auc', np.nan):.3f})", linewidth=2.0)
            plt.fill_between(fpr_grid, np.clip(tpr_mean - tpr_std, 0, 1), np.clip(tpr_mean + tpr_std, 0, 1), alpha=0.2, label="±1 SD")
            plt.plot([0,1],[0,1], linestyle="--", linewidth=1.5)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Mean ROC ± SD across models")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(fig_dir / "roc_curve_mean.png", dpi=300)
            plt.close()
            # also copy mean curves into ensemble folder (publication-ready bundle)
            try:
                shutil.copyfile(fig_dir / "roc_curve_mean.png", ensemble_fig_dir / "roc_curve_mean.png")
            except Exception:
                pass
    except Exception:
        pass

    try:
        if y_true.size > 0:
            recall_grid = np.linspace(0, 1, 101)
            prec_models = []
            for y_prob in y_probs_models:
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                recall_rev = recall[::-1]
                precision_rev = precision[::-1]
                prec_interp = np.interp(recall_grid, recall_rev, precision_rev,
                                        left=precision_rev[0], right=precision_rev[-1])
                prec_models.append(prec_interp)
            prec_models = np.array(prec_models)
            prec_mean = prec_models.mean(axis=0)
            prec_std  = prec_models.std(axis=0, ddof=1)

            sns.set_theme(context="talk", style="whitegrid")
            plt.figure(figsize=(7,6))
            plt.plot(recall_grid, prec_mean, label="Mean PR", linewidth=2.0)
            plt.fill_between(recall_grid, np.clip(prec_mean - prec_std, 0, 1), np.clip(prec_mean + prec_std, 0, 1), alpha=0.2, label="±1 SD")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Mean Precision–Recall ± SD across models")
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(fig_dir / "pr_curve_mean.png", dpi=300)
            plt.close()
            try:
                shutil.copyfile(fig_dir / "pr_curve_mean.png", ensemble_fig_dir / "pr_curve_mean.png")
            except Exception:
                pass
    except Exception:
        pass

    # ---- Ensemble confusion matrix (matplotlib only; mean ± SD across folds) ----
    if len(cm_pct_list) > 0:
        cm_stack = np.stack(cm_pct_list, axis=0)  # [F,2,2]
        cm_mean  = cm_stack.mean(axis=0)
        cm_std   = cm_stack.std(axis=0, ddof=1) if cm_stack.shape[0] > 1 else np.zeros_like(cm_mean)
    
        have_counts = (len(cm_counts_list) == len(cm_pct_list))
        if have_counts:
            counts_stack = np.stack(cm_counts_list, axis=0).astype(float)
            counts_mean  = counts_stack.mean(axis=0)
            counts_std   = counts_stack.std(axis=0, ddof=1) if counts_stack.shape[0] > 1 else np.zeros_like(counts_mean)
    
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm_mean, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"])
        ax.set_title("Ensemble Confusion Matrix (mean ± SD) — % of total")
    
        for i in range(2):
            for j in range(2):
                m_pct = cm_mean[i, j] * 100.0
                s_pct = cm_std[i, j] * 100.0
                if have_counts:
                    m_cnt = counts_mean[i, j]
                    s_cnt = counts_std[i, j]
                    text = f"{m_cnt:.0f}±{s_cnt:.0f}\n{m_pct:.2f}%±{s_pct:.2f}%"
                else:
                    text = f"{m_pct:.2f}%±{s_pct:.2f}%"
                color = "white" if cm_mean[i, j] > 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=12, color=color, fontweight="bold")
    
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        from matplotlib.ticker import PercentFormatter
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        cbar.ax.set_ylabel("% of total (mean)", rotation=90)
    
        fig.tight_layout()
        fig.savefig(ensemble_fig_dir / "confusion_matrix_mean.png", dpi=300)
        plt.close(fig)

    out_json = {"mean": mean_metrics, "std": std_metrics}
    with open(out_dir / "test_metrics_mean_std.json", "w") as f:
        json.dump(out_json, f, indent=2)


    # ---- Ensemble CAMs (only if per-fold CAMs were saved) ----
    if save_per_fold and enable_cams:
        try:
            build_ensemble_cams_and_top10(
                out_dir=out_dir,
                imgs=imgs_cat,
                paths=all_paths,
                sidxs=all_sidxs,
                infarct_dir_map=infarct_dir_map,
                cam_thresh=cam_thresh,
                spatial_size_hw=spatial_size_hw,
                topk=topk_dice_figs
            )
        except Exception as e:
            print(f"[WARN] Ensemble CAMs/Dice failed: {e}")
    else:
        print("[INFO] Skipping ensemble CAMs because per-fold CAMs are not being saved.")

    print("\n===== TEST METRICS (mean ± std across models) =====")
    for k in metric_keys:
        m = mean_metrics.get(k, np.nan)
        s = std_metrics.get(k, np.nan)
        print(f"{k:>24s}: {m:.4f} ± {s:.4f}")

    return out_json


# -----------------------------
# Pipeline
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="2D Classification with patient-wise CV + CAMs + Dice + Ensemble figs (slice-per-file aware)")
    parser.add_argument("--images_dir_emidec", type=str, default="/rds/general/project/cardiacmodelsandimages/live/EMIDEC/2D_classified_images_emidec/2D_classified_images",
                        help="Directory with EMIDEC 2D NIfTI images (slices may be per-file)")
    parser.add_argument("--images_dir_imperial", type=str, default="/rds/general/project/cardiacmodelsandimages/live/EMIDEC/2D_classified_images_imperial/2D_classified_images",
                        help="Directory with Imperial 2D NIfTI images (slices may be per-file)")
    parser.add_argument("--labels_csv_emidec", type=str, default="/rds/general/project/cardiacmodelsandimages/live/EMIDEC/Monai_res/emidec_classif_labels_cascaded_2D_2nd_stage_modified.csv",
                        help="EMIDEC labels CSV (with 'label' and filename column)")
    parser.add_argument("--labels_csv_imperial", type=str, default="/rds/general/project/cardiacmodelsandimages/live/EMIDEC/Monai_res/imperial_classif_labels_cascaded_2D_2nd_stage_modified.csv",
                        help="Imperial labels CSV (with 'label' and filename column)")
    parser.add_argument("--infarct_dir_emidec", type=str, default="/rds/general/project/cardiacmodelsandimages/live/EMIDEC/2D_classified_images_emidec/2D_classified_infarc_labels",
                        help="Directory with EMIDEC infarct masks (paired to image filenames)")
    parser.add_argument("--infarct_dir_imperial", type=str, default="/rds/general/project/cardiacmodelsandimages/live/EMIDEC/2D_classified_images_imperial/2D_classified_infarc_labels",
                        help="Directory with Imperial infarct masks (paired to image filenames)")
    parser.add_argument("--out_dir", type=str, default="/rds/general/project/cardiacmodelsandimages/live/EMIDEC/classif_results_2D",
                        help="Output directory for logs, models, figures")
    parser.add_argument("--model_name", type=str, default="densenet121",
                        choices=["densenet121","densenet169","densenet201","densenet264",
                                 "resnet18","resnet34","resnet50","resnet101","resnet152"],
                        help="2D model architecture")
    parser.add_argument("--spatial_size", type=int, nargs=2, default=[224, 224],
                        help="Resize target (H W) for 2D slices")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100,
                        help="Early stopping patience (epochs without AUC improvement)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, default="full", choices=["train","test","full"])
    parser.add_argument("--enable_cams", type=int, default=1, help="1=generate CAMs, 0=skip")
    parser.add_argument("--cam_thresh", type=float, default=0.3, help="Threshold for GradCAM when computing Dice")
    parser.add_argument("--topk_dice_figs", type=int, default=30,
                        help="Save figures only for the top-K Dice slices (per fold and ensemble).")
    parser.add_argument("--strict_label_match", type=int, default=1, help="1=error if any image has no CSV label; 0=log & continue")
    parser.add_argument("--save_per_fold", type=int, default=1,
                        help="1=safe per-fold plots/CAMs/reports; 0=do NOT save any per-fold files (default).")
    args = parser.parse_args()

    print_config()
    seed_everything(args.seed)

    mode = args.mode
    out_dir = Path(args.out_dir) / args.model_name
    out_dir = out_dir.absolute()
    ensure_out_dir(out_dir)
    (out_dir / "models").mkdir(exist_ok=True, parents=True)
    (out_dir / "logs").mkdir(exist_ok=True, parents=True)

    # Link images to labels (and patient ids), log unmatched
    X_emidec, y_emidec, pids_emidec, unm_emidec = link_images_to_labels(
        Path(args.images_dir_emidec), Path(args.labels_csv_emidec), "*.nii.gz", strict=bool(args.strict_label_match)
    )
    X_imperial, y_imperial, pids_imperial, unm_imperial = link_images_to_labels(
        Path(args.images_dir_imperial), Path(args.labels_csv_imperial), "*.nii", strict=bool(args.strict_label_match)
    )
    
    # per-slice label maps
    pathlab_emidec   = {p: int(l) for p, l in zip(X_emidec,   y_emidec)}
    pathlab_imperial = {p: int(l) for p, l in zip(X_imperial, y_imperial)}
    pathlab_all = {**pathlab_emidec, **pathlab_imperial}
        
    # log unmatched images with no labels
    if unm_emidec:
        with open(out_dir / "logs" / "emidec_images_without_label.txt", "w") as f:
            for p in unm_emidec:
                f.write(str(p) + "\n")
    if unm_imperial:
        with open(out_dir / "logs" / "imperial_images_without_label.txt", "w") as f:
            for p in unm_imperial:
                f.write(str(p) + "\n")

    # path -> patient id maps
    p2pid_emidec   = {p: pid for p, pid in zip(X_emidec, pids_emidec)}
    p2pid_imperial = {p: pid for p, pid in zip(X_imperial, pids_imperial)}
    p2pid_all = {**p2pid_emidec, **p2pid_imperial}

    # Build per-dataset patient arrays & patient labels (max over slice labels)
    def split_patients_80_20(paths: List[str], labels: List[int], pids: List[str], seed: int):
        """
        Patient-wise stratified split (80/20).
        Patient label = max of that patient's slice labels (positive if any slice positive).
        """
        pid_to_label: Dict[str, int] = {}
        for lb, pid in zip(labels, pids):
            pid_to_label[pid] = max(pid_to_label.get(pid, 0), int(lb))

        uniq_pids = sorted(pid_to_label.keys())
        y_pid = [pid_to_label[pid] for pid in uniq_pids]

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_idx, te_idx = next(splitter.split(uniq_pids, y_pid))
        tr_pids = set(uniq_pids[i] for i in tr_idx)
        te_pids = set(uniq_pids[i] for i in te_idx)

        # project back to image-level using p2pid_all
        X_tr = [pth for pth in paths if p2pid_all[pth] in tr_pids]
        y_tr = [labels[i] for i, pth in enumerate(paths) if p2pid_all[pth] in tr_pids]
        X_te = [pth for pth in paths if p2pid_all[pth] in te_pids]
        y_te = [labels[i] for i, pth in enumerate(paths) if p2pid_all[pth] in te_pids]
        return X_tr, y_tr, X_te, y_te, list(tr_pids), list(te_pids), pid_to_label

    Xtr_emidec, ytr_emidec, Xte_emidec, yte_emidec, trp_emidec, tep_emidec, pidlab_emidec = split_patients_80_20(
        X_emidec, y_emidec, pids_emidec, args.seed
    )
    Xtr_imperial, ytr_imperial, Xte_imperial, yte_imperial, trp_imperial, tep_imperial, pidlab_imperial = split_patients_80_20(
        X_imperial, y_imperial, pids_imperial, args.seed
    )

    # Combine TRAIN pools on patients; Combine TEST sets
    X_train_patients = Xtr_emidec + Xtr_imperial
    y_train_patients = ytr_emidec + ytr_imperial
    X_test_patients  = Xte_emidec + Xte_imperial
    y_test_patients  = yte_emidec + yte_imperial

    # Build patient label map for combined training pool (pid -> label)
    pidlab = {}
    pidlab.update(pidlab_emidec)
    pidlab.update(pidlab_imperial)
    
    # ===== QC PREVIEW: print 10 random cases per class and STOP (comment out to continue) =====
    # Build pid -> list of slice paths (from all matched slices)
    pid_to_paths = defaultdict(list)
    for p in pathlab_all.keys():
        pid = p2pid_all[p]
        pid_to_paths[pid].append(p)
    
    # Patient-class labels (max over that patient's slice labels)
    pos_pids = [pid for pid, lb in pidlab.items() if int(lb) == 1]
    neg_pids = [pid for pid, lb in pidlab.items() if int(lb) == 0]
    
    random.seed(args.seed)
    k = 10
    pick_pos = random.sample(pos_pids, min(k, len(pos_pids))) if pos_pids else []
    pick_neg = random.sample(neg_pids, min(k, len(neg_pids))) if neg_pids else []
    
    def _qc_lines(pids):
        lines = []
        for pid in pids:
            paths = sorted(pid_to_paths.get(pid, []))
            n_slices = len(paths)
            # true slice-level labels from CSV
            n_pos_slices = sum(int(pathlab_all[p]) for p in paths)
            # show up to 3 example slice filenames
            examples = [Path(p).name for p in paths[:3]]
            lines.append(f"  - {pid} | slices={n_slices} | pos_slices={n_pos_slices} | ex: {examples}")
        return lines
    
    # print("\n[QC] Random NEGATIVE cases (patient label=0):")
    # for ln in _qc_lines(pick_neg): 
    #     print(ln)
    
    # print("\n[QC] Random POSITIVE cases (patient label=1):")
    # for ln in _qc_lines(pick_pos): 
    #     print(ln)
    
    # # Overall slice distribution (from slice-level CSV labels)
    # total_pos_slices = sum(int(pathlab_all[p]) for p in pathlab_all)
    # total_neg_slices = len(pathlab_all) - total_pos_slices
    # print(f"\n[QC] Totals — patients: pos={len(pos_pids)} neg={len(neg_pids)} | "
    #       f"slices: pos={total_pos_slices} neg={total_neg_slices}")
    
    # print("\n[QC] Stopping after QC preview. Comment out this QC block to proceed with training/testing.")
    # sys.exit(0)
    # ===== END QC PREVIEW =====


    print(f"Train patients: N={len(set(trp_emidec)|set(trp_imperial))} | "
          f"Test patients: N={len(set(tep_emidec)|set(tep_imperial))}")

    # 2D transforms
    train_tf, eval_tf = make_2d_transforms(tuple(args.spatial_size))
    device = torch.device(args.device)

    # Patient-wise 5-fold CV on combined training patients
    if mode in ("train", "full"):
        uniq_pids = sorted({p2pid_all[p] for p in X_train_patients})
        y_pid = [pidlab[pid] for pid in uniq_pids]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

        ckpt_paths: List[Path] = []
        for fold_idx, (tr_i, va_i) in enumerate(skf.split(uniq_pids, y_pid), start=1):
            tr_pid_set = set(uniq_pids[i] for i in tr_i)
            va_pid_set = set(uniq_pids[i] for i in va_i)

            # project to their image paths
            X_tr = [p for p in X_train_patients if p2pid_all[p] in tr_pid_set]
            X_va = [p for p in X_train_patients if p2pid_all[p] in va_pid_set]

            # expand to slices
            sl_tr = SliceList(X_tr); sl_tr.build()
            sl_va = SliceList(X_va); sl_va.build()

            ck = train_one_fold_2d(
                fold_idx=fold_idx,
                train_slice_list=sl_tr,
                val_slice_list=sl_va,
                train_tf=train_tf,
                eval_tf=eval_tf,
                args=args,
                device=device,
                out_dir=out_dir / "models",
                path_to_pid=p2pid_all,
                path_label_map=pathlab_all,
            )
            ckpt_paths.append(ck)

    if mode in ("test", "full"):
        remove_old_cm_files(Path(args.out_dir), args.model_name)
        # If just trained, ckpt_paths exist; otherwise load from disk
        if "ckpt_paths" not in locals() or not ckpt_paths:
            models_dir = out_dir / "models"
            ckpt_paths = sorted(models_dir.glob("fold_*/best_model.pth"))
            if not ckpt_paths:
                raise FileNotFoundError(f"No checkpoints found under {models_dir}. "
                                        f"Expected files like fold_1/best_model.pth")

        # Build test slice list from test patients
        test_pid_set = set([p2pid_all[p] for p in X_test_patients])
        X_te = [p for p in X_test_patients if p2pid_all[p] in test_pid_set]
        sl_te = SliceList(X_te); sl_te.build()
        
        # ===== QC PREVIEW (TEST) — print 10 random cases per class and STOP =====
        # Build pid -> list of slice paths *from the TEST set only*
        pid_to_paths_test = defaultdict(list)
        for p in X_te:
            pid_to_paths_test[p2pid_all[p]].append(p)
        
        # Patient labels for TEST patients (max over that patient's slice labels)
        pos_pids_test = [pid for pid in test_pid_set if int(pidlab.get(pid, 0)) == 1]
        neg_pids_test = [pid for pid in test_pid_set if int(pidlab.get(pid, 0)) == 0]
        
        random.seed(args.seed)
        k = 10
        pick_pos_t = random.sample(pos_pids_test, min(k, len(pos_pids_test))) if pos_pids_test else []
        pick_neg_t = random.sample(neg_pids_test, min(k, len(neg_pids_test))) if neg_pids_test else []
        
        # def _qc_lines_test(pids):
        #     lines = []
        #     for pid in pids:
        #         paths = sorted(pid_to_paths_test.get(pid, []))
        #         n_slices = len(paths)
        #         # true slice-level labels from CSV
        #         n_pos_slices = sum(int(pathlab_all[p]) for p in paths)
        #         # show up to 3 example slice filenames
        #         examples = [Path(p).name for p in paths[:3]]
        #         lines.append(f"  - {pid} | slices={n_slices} | pos_slices={n_pos_slices} | ex: {examples}")
        #     return lines
        
        # print("\n[QC-TEST] Random NEGATIVE cases (patient label=0):")
        # for ln in _qc_lines_test(pick_neg_t):
        #     print(ln)
        
        # print("\n[QC-TEST] Random POSITIVE cases (patient label=1):")
        # for ln in _qc_lines_test(pick_pos_t):
        #     print(ln)
        
        # # Overall slice distribution in TEST (from slice-level CSV labels)
        # total_pos_slices_t = sum(int(pathlab_all[p]) for p in X_te)
        # total_neg_slices_t = len(X_te) - total_pos_slices_t
        # print(f"\n[QC-TEST] Totals — patients: pos={len(pos_pids_test)} neg={len(neg_pids_test)} | "
        #       f"slices: pos={total_pos_slices_t} neg={total_neg_slices_t}")
        
        # print("\n[QC-TEST] Stopping after QC preview for the TEST set. "
        #       "Comment out this QC block to proceed with evaluation.")
        # sys.exit(0)
        # # ===== END QC PREVIEW (TEST) =====


        # infarct dirs mapping
        infarct_dir_map = {
            "emidec": Path(args.infarct_dir_emidec) if args.infarct_dir_emidec else None,
            "imperial": Path(args.infarct_dir_imperial) if args.infarct_dir_imperial else None,
        }

        # Evaluate combined test
        metrics_all = evaluate_fold_models_2d(
            ckpt_paths=ckpt_paths,
            model_name=args.model_name,
            test_slice_list=sl_te,
            eval_tf=eval_tf,
            device=device,
            out_dir=out_dir / "test_combined",
            enable_cams=bool(args.enable_cams),
            infarct_dir_map=infarct_dir_map,
            cam_thresh=float(args.cam_thresh),
            spatial_size_hw=tuple(args.spatial_size),
            path_to_pid=p2pid_all,
            path_label_map=pathlab_all,
            topk_dice_figs=args.topk_dice_figs,
            save_per_fold=bool(args.save_per_fold),
        )

        # Also evaluate EMIDEC-only and Imperial-only (patient-wise)
        # EMIDEC
        test_pid_emidec = set(trp_emidec + tep_emidec)  # all seen pids; we'll filter to test set below
        X_te_e = [p for p in Xte_emidec if p2pid_all[p] in test_pid_emidec]
        sl_te_e = SliceList(X_te_e); sl_te_e.build()
        metrics_emidec = evaluate_fold_models_2d(
            ckpt_paths=ckpt_paths,
            model_name=args.model_name,
            test_slice_list=sl_te_e,
            eval_tf=eval_tf,
            device=device,
            out_dir=out_dir / "test_emidec",
            enable_cams=bool(args.enable_cams),
            infarct_dir_map={"emidec": Path(args.infarct_dir_emidec) if args.infarct_dir_emidec else None,
                             "imperial": None},
            cam_thresh=float(args.cam_thresh),
            spatial_size_hw=tuple(args.spatial_size),
            path_to_pid=p2pid_all,
            path_label_map=pathlab_all,
            topk_dice_figs=args.topk_dice_figs,
            save_per_fold=bool(args.save_per_fold),
        )

        # Imperial
        test_pid_imp = set(trp_imperial + tep_imperial)
        X_te_i = [p for p in Xte_imperial if p2pid_all[p] in test_pid_imp]
        sl_te_i = SliceList(X_te_i); sl_te_i.build()
        metrics_imperial = evaluate_fold_models_2d(
            ckpt_paths=ckpt_paths,
            model_name=args.model_name,
            test_slice_list=sl_te_i,
            eval_tf=eval_tf,
            device=device,
            out_dir=out_dir / "test_imperial",
            enable_cams=bool(args.enable_cams),
            infarct_dir_map={"emidec": None,
                             "imperial": Path(args.infarct_dir_imperial) if args.infarct_dir_imperial else None},
            cam_thresh=float(args.cam_thresh),
            spatial_size_hw=tuple(args.spatial_size),
            path_to_pid=p2pid_all,
            path_label_map=pathlab_all,
            topk_dice_figs=args.topk_dice_figs,
            save_per_fold=bool(args.save_per_fold),
        )

        # Summary tables at top-level
        summary_mean = {
            "combined": metrics_all.get("mean", {}),
            "emidec":   metrics_emidec.get("mean", {}),
            "imperial": metrics_imperial.get("mean", {}),
        }
        summary_std = {
            "combined": metrics_all.get("std", {}),
            "emidec":   metrics_emidec.get("std", {}),
            "imperial": metrics_imperial.get("std", {}),
        }
        with open(out_dir / "metrics_summary.json", "w") as f:
            json.dump({"mean": summary_mean, "std": summary_std}, f, indent=2)

        df_mean = pd.DataFrame(summary_mean).T
        df_std  = pd.DataFrame(summary_std).T
        df_mean.to_csv(out_dir / "metrics_summary_mean.csv", index=True)
        df_std.to_csv(out_dir / "metrics_summary_std.csv", index=True)

        print("\n===== SUMMARY (MEAN across models) =====")
        print(df_mean.to_string(float_format=lambda x: f"{x:.4f}"))
        print("\n===== SUMMARY (STD across models) =====")
        print(df_std.to_string(float_format=lambda x: f"{x:.4f}"))

        print("\nDone.")
        print("Combined metrics:", out_dir / "test_combined" / "test_metrics_mean_std.json")
        print("EMIDEC metrics:",  out_dir / "test_emidec"   / "test_metrics_mean_std.json")
        print("Imperial metrics:", out_dir / "test_imperial" / "test_metrics_mean_std.json")
        print("Figures (per-fold + ensemble + CAMs) under each test_* split.")
        print("Logs for unmatched labels/masks under:", out_dir / "logs")

if __name__ == "__main__":
    main()
