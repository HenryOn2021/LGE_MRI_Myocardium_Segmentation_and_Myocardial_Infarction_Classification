# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:55:46 2025

3D DenseNet (MONAI) — Stratified 80/20 hold‑out + 5‑fold CV (train/val) across EMIDEC & Imperial
------------------------------------------------------------------------------------------------

What this script does
---------------------
1) Reads two datasets (EMIDEC .nii.gz and Imperial .nii) and their CSV label files.
   The CSV must contain a binary column named "label" and a filename/case column
   (auto-detected among: ["filename","file","case","id","name"]). We link images
   to labels by checking whether the CSV filename (without extension) is contained
   in the image filename. Robust matching and clear errors are provided.

2) Per-dataset stratified 80:20 split -> hold-out TEST sets.
   The remaining 80% from both datasets are COMBINED into a single pool for training/validation.

3) On the combined pool, perform STRATIFIED 5-FOLD CROSS-VALIDATION.
   For each fold: train a 3D DenseNet (user-selectable: 121/169/201/264), monitor val AUC,
   and save the best model checkpoint for that fold.

4) After all 5 folds finish, run INFERENCE on the combined hold-out TEST set:
   load all 5 best models, do soft-voting (probability averaging), and report:
   - Accuracy, Specificity, Sensitivity (Recall for positive class)
   - Weighted Precision, Weighted F1
   - ROC AUC
   - Confusion Matrix (seaborn heatmap)
   - ROC curve (micro-average)
   - Precision-Recall curve
   All figures are saved with large, legible fonts.

Usage (example)
---------------
python lge_3d_densenet_cv_train_infer.py \
  --images_dir_emidec "/path/to/emidec_images" \
  --images_dir_imperial "/path/to/imperial_images" \
  --labels_csv_emidec "/path/to/emidec_labels.csv" \
  --labels_csv_imperial "/path/to/imperial_classif_labels_henry_3D.csv" \
  --out_dir "./outputs_lge_3d" \
  --model_name "densenet121" \
  --spatial_size 96 96 64 \
  --batch_size 4 \
  --max_epochs 100 \
  --patience 20

@author: Henry
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re

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
    LoadImage,
    EnsureChannelFirst,
    Resize,
    NormalizeIntensity,
    RandFlip,
    RandRotate90,
    RandBiasField,
    RandAdjustContrast,
    RandGaussianNoise,
    EnsureType,
)
from monai.networks.nets import DenseNet121, DenseNet169, DenseNet201, DenseNet264
from monai.utils import set_determinism

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
# Data IO
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


def link_images_to_labels(
    images_dir: Path,
    labels_csv: Path,
    pattern: str,
    strict: bool = True,
) -> Tuple[List[str], List[int]]:
    """
    Build (image_path, label) pairs by matching the case/filename from the CSV
    to the image filename (substring-insensitive, extension-agnostic).

    Args:
        images_dir: directory of images (.nii or .nii.gz).
        labels_csv: CSV with columns [<name_col>, "label"].
        pattern: glob pattern to enumerate images (e.g., "*.nii.gz" or "*.nii").

    Returns:
        image_paths, labels (aligned lists).
    """
    images = sorted(list(images_dir.rglob(pattern)))
    if len(images) == 0:
        raise FileNotFoundError(f"No images found in {images_dir} matching {pattern}.")
    df = pd.read_csv(labels_csv)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:")]
    if "label" not in [c.lower() for c in df.columns]:
        raise ValueError(f'"label" column not found in {labels_csv}')
    # Resolve real label column name (case-insensitive match to 'label')
    label_col = [c for c in df.columns if str(c).lower() == "label"][0]
    name_col = _detect_name_col(df)

    # Normalise CSV names (drop extensions if present)
    df["_name_norm"] = df[name_col].apply(_norm_token)

    # Map from normalised name -> label
    name_to_label: Dict[str, int] = {}
    for _, row in df.iterrows():
        name_to_label[str(row["_name_norm"])] = int(row[label_col])

    matched_paths: List[str] = []
    matched_labels: List[int] = []

    # Strategy:
    #   1) exact match on a robustly normalised token (no ext, strip _seg/_mask)
    #   2) else allow two-way substring matching and pick the most specific (longest token)
    for img_path in images:
        base = _norm_token(_basename_no_ext(img_path))
        label: Optional[int] = None
        if base in name_to_label:
            label = name_to_label[base]
        else:
            cands = []
            for nm, lb in name_to_label.items():
                if nm and (nm in base or base in nm):
                    cands.append((nm, lb))
            if cands:
                nm, lb = max(cands, key=lambda t: len(t[0]))
                label = lb
        if label is not None:
            matched_paths.append(str(img_path))
            matched_labels.append(label)

    if len(matched_paths) == 0:
        raise RuntimeError(
            f"No matches between images in {images_dir} and rows in {labels_csv}. "
            f"Check the filename tokens."
        )

    # Drop potential duplicates (same image matched multiple CSV rows)
    uniq: Dict[str, int] = {}
    for p, lb in zip(matched_paths, matched_labels):
        uniq[p] = lb  # last write wins; filenames assumed unique

    image_paths = list(uniq.keys())
    labels = [uniq[p] for p in image_paths]
    
    # Fail fast if any images have no label match
    unmatched = [str(p) for p in images if str(p) not in image_paths]
    if strict and len(unmatched) > 0:
        examples = "\n".join(f"  - {Path(u).name}" for u in unmatched[:20])
        raise RuntimeError(
            f"{len(unmatched)} image(s) in {images_dir} have no label match in {labels_csv}.\n"
            f"Examples:\n{examples}\n"
            "Tips: ensure the CSV filename column is detected correctly "
            "(e.g., 'image_name'), and that tokens align after normalisation "
            "(extensions removed; '_seg'/'_mask' stripped)."
        )
    elif len(unmatched) > 0:
        print(f"[WARN] {len(unmatched)} image(s) in {images_dir} had no label match. "
              "Proceeding because strict=False.")
        
    return image_paths, labels


class LGE3DDataset(Dataset):
    """
    Minimal MONAI/tensor dataset for 3D classification.
    Loads one volume and returns (tensor[C=1,H,W.D], label).
    """
    def __init__(self, image_files: List[str], labels: List[int], transforms: Compose):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img = self.transforms(self.image_files[idx])  # tensor [1,H,W,D]
        lb = int(self.labels[idx])
        return img, lb


def make_transforms(spatial_size: Tuple[int, int, int]) -> Tuple[Compose, Compose]:
    """
    Training and validation/test transforms. Uses light 3D augmentations.
    """
    train_tf = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize(spatial_size=spatial_size),
        NormalizeIntensity(nonzero=True),
        RandFlip(spatial_axis=0, prob=0.2),
        RandFlip(spatial_axis=1, prob=0.2),
        RandRotate90(prob = 0.2, spatial_axes= [0,1]),
        RandBiasField(prob = 0.2, coeff_range = (0, 0.1)),
        RandAdjustContrast(prob = 0.2,  gamma = (0.5, 1)),
        RandGaussianNoise(prob=0.2, mean=0.0, std=0.05),
        EnsureType(),
    ])
    eval_tf = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize(spatial_size=spatial_size),
        NormalizeIntensity(nonzero=True),
        EnsureType(),
    ])
    return train_tf, eval_tf


# -----------------------------
# Models
# -----------------------------

def get_densenet(model_name: str, out_channels: int = 2) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "densenet121":
        return DenseNet121(spatial_dims=3, in_channels=1, out_channels=out_channels)
    if model_name == "densenet169":
        return DenseNet169(spatial_dims=3, in_channels=1, out_channels=out_channels)
    if model_name == "densenet201":
        return DenseNet201(spatial_dims=3, in_channels=1, out_channels=out_channels)
    if model_name == "densenet264":
        return DenseNet264(spatial_dims=3, in_channels=1, out_channels=out_channels)
    raise ValueError(f"Unknown model_name={model_name}. "
                     f"Choose from ['densenet121','densenet169','densenet201','densenet264'].")


# -----------------------------
# Training / Validation (per fold)
# -----------------------------

def train_one_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X: List[str],
    y: List[int],
    train_tf: Compose,
    eval_tf: Compose,
    args,
    device: torch.device,
    out_dir: Path,
) -> Path:
    """
    Train one fold, save best checkpoint (by val AUC), and return its path.
    """
    fold_dir = out_dir / f"fold_{fold_idx}"
    ensure_out_dir(fold_dir)
    ckpt_path = fold_dir / "best_model.pth"
    log_csv = fold_dir / "train_log.csv"

    # Datasets & loaders
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val   = [X[i] for i in val_idx]
    y_val   = [y[i] for i in val_idx]

    ds_train = LGE3DDataset(X_train, y_train, train_tf)
    ds_val   = LGE3DDataset(X_val,   y_val,   eval_tf)

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

    # Model / loss / opt / sched
    model = get_densenet(args.model_name, out_channels=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_auc = -np.inf
    best_state = None
    no_improve = 0

    # Logging
    log_rows = []
    print(f"[Fold {fold_idx}] Train size={len(ds_train)} | Val size={len(ds_val)}")

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, labels in dl_train:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * imgs.size(0)

        epoch_train_loss = running_loss / len(ds_train)

        # ---- Validation
        model.eval()
        all_probs = []
        all_true  = []
        with torch.no_grad():
            for imgs, labels in dl_val:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)[:, 1]  # p(class=1)
                all_probs.append(probs.detach().cpu().numpy())
                all_true.append(labels.detach().cpu().numpy())
        y_true = np.concatenate(all_true)
        y_prob = np.concatenate(all_probs)

        # Guard: if val fold accidentally contains single class (rare with stratification)
        try:
            val_auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            val_auc = np.nan

        y_pred = (y_prob >= 0.5).astype(int)
        val_acc = accuracy_score(y_true, y_pred)
        val_f1w = f1_score(y_true, y_pred, average="weighted")
        val_prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        val_rec_w  = recall_score(y_true, y_pred, average="weighted", zero_division=0)

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

        # Early stopping on AUC (or accuracy if AUC NaN)
        score = val_auc if not np.isnan(val_auc) else val_acc
        improved = score > best_auc
        if improved:
            best_auc = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
            torch.save(best_state, ckpt_path)  # save incrementally
        else:
            no_improve += 1

        print(f"[Fold {fold_idx:02d}][Epoch {epoch:03d}] "
              f"loss={epoch_train_loss:.4f} | AUC={val_auc:.4f} | "
              f"ACC={val_acc:.4f} | F1w={val_f1w:.4f} | "
              f"no_improve={no_improve}/{args.patience}")

        if no_improve >= args.patience:
            print(f"[Fold {fold_idx}] Early stopping at epoch {epoch}.")
            break

    # Write log CSV
    pd.DataFrame(log_rows).to_csv(log_csv, index=False)

    # Ensure a checkpoint exists
    if not ckpt_path.exists() and best_state is not None:
        torch.save(best_state, ckpt_path)

    return ckpt_path


# -----------------------------
# Inference on Hold-out Test (5-fold ensemble)
# -----------------------------

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute required metrics from binary labels and positive-class probs."""
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "specificity": specificity,
        "sensitivity": sensitivity,  # recall for positive class
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = np.nan
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, out_png: Path) -> None:
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])

    sns.set_theme(context="talk", style="whitegrid")
    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(cm_df, annot=True, fmt="d", cbar=False)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_png: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    sns.set_theme(context="talk", style="whitegrid")
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})", linewidth=2.0)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, out_png: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = np.trapz(precision[::-1], recall[::-1])  # rough area; use average_precision_score if needed

    sns.set_theme(context="talk", style="whitegrid")
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"PR Curve (AUC≈{ap:.3f})", linewidth=2.0)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def ensemble_inference_on_test(
    ckpt_paths: List[Path],
    model_name: str,
    X_test: List[str],
    y_test: List[int],
    eval_tf: Compose,
    device: torch.device,
    out_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate each fold checkpoint separately (NO soft voting).
    Saves per-fold figures + classification reports, plus mean ROC/PR with ±1 SD.
    Returns: {"mean": {...}, "std": {...}} across folds.
    """
    ensure_out_dir(out_dir)
    fig_dir = out_dir / "figures"
    ensure_out_dir(fig_dir)

    ds_test = LGE3DDataset(X_test, y_test, eval_tf)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0,
                         pin_memory=torch.cuda.is_available())

    # Load models
    models = []
    for ck in ckpt_paths:
        m = get_densenet(model_name, out_channels=2).to(device)
        state = torch.load(ck, map_location=device)
        m.load_state_dict(state)
        m.eval()
        models.append(m)

    # Per-model probabilities and shared y_true
    y_true: List[int] = []
    y_probs_models: List[List[float]] = [[] for _ in range(len(models))]

    with torch.no_grad():
        for imgs, labels in dl_test:
            imgs = imgs.to(device, non_blocking=True)
            y_true.append(int(labels.item()))
            for mi, m in enumerate(models):
                logits = m(imgs)
                prob_pos = torch.softmax(logits, dim=1)[:, 1].item()
                y_probs_models[mi].append(float(prob_pos))

    y_true = np.asarray(y_true, dtype=int)
    y_probs_models = [np.asarray(p, dtype=float) for p in y_probs_models]

    # Compute metrics per model + save figures/reports
    per_model_metrics = []
    for mi, y_prob in enumerate(y_probs_models, start=1):
        m = compute_metrics(y_true, y_prob)  # uses your helper
        m["model_idx"] = mi
        per_model_metrics.append(m)

        plot_confusion_matrix(y_true, y_prob, fig_dir / f"confusion_matrix_fold{mi}.png")
        try:
            plot_roc_curve(y_true, y_prob, fig_dir / f"roc_curve_fold{mi}.png")
        except Exception:
            pass
        plot_pr_curve(y_true, y_prob, fig_dir / f"pr_curve_fold{mi}.png")

        # Per-model classification report (threshold 0.5)
        y_pred = (y_prob >= 0.5).astype(int)
        report = classification_report(y_true, y_pred, digits=4, target_names=["Class 0", "Class 1"])
        with open(out_dir / f"classification_report_fold{mi}.txt", "w") as f:
            f.write(report)

    # Save per-model metrics table
    df_per_model = pd.DataFrame(per_model_metrics).set_index("model_idx").sort_index()
    df_per_model.to_csv(out_dir / "metrics_per_model.csv", index=True)

    # Aggregate mean/std across models
    metric_keys = ["accuracy", "specificity", "sensitivity", "precision_weighted", "f1_weighted", "roc_auc"]
    mean_metrics = {k: float(np.nanmean(df_per_model[k].values)) for k in metric_keys if k in df_per_model.columns}
    std_metrics  = {k: float(np.nanstd(df_per_model[k].values, ddof=1)) for k in metric_keys if k in df_per_model.columns}

    # Mean ROC ± SD (interpolate TPR on common FPR grid)
    try:
        fpr_grid = np.linspace(0, 1, 101)
        tpr_stack = []
        for y_prob in y_probs_models:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ufpr, idx = np.unique(fpr, return_index=True)
            tpr = tpr[idx]
            tpr_interp = np.interp(fpr_grid, ufpr, tpr, left=0, right=1)
            tpr_stack.append(tpr_interp)
        tpr_stack = np.asarray(tpr_stack)
        tpr_mean, tpr_sd = tpr_stack.mean(axis=0), tpr_stack.std(axis=0, ddof=1)

        sns.set_theme(context="talk", style="whitegrid")
        plt.figure(figsize=(7, 6))
        plt.plot(fpr_grid, tpr_mean, label=f"Mean ROC (AUC={mean_metrics.get('roc_auc', np.nan):.3f})", linewidth=2.0)
        plt.fill_between(fpr_grid, np.clip(tpr_mean - tpr_sd, 0, 1), np.clip(tpr_mean + tpr_sd, 0, 1), alpha=0.2, label="±1 SD")
        plt.plot([0, 1], [0, 1], "--", linewidth=1.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Mean ROC ± SD across models")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(fig_dir / "roc_curve_mean.png", dpi=300)
        plt.close()
    except Exception:
        pass

    # Mean PR ± SD (interpolate precision on common recall grid)
    try:
        recall_grid = np.linspace(0, 1, 101)
        prec_stack = []
        for y_prob in y_probs_models:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            # Make recall increasing for interpolation
            recall_rev = recall[::-1]
            precision_rev = precision[::-1]
            prec_interp = np.interp(recall_grid, recall_rev, precision_rev,
                                    left=precision_rev[0], right=precision_rev[-1])
            prec_stack.append(prec_interp)
        prec_stack = np.asarray(prec_stack)
        prec_mean, prec_sd = prec_stack.mean(axis=0), prec_stack.std(axis=0, ddof=1)

        sns.set_theme(context="talk", style="whitegrid")
        plt.figure(figsize=(7, 6))
        plt.plot(recall_grid, prec_mean, label="Mean PR", linewidth=2.0)
        plt.fill_between(recall_grid, np.clip(prec_mean - prec_sd, 0, 1), np.clip(prec_mean + prec_sd, 0, 1), alpha=0.2, label="±1 SD")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Mean Precision–Recall ± SD across models")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(fig_dir / "pr_curve_mean.png", dpi=300)
        plt.close()
    except Exception:
        pass

    # Save mean±std summary
    out_json = {"mean": mean_metrics, "std": std_metrics}
    with open(out_dir / "test_metrics_mean_std.json", "w") as f:
        json.dump(out_json, f, indent=2)

    # Console print
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
    parser = argparse.ArgumentParser(description="3D DenseNet CV on EMIDEC + Imperial")
    parser.add_argument("--images_dir_emidec", type=str, default="/rds/general/project/cardiacmodelsandimages/live/EMIDEC/3D_classified_images_emidec",
                        help="Directory with EMIDEC .nii.gz images")
    parser.add_argument("--images_dir_imperial", type=str, default="/rds/general/project/cardiacmodelsandimages/live/EMIDEC/3D_classified_images_imperial",
                        help="Directory with Imperial .nii images")
    parser.add_argument("--labels_csv_emidec", type=str, default="/rds/general/project/cardiacmodelsandimages/live/EMIDEC/Monai_res/emidec_classif_labels_henry_3D.csv",
                        help="EMIDEC labels CSV (with 'label' and filename column)")
    parser.add_argument("--labels_csv_imperial", type=str, default="/rds/general/project/cardiacmodelsandimages/live/EMIDEC/Monai_res/imperial_classif_labels_henry_3D.csv",
                        help="Imperial labels CSV (with 'label' and filename column)")
    parser.add_argument("--out_dir", type=str, default="/rds/general/project/cardiacmodelsandimages/live/EMIDEC/classif_results_3D",
                        help="Output directory for logs, models, figures")
    parser.add_argument("--model_name", type=str, default="densenet121",
                        choices=["densenet121","densenet169","densenet201","densenet264"],
                        help="DenseNet variant")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[224, 224, 32],
                        help="Resize target (H W D) for volumes")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100,
                        help="Early stopping patience (epochs without AUC improvement)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, default="full",
                        help="train/test/full")
    args = parser.parse_args()

    print_config()
    seed_everything(args.seed)
    
    mode = args.mode
    out_dir = args.out_dir + f"/{args.model_name}"
    out_dir = Path(out_dir).absolute()
    ensure_out_dir(out_dir)
    (out_dir / "models").mkdir(exist_ok=True, parents=True)

    # -------------------------
    # Link images to labels
    # -------------------------
    images_dir_emidec = Path(args.images_dir_emidec)
    images_dir_imperial = Path(args.images_dir_imperial)
    labels_csv_emidec = Path(args.labels_csv_emidec)
    labels_csv_imperial = Path(args.labels_csv_imperial)

    X_emidec, y_emidec = link_images_to_labels(images_dir_emidec, labels_csv_emidec, "*.nii.gz")
    X_imperial, y_imperial = link_images_to_labels(images_dir_imperial, labels_csv_imperial, "*.nii")

    # -------------------------
    # Per-dataset stratified 80:20 split (hold-out test)
    # -------------------------
    def stratified_split_80_20(X: List[str], y: List[int], seed: int):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y))
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test  = [X[i] for i in test_idx]
        y_test  = [y[i] for i in test_idx]
        return X_train, y_train, X_test, y_test

    Xtr_emidec, ytr_emidec, Xte_emidec, yte_emidec = stratified_split_80_20(X_emidec, y_emidec, args.seed)
    Xtr_imperial, ytr_imperial, Xte_imperial, yte_imperial = stratified_split_80_20(X_imperial, y_imperial, args.seed)

    # Combine TRAIN pools; Combine TEST sets
    X_train_pool = Xtr_emidec + Xtr_imperial
    y_train_pool = ytr_emidec + ytr_imperial
    X_test_all   = Xte_emidec + Xte_imperial
    y_test_all   = yte_emidec + yte_imperial

    print(f"Train pool: N={len(X_train_pool)} | Test (hold-out): N={len(X_test_all)}")
    
    
    # -------------------------
    # CV setup
    # -------------------------
    train_tf, eval_tf = make_transforms(tuple(args.spatial_size))
    device = torch.device(args.device)
    
    if mode in ("train", "full"):
        # Stratified 5-fold on COMBINED TRAIN POOL
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        
        ckpt_paths: List[Path] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train_pool, y_train_pool), start=1):
            ck = train_one_fold(
                fold_idx=fold_idx,
                train_idx=tr_idx,
                val_idx=va_idx,
                X=X_train_pool,
                y=y_train_pool,
                train_tf=train_tf,
                eval_tf=eval_tf,
                args=args,
                device=device,
                out_dir=out_dir / "models",
            )
            ckpt_paths.append(ck)
    
    if mode in ("test", "full"):
        # -------------------------
        # Inference on combined hold‑out test with 5-fold ensemble
        # -------------------------
        # If you just trained above, ckpt_paths already exists.
        # If running inference-only, collect from disk:
        if "ckpt_paths" not in locals() or not ckpt_paths:
            models_dir = out_dir / "models"
            ckpt_paths = sorted(models_dir.glob("fold_*/best_model.pth"))
            if not ckpt_paths:
                raise FileNotFoundError(f"No checkpoints found under {models_dir}. "
                                        f"Expected files like fold_1/best_model.pth")
        
        # Evaluate on combined, EMIDEC-only, and Imperial-only hold-out tests
        metrics_all = ensemble_inference_on_test(
            ckpt_paths=ckpt_paths,
            model_name=args.model_name,
            X_test=X_test_all,
            y_test=y_test_all,
            eval_tf=eval_tf,
            device=device,
            out_dir=out_dir / "test_combined",
        )
        metrics_emidec = ensemble_inference_on_test(
            ckpt_paths=ckpt_paths,
            model_name=args.model_name,
            X_test=Xte_emidec,
            y_test=yte_emidec,
            eval_tf=eval_tf,
            device=device,
            out_dir=out_dir / "test_emidec",
        )
        metrics_imperial = ensemble_inference_on_test(
            ckpt_paths=ckpt_paths,
            model_name=args.model_name,
            X_test=Xte_imperial,
            y_test=yte_imperial,
            eval_tf=eval_tf,
            device=device,
            out_dir=out_dir / "test_imperial",
        )
        
        # Write nested summary (do NOT overwrite it again elsewhere)
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
        print("Figures:")
        print("  Combined:", out_dir / "test_combined" / "figures")
        print("  EMIDEC:",   out_dir / "test_emidec"   / "figures")
        print("  Imperial:", out_dir / "test_imperial" / "figures")

if __name__ == "__main__":
    main()
