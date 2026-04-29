# -*- coding: utf-8 -*-
"""
Standardised segmentation inference:
- 3D / Cascaded3D: TorchIO val transforms → forward → resample predictions to ORIGINAL space
                   using TorchIO Resample against the native image (no history inversion).
- 2D            : MONAI volume-level preprocess (invertible, deterministic) -> per-slice inference
                   with runtime padding (model-only) -> stack -> MONAI Invertd back to ORIGINAL space.
- Postproc      : threshold(0.5) + Largest Connected Component (3D/2D connectivity handled).
- Metrics       : processed-space (Dice/HD95 in vox) on **binary GT (lab>0)** with include_background=False;
                  native-space (Dice, HD95 in mm).
- Outputs       : per-fold masks + probs; optional nnU-Net style ensemble across folds;
                  CSV reports (no Excel dependency).

Author: Henry
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="nibabel")

import SimpleITK as sitk
sitk.ProcessObject.SetGlobalWarningDisplay(False)

import os
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchio as tio

from skimage.measure import label as _sklabel

# MONAI (processed-space metrics and 2D inversions only)
from monai.networks.nets import UNet
from monai.metrics import DiceMetric, HausdorffDistanceMetric

# ---- project-local utilities
from dataset import (
    load_json_config, build_segmentation_paths, check_files_exist,
    make_torchio_subjects
)
from transforms import (
    get_tio_seg3d_transforms,          # TorchIO 3D transforms (train/val)
)


# ============================================================================
#                          Filename / IO helpers
# ============================================================================

def _add_suffix_before_ext(filename: str, add_suffix: str) -> str:
    """
    Insert a suffix (e.g., '__prob') before the extension, handling both '.nii' and '.nii.gz'.
    Examples:
      'Case_A.nii'      -> 'Case_A__prob.nii'
      'Case_B.nii.gz'   -> 'Case_B__prob.nii.gz'
    """
    name = Path(filename).name
    if name.endswith(".nii.gz"):
        core = name[:-7]
        return f"{core}{add_suffix}.nii.gz"
    stem, ext = os.path.splitext(name)  # ext is '.nii' or ''
    if ext == "":
        ext = ".nii"  # default to NIfTI if somehow missing
    return f"{stem}{add_suffix}{ext}"


def _write_mask_sitk(arr_DHW: np.ndarray, ref_itk: sitk.Image, out_path: Path, dtype=np.uint8):
    """
    Save a (D, H, W) numpy array as a NIfTI with header from ref_itk (origin/direction/spacing).
    dtype can be np.uint8 (labels) or np.float32 (probabilities).
    """
    arr = np.asarray(arr_DHW)
    if dtype == np.uint8:
        arr = (arr > 0).astype(np.uint8)
    img = sitk.GetImageFromArray(arr.astype(dtype))  # (D,H,W)
    img.CopyInformation(ref_itk)
    sitk.WriteImage(img, str(out_path))


# ============================================================================
#                      SimpleITK-native metric helpers (robust)
# ============================================================================

def _resample_label_like(moving_itk: sitk.Image, reference_itk: sitk.Image) -> sitk.Image:
    """
    Resample a *label* (binary or index) image to the reference image grid using nearest-neighbor.
    Keeps reference origin/direction/spacing exactly.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_itk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputOrigin(reference_itk.GetOrigin())
    resampler.SetOutputDirection(reference_itk.GetDirection())
    resampler.SetOutputSpacing(reference_itk.GetSpacing())
    resampler.SetSize(reference_itk.GetSize())
    return resampler.Execute(moving_itk)


def _binarize_numpy(arr: np.ndarray) -> np.ndarray:
    """Ensure binary 0/1 array."""
    return (arr > 0).astype(np.uint8)


def _dice_native(pred_arr: np.ndarray, gt_arr: np.ndarray) -> float:
    """
    Dice in native space from (D,H,W) arrays (0/1).
    Edge cases:
      - both empty -> 1.0
      - one empty  -> 0.0
    """
    p = _binarize_numpy(pred_arr)
    g = _binarize_numpy(gt_arr)
    ps, gs = int(p.sum()), int(g.sum())
    if ps == 0 and gs == 0:
        return 1.0
    if ps == 0 or gs == 0:
        return 0.0
    inter = int((p & g).sum())
    return 2.0 * inter / (ps + gs)


def _hd95_native_mm(pred_itk: sitk.Image, gt_itk: sitk.Image) -> float:
    """
    Symmetric HD95 (95th-percentile surface distance) in millimetres using SimpleITK only.
    """
    pred_bin = sitk.Cast(pred_itk > 0, sitk.sitkUInt8)
    gt_bin   = sitk.Cast(gt_itk > 0,  sitk.sitkUInt8)

    # Empty cases
    if int(sitk.GetArrayFromImage(pred_bin).sum()) == 0 and int(sitk.GetArrayFromImage(gt_bin).sum()) == 0:
        return 0.0
    if int(sitk.GetArrayFromImage(pred_bin).sum()) == 0 or int(sitk.GetArrayFromImage(gt_bin).sum()) == 0:
        return float("inf")

    pred_surf = sitk.BinaryContour(pred_bin, fullyConnected=True)
    gt_surf   = sitk.BinaryContour(gt_bin,   fullyConnected=True)

    dist_to_gt   = sitk.DanielssonDistanceMap(gt_bin,  inputIsBinary=True, squaredDistance=False, useImageSpacing=True)
    dist_to_pred = sitk.DanielssonDistanceMap(pred_bin, inputIsBinary=True, squaredDistance=False, useImageSpacing=True)

    a = sitk.GetArrayFromImage(dist_to_gt)
    a_mask = sitk.GetArrayFromImage(pred_surf).astype(bool)
    a = a[a_mask]

    b = sitk.GetArrayFromImage(dist_to_pred)
    b_mask = sitk.GetArrayFromImage(gt_surf).astype(bool)
    b = b[b_mask]

    if a.size == 0 or b.size == 0:
        return float("inf")

    return max(float(np.percentile(a, 95)), float(np.percentile(b, 95)))


def _resample_like(moving: sitk.Image, reference: sitk.Image, is_label: bool) -> sitk.Image:
    """
    Resample `moving` to `reference` geometry.
    - Nearest for labels, Linear for probabilities/intensities.
    - Preserves the moving image's pixel type.
    """
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    return sitk.Resample(
        moving,                 # image to resample
        reference,              # reference image (size, spacing, direction, origin)
        sitk.Transform(),       # identity
        interp,                 # interpolator
        0.0,                    # default pixel value for out-of-bounds
        moving.GetPixelIDValue()
    )


def _resample_to_grid(img_itk: sitk.Image,
                      out_spacing: Tuple[float, float, float],
                      out_size_xy: Tuple[int, int],
                      is_label: bool) -> sitk.Image:
    """
    Resample a 3D volume to target in-plane spacing and fixed HxW,
    preserving original depth and z-spacing.
    """
    in_size = img_itk.GetSize()  # (W,H,D)
    out_size = (out_size_xy[1], out_size_xy[0], in_size[2])  # (W,H,D)

    out_img = sitk.Image(out_size, img_itk.GetPixelID())
    out_img.SetOrigin(img_itk.GetOrigin())
    out_img.SetDirection(img_itk.GetDirection())
    out_img.SetSpacing((out_spacing[0], out_spacing[1], img_itk.GetSpacing()[2]))  # keep native z spacing

    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    default = 0.0 if not is_label else 0

    # ---- identity transform (version-safe)
    try:
        tx = sitk.IdentityTransform(3)  # some builds have this
    except AttributeError:
        tx = sitk.Transform(3, sitk.sitkIdentity)  # portable identity

    # Use the reference-image overload so origin/spacing/direction match out_img
    return sitk.Resample(img_itk, out_img, tx, interp, default)

def _to_numpy_DHW(img_itk: sitk.Image, dtype=np.float32) -> np.ndarray:
    arr = sitk.GetArrayFromImage(img_itk).astype(dtype)  # (D,H,W)
    return arr

def _from_numpy_DHW(arr: np.ndarray, ref_like: sitk.Image, dtype_out=sitk.sitkFloat32) -> sitk.Image:
    itk = sitk.GetImageFromArray(arr)
    itk = sitk.Cast(itk, dtype_out)
    itk.CopyInformation(ref_like)
    return itk

def _center_pad_or_crop_to(arr: np.ndarray, target: Tuple[int,int,int]) -> np.ndarray:
    D,H,W = arr.shape; Dt,Ht,Wt = target
    def pc(x,t):
        if x==t: return 0,0,slice(0,x)
        if x>t:
            s=(x-t)//2; return 0,0,slice(s,s+t)
        pad=t-x; L=pad//2; R=pad-L; return L,R,slice(0,x)
    pdL,pdR,sd = pc(D,Dt); phL,phR,sh = pc(H,Ht); pwL,pwR,sw = pc(W,Wt)
    out = arr[sd,sh,sw]
    if any([pdL,pdR,phL,phR,pwL,pwR]):
        out = np.pad(out, ((pdL,pdR),(phL,phR),(pwL,pwR)), mode="constant")
    return out


# ============================================================================
#                               Model builder
# ============================================================================

def _build_model(mode: str, num_res_units: int, device: torch.device) -> torch.nn.Module:
    """
    Mirror the latest training UNet:
      spatial_dims = 2 for "2D", else 3
      channels=(16,32,64,128,256), strides=(2,2,2,2)
      out_channels=1 (binary), sigmoid at inference
    """
    spatial_dims = 2 if mode == "2D" else 3
    in_channels = 2 if mode == "Cascaded3D" else 1
    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=num_res_units,
        act="PRELU",
        norm="INSTANCE",
    ).to(device)
    model.eval()
    return model


# ============================================================================
#                  Post-processing: largest connected component
# ============================================================================

def _keep_largest_component(pred_np: np.ndarray) -> np.ndarray:
    m = np.asarray(pred_np > 0, dtype=np.uint8)
    squeezed = np.squeeze(m)
    if squeezed.ndim == 2:
        conn = 2
    elif squeezed.ndim == 3:
        conn = 3
    else:
        return m  # unusual shape; no-op
    lab = _sklabel(squeezed.astype(bool), connectivity=conn)
    if lab.max() == 0:
        return m
    counts = np.bincount(lab.ravel().astype(np.int64))
    counts[0] = 0
    largest = counts.argmax()
    largest_mask = (lab == largest).astype(np.uint8)
    return largest_mask.reshape(m.shape) if largest_mask.shape != m.shape else largest_mask


# ============================================================================
#                          3D / Cascaded3D inference
# ============================================================================

def _infer_subjects_tio(
    model: torch.nn.Module,
    subjects: List[tio.Subject],
    val_transform: tio.Compose,
    device: torch.device,
    out_dir: Path,
    metrics: Optional[Dict[str, object]] = None,
    pred_path_getter: Optional[Callable[[str], str]] = None,  # only for Cascaded3D dynamic stacking
    fold_index: int = 1,
    native_rows: Optional[list] = None,
):
    """
    Inference for 3D / Cascaded3D with **native-space** saving and metrics.

    Implementation notes:
      - Iterate subjects (no DataLoader) to avoid pin-memory/collation issues.
      - After transforms, pad to /16 on the RIGHT only (for UNet concatenations).
      - Predict -> threshold(0.5) -> LCC(3D).
      - Build TorchIO images on processed grid (affine from transformed image).
      - Resample both pred and prob back to the ORIGINAL grid using `tio.Resample(target=subject["image"])`.
      - Save with correct filename handling for '.nii' and '.nii.gz'.
      - Compute processed-space metrics if requested (quick log).
      - Compute native-space metrics with SimpleITK (Dice & HD95 mm) and append rows to `native_rows`.
    """

    dice_metric: DiceMetric = metrics.get("dice") if metrics else None
    hd_metric: HausdorffDistanceMetric = metrics.get("hd") if metrics else None

    out_dir.mkdir(parents=True, exist_ok=True)
    all_case_dice, all_case_hd = [], []

    with torch.no_grad():
        for subject in tqdm(subjects, desc="Infer (TorchIO)", leave=False):
            # If Cascaded3D: dynamically stack predicted mask as 2nd channel
            if pred_path_getter is not None:
                img_ = subject["image"]
                img_path = img_.path if isinstance(img_.path, (str, Path)) else img_.path[0]
                fname = Path(img_path).name
                pmask_path = pred_path_getter(fname)
                if not Path(pmask_path).exists():
                    raise FileNotFoundError(f"[Cascaded3D] Missing offline pred: {pmask_path}")
                subject = tio.Subject(
                    image=tio.ScalarImage([img_path, pmask_path]),
                    label=subject.get("label", None)
                )

            # Deterministic validation transforms
            subject_t = val_transform(subject) if val_transform is not None else subject
            subject_t = tio.EnsureShapeMultiple(16, method="pad")(subject_t)

            # ---- Forward pass (add batch dim if needed)
            x = subject_t["image"][tio.DATA]  # (C,H,W,D)
            if x.dim() == 4:
                x = x.unsqueeze(0)            # (1,C,H,W,D)
            elif x.dim() != 5:
                raise RuntimeError(f"Unexpected image tensor shape: {tuple(x.shape)}")
            x = x.to(device)

            # Right-padding to next multiple of 16 (safety)
            def _next_mult(v, m=16): return ((v + m - 1) // m) * m
            _, _, H, W, D = x.shape
            Ht, Wt, Dt = _next_mult(H), _next_mult(W), _next_mult(D)
            pad_h, pad_w, pad_d = Ht - H, Wt - W, Dt - D
            if pad_h or pad_w or pad_d:
                x_padded = F.pad(x, (pad_d, 0, pad_w, 0, pad_h, 0))
            else:
                x_padded = x

            logits = model(x_padded)
            prob_full = torch.sigmoid(logits)
            prob = prob_full[..., -H:, -W:, -D:]            # crop back to (H,W,D)

            # numpy volumes in processed space
            prob_np = prob.cpu().numpy().astype(np.float32)[0, 0]    # (H,W,D)
            pred_np = (prob_np >= 0.5).astype(np.uint8)
            pred_np = _keep_largest_component(pred_np)

            # ---- Build TorchIO images on processed grid (affine from subject_t image)
            proc_img: tio.ScalarImage = subject_t["image"]
            proc_affine = proc_img.affine

            pred_proc_img = tio.LabelMap(tensor=torch.from_numpy(pred_np[None, ...]), affine=proc_affine)
            prob_proc_img = tio.ScalarImage(tensor=torch.from_numpy(prob_np[None, ...]), affine=proc_affine)

            # ---- Resample both to ORIGINAL grid using the native image as target
            native_target: tio.ScalarImage = subject["image"]
            resample_to_native = tio.Resample(native_target)

            pred_native_img: tio.LabelMap = resample_to_native(pred_proc_img)
            prob_native_img: tio.ScalarImage = resample_to_native(prob_proc_img)

            # ---- Save with original filename and corrected suffix handling
            src_field = subject["image"]
            src_path = src_field.path if isinstance(src_field.path, (str, Path)) else src_field.path[0]
            base = Path(src_path).name

            mask_path = out_dir / base
            prob_name = _add_suffix_before_ext(base, "__prob")
            prob_path = out_dir / prob_name

            pred_native_img.save(str(mask_path))
            prob_native_img.save(str(prob_path))

            # ---- Processed-space metrics (optional)
            if metrics is not None and "label" in subject:
                lab_proc = val_transform(subject)
                lab_np = lab_proc["label"][tio.DATA][0].numpy().astype(np.uint8)  # (H,W,D)
                y_pred = torch.from_numpy(pred_np[None, None, ...])
                y_true = torch.from_numpy(lab_np[None, None, ...])
                dice_metric(y_pred=y_pred, y=y_true)
                hd_metric(y_pred=y_pred, y=y_true)
                d = dice_metric.aggregate().item()
                h = hd_metric.aggregate().item()
                dice_metric.reset(); hd_metric.reset()
                all_case_dice.append(d); all_case_hd.append(h)

            # ---- Native-space metrics via SimpleITK
            if native_rows is not None and "label" in subject:
                lab_path = subject["label"].path if isinstance(subject["label"].path, (str, Path)) else subject["label"].path[0]
                if lab_path and Path(lab_path).exists():
                    pred_itk = sitk.ReadImage(str(mask_path))      # already on native grid
                    img_itk  = sitk.ReadImage(str(src_path))       # native reference
                    gt_raw   = sitk.ReadImage(str(lab_path))       # GT (maybe different grid)
                    gt_native = _resample_label_like(gt_raw, img_itk)

                    dice_nat = _dice_native(
                        sitk.GetArrayFromImage(pred_itk), sitk.GetArrayFromImage(gt_native)
                    )
                    hd95_mm  = _hd95_native_mm(pred_itk, gt_native)

                    native_rows.append({
                        "fold": int(fold_index),
                        "case": base,
                        "dice_native": float(dice_nat),
                        "hd95_native_mm": float(hd95_mm),
                    })

    return all_case_dice, all_case_hd


# ============================================================================
#                         nnU-Net–style ENSEMBLE
# ============================================================================

def _ensemble_across_folds_native(
    out_root: Path,
    fold_indices: Sequence[int],
    test_images: Sequence[str],
    test_labels: Optional[Sequence[str]] = None,
    write_metrics: bool = True,
) -> None:
    """
    nnU-Net–style ensemble in NATIVE space:
      - For each case, load all available per-fold <case>__prob.nii[.gz] volumes
      - Average probs → threshold at 0.5 → LCC
      - Save <case>__prob.nii[.gz] and <case>.nii[.gz] under <out_root>/ensemble/
      - If GT available, compute native-space Dice & HD95 (mm) via SimpleITK and write CSVs
    """
    ens_dir = out_root / "ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for ip, lp in tqdm(list(zip(test_images, test_labels or [None] * len(test_images))), desc="Ensemble", leave=False):
        base = Path(ip).name
        prob_name = _add_suffix_before_ext(base, "__prob")

        # Gather available prob paths across folds
        prob_paths = []
        for k in fold_indices:
            p = out_root / f"fold{k}" / prob_name
            if p.exists():
                prob_paths.append(p)

        if not prob_paths:
            print(f"[Ensemble] No fold probs found for {base}; skipping.")
            continue

        # Load and accumulate (D,H,W), SimpleITK ensures header if needed
        ref_img_itk = sitk.ReadImage(str(ip))
        acc = None
        count = 0
        for p in prob_paths:
            prob_itk = sitk.ReadImage(str(p))
            # sanity: resample to ref grid if shapes differ (shouldn't, but safe)
            if prob_itk.GetSize() != ref_img_itk.GetSize() or prob_itk.GetSpacing() != ref_img_itk.GetSpacing():
                prob_itk = sitk.Resample(prob_itk, ref_img_itk, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
            arr = sitk.GetArrayFromImage(prob_itk).astype(np.float32)  # (D,H,W)
            if acc is None:
                acc = arr
            else:
                acc += arr
            count += 1

        avg_prob = acc / max(count, 1)

        # Threshold + LCC
        pred = (avg_prob >= 0.5).astype(np.uint8)
        pred = _keep_largest_component(pred)

        # Save ensemble outputs
        mask_path = ens_dir / base
        prob_path = ens_dir / prob_name
        _write_mask_sitk(pred, ref_img_itk, mask_path, dtype=np.uint8)
        _write_mask_sitk(avg_prob, ref_img_itk, prob_path, dtype=np.float32)

        # Metrics (native-space)
        if write_metrics and lp and Path(lp).exists():
            gt_raw = sitk.ReadImage(str(lp))
            gt_native = _resample_label_like(gt_raw, ref_img_itk)

            dice_nat = _dice_native(pred, sitk.GetArrayFromImage(gt_native))
            hd95_mm  = _hd95_native_mm(sitk.ReadImage(str(mask_path)), gt_native)

            rows.append({
                "case": base,
                "n_folds_used": count,
                "dice_native": float(dice_nat),
                "hd95_native_mm": float(hd95_mm),
            })

    # Write CSVs
    if write_metrics and len(rows) > 0:
        df = pd.DataFrame(rows)
        df.to_csv(ens_dir / "native_metrics_ensemble.csv", index=False)

        summary = pd.DataFrame({
            "dice_native_mean": [float(df["dice_native"].mean())],
            "dice_native_std":  [float(df["dice_native"].std(ddof=0))],
            "hd95_native_mm_mean": [float(df["hd95_native_mm"].mean())],
            "hd95_native_mm_std":  [float(df["hd95_native_mm"].std(ddof=0))],
            "n_cases": [int(len(df))],
            "avg_folds_used": [float(df["n_folds_used"].mean())],
        })
        summary.to_csv(ens_dir / "overall_native_summary_ensemble.csv", index=False)
        print("\n========== Ensemble (native-space) ==========")
        print(summary.to_string(index=False))


# ============================================================================
#                           Public entrypoint
# ============================================================================

def run_segmentation_inference(
    root_dir: str,
    json_path: str,
    mode: str,                       # keep required
    n_folds: Optional[int] = None,   # discover if None
    num_res_units: Optional[int] = None,  # infer from ckpt if None
    date_str: str = "",
    test_version: Optional[int] = None,   # discover if None
    run_tag: Optional[str] = None,        # default if None
    no_metrics: bool = False,
    spacing_3d: Optional[Tuple[float, float, float]] = None,
    roi_3d: Optional[Tuple[int, int, int]] = None,
    cascaded_pred_root: Optional[str] = None,
    do_ensemble: bool = True,
):
    """
    Orchestrates cross-fold inference with saving + optional metrics.
    - Saves per-fold predictions to <root_dir>/predictions/<run_tag>/fold{k}/
    - Writes per-fold CSVs:
        * processed_metrics_fold{k}.csv   (if GT available and no_metrics=False)
        * native_metrics_fold{k}.csv      (if GT available and no_metrics=False)
    - Writes overall summary CSV:
        * overall_native_summary.csv      (mean across folds, native-space)
    - If do_ensemble=True:
        * Averages per-fold native __prob volumes and writes ensemble outputs + metrics under:
            <root_dir>/predictions/<run_tag>/ensemble/
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoints
    model_dir = Path(root_dir) / "models" / f"lge_{mode}_unet_{date_str}"
    out_root = Path(root_dir) / "predictions" / run_tag
    out_root.mkdir(parents=True, exist_ok=True)

    # Dataset lists (test split)
    data = load_json_config(json_path)
    test_images, test_labels = build_segmentation_paths(data, split="test")
    check_files_exist(test_images)
    if not no_metrics:
        check_files_exist(test_labels)

    # Processed-space metrics (optional)
    metrics = None
    if not no_metrics:
        metrics = {
            "dice": DiceMetric(include_background=True, reduction="mean"),
            "hd": HausdorffDistanceMetric(include_background=True, distance_metric="euclidean",
                                          reduction="mean", get_not_nans=False),
        }

    # Preparations per mode
    if mode in ("3D", "Cascaded3D"):
        subjects = make_torchio_subjects(test_images, test_labels if not no_metrics else None)
        val_transform = get_tio_seg3d_transforms(
            stage="val",
            spacing=spacing_3d,
            roi=roi_3d,
            ensure_mult=16,
            merge_labels={1: 0, 2: 1, 3: 1, 4: 1},  # merged to single foreground for processed-space metrics
        )
        pred_path_getter = None
        if mode == "Cascaded3D":
            assert cascaded_pred_root is not None, "Cascaded3D requires cascaded_pred_root."
            cascaded_pred_root = Path(cascaded_pred_root)
            def _getter(fname: str) -> str:
                # Try exact; fallback from '.nii' to '.nii.gz'
                p = cascaded_pred_root / fname
                if p.exists():
                    return str(p)
                if fname.endswith(".nii"):
                    gz = cascaded_pred_root / (Path(fname).stem + ".nii.gz")
                    if gz.exists():
                        return str(gz)
                return str(p)  # will raise FileNotFoundError upstream if missing
            pred_path_getter = _getter

    # Run all folds
    per_fold_means = []  # list of (native_dice_mean, native_hd95_mean)
    available_folds: List[int] = []

    for k in range(1, n_folds + 1):
        print(f"\n========== Fold {k} ==========")
        ckpt_name = f"best_model_segmentation_{mode}_Unet_{date_str}_fold{k}_test{test_version}.pth"
        ckpt_path = model_dir / ckpt_name
        if not ckpt_path.exists():
            print(f"[WARN] Missing checkpoint for fold {k}: {ckpt_path}")
            continue

        # Build/load model
        model = _build_model(mode, num_res_units, device)
        state = torch.load(str(ckpt_path), map_location=device)
        mstate = state.get("model_state", state)
        model.load_state_dict(mstate)

        out_dir = out_root / f"fold{k}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Accumulators for CSVs
        native_rows: List[Dict] = []
        proc_rows: List[Dict] = []

        # Inference per mode
        if mode in ("3D", "Cascaded3D"):
            d_list, h_list = _infer_subjects_tio(
                model=model,
                subjects=subjects,
                val_transform=val_transform,
                device=device,
                out_dir=out_dir,
                metrics=metrics,
                pred_path_getter=pred_path_getter if mode == "Cascaded3D" else None,
                fold_index=k,
                native_rows=native_rows,
            )

        # ---- Write per-fold CSVs (if GT exists)
        if not no_metrics:
            # Processed-space metrics (summary + optional per-case rows if lengths line up)
            if len(d_list) > 0:
                d_mean = float(np.nanmean(d_list))
                d_std  = float(np.nanstd(d_list))
                h_mean = float(np.nanmean(h_list)) if np.isfinite(np.nanmean(h_list)) else float("nan")
                print(f"[Fold {k}] Processed-space: N={len(d_list)} | Dice {d_mean:.4f}±{d_std:.4f} | "
                      f"HD95 {('%.2f' % h_mean) if np.isfinite(h_mean) else 'nan'} vox")
            
                # Always write at least a summary CSV
                pd.DataFrame([{
                    "fold": k,
                    "case": "__summary__",
                    "dice_processed": d_mean,
                    "hd95_processed_vox": h_mean,
                    "n_cases": len(d_list),
                }]).to_csv(out_dir / f"processed_metrics_fold{k}.csv", index=False)
            
                # If counts line up, append per-case rows too (optional)
                if len(d_list) == len(test_images):
                    for case_idx, (d, h) in enumerate(zip(d_list, h_list)):
                        proc_rows.append({
                            "fold": k,
                            "case_idx": case_idx,
                            "dice_processed": float(d),
                            "hd95_processed_vox": float(h),
                        })
                    # append to the same CSV
                    pd.DataFrame(proc_rows).to_csv(out_dir / f"processed_metrics_fold{k}.csv", mode="a", index=False, header=False)
            else:
                print(f"[Fold {k}] Processed-space: skipped (no metrics computed for any case).")


            # Native-space metrics (per-case)
            pd.DataFrame(native_rows).to_csv(out_dir / f"native_metrics_fold{k}.csv", index=False)

            # Track fold means for overall summary
            if len(native_rows) > 0:
                dice_mean = float(np.mean([r["dice_native"] for r in native_rows]))
                hd_mean   = float(np.mean([r["hd95_native_mm"] for r in native_rows]))
                per_fold_means.append((dice_mean, hd_mean))
                available_folds.append(k)
                print(f"[Fold {k}] N={len(native_rows)} | Native Dice {dice_mean:.4f} | Native HD95 {hd_mean:.2f} mm")
            else:
                print(f"[Fold {k}] No native metrics computed (no GT?)")
                available_folds.append(k)  # still mark fold as available for ensembling (preds exist)
        else:
            # metrics skipped but predictions exist — mark fold as available
            available_folds.append(k)

    # ---- Overall summary CSV (native-space)
    if not no_metrics and len(per_fold_means) > 0:
        dice_means = np.array([x[0] for x in per_fold_means], dtype=float)
        hd_means   = np.array([x[1] for x in per_fold_means], dtype=float)
        overall = pd.DataFrame({
            "dice_native_mean": [float(dice_means.mean())],
            "dice_native_std":  [float(dice_means.std(ddof=0))],
            "hd95_native_mm_mean": [float(hd_means.mean())],
            "hd95_native_mm_std":  [float(hd_means.std(ddof=0))],
            "n_folds": [len(per_fold_means)],
        })
        overall.to_csv(out_root / "overall_native_summary.csv", index=False)
        print("\n========== Overall (native-space across folds) ==========")
        print(overall.to_string(index=False))

    # ---- nnU-Net–style ENSEMBLE (optional)
    if do_ensemble:
        print("\n[Ensemble] Averaging per-fold native probabilities → ensemble/")
        _ensemble_across_folds_native(
            out_root=out_root,
            fold_indices=available_folds if len(available_folds) > 0 else list(range(1, n_folds + 1)),
            test_images=test_images,
            test_labels=None if no_metrics else test_labels,
            write_metrics=not no_metrics,
        )