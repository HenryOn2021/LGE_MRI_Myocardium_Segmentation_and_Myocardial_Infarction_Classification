# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 13:34:15 2025

Trainer for 2D / 3D / Cascaded3D UNet segmentation on a user-selected CV fold.

Modes
-----
- "2D":          MONAI dict pipeline, but trains on 2D slices extracted from the 3D volumes.
- "3D":          TorchIO 3D pipeline, full volumes.
- "Cascaded3D":  TorchIO 3D pipeline, full volumes, **expects 2-channel input prepared OFFLINE**
                  (e.g., channel-0: image, channel-1: predicted mask from 2D/3D UNet).

Key points
----------
- Runs ONLY the user-chosen fold (1..n_folds) with KFold.
- Saves best model (by Dice) for that fold.
- DiceFocal loss (binary; sigmoid=True).
- 2D mode slices volumes after a deterministic pre-transform; then applies per-slice random augs.

@author: Henry
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="nibabel")

import SimpleITK as sitk
sitk.ProcessObject.SetGlobalWarningDisplay(False)

import os
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable

import numpy as np
import torch
import torchio as tio
from sklearn.model_selection import KFold
from tqdm import tqdm
import datetime as _dt

from monai.networks.nets import UNet
from monai.losses import DiceFocalLoss
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric, HausdorffDistanceMetric

# project-local utilities
from utils import set_seed
from transforms import (
    get_tio_seg3d_transforms,      # TorchIO 3D transforms
)
from dataset import (
    load_json_config, build_segmentation_paths, check_files_exist,
    make_torchio_subjects, make_loaders
)

# ---------------------------
# Helper: polynomial LR schedule (nnU-Net style)
# ---------------------------
def _poly_lambda(power: float, total_iters: int):
    def _inner(it: int):
        return (1.0 - it / float(total_iters)) ** power
    return _inner

# ---------------------------
# Main trainer
# ---------------------------
class SegmentationTrainer:
    """
    Trainer for 2D / 3D / Cascaded3D UNet segmentation on a single selected CV fold.

    mode:
      - "2D"         : MONAI dict (volume-to-slice), in_channels=1
      - "3D"         : TorchIO Subjects, in_channels=1
      - "Cascaded3D" : TorchIO Subjects, in_channels=2  (expects channel-1 to be a predicted mask,
                         prepared OFFLINE; this script does NOT concatenate anything)
    """

    def __init__(
        self,
        root_dir=None,
        json_path=None,
        mode=None,                   # '2D' | '3D' | 'Cascaded3D'
        fold_to_run=None,
        n_folds=None,
        num_res_units=None,
        max_epochs=None,
        train_batch_size=None,
        val_batch_size=None,
        num_workers=None,
        spacing_3d=None,
        roi_3d=None,
        poly_power=None,            # polynomial LR power
        test_version=None,
        # For Cascaded3D only: map filename -> predicted mask path (if you want to build pairs here)
        pred_path_getter=None,
    ):
        set_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.root_dir = root_dir
        self.json_path = json_path
                
        self.mode            = mode            if mode            is not None else "3D"
        self.n_folds         = n_folds         if n_folds         is not None else 5
        self.fold_to_run     = fold_to_run     if fold_to_run     is not None else 1
        self.num_res_units   = num_res_units   if num_res_units   is not None else 2
        self.max_epochs      = max_epochs      if max_epochs      is not None else 1000
        self.train_batch_size= train_batch_size if train_batch_size is not None else 4
        self.val_batch_size  = val_batch_size  if val_batch_size  is not None else 2
        self.num_workers     = num_workers     if num_workers     is not None else 8
        self.poly_power      = poly_power      if poly_power      is not None else 0.9
        self.test_version    = test_version    if test_version    is not None else 1
        
        self.spacing_3d      = spacing_3d      if spacing_3d      is not None else (1.0, 1.0, 8.0)
        self.roi_3d          = roi_3d          if roi_3d          is not None else (256, 256, 18)
        
        self.pred_path_getter = pred_path_getter

        self.date_str = _dt.datetime.today().strftime("%d%m%y")

        # Where models will be saved
        self.save_dir = Path(root_dir) / "models" / f"lge_{mode}_unet_{self.date_str}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Loss and metrics (binary)
        self.loss_function = DiceFocalLoss(
            include_background=True, to_onehot_y=False,
            sigmoid=True, softmax=False, squared_pred=False,
            smooth_nr=0.0, smooth_dr=1e-5,
            gamma=2.0, weight=None,
            lambda_dice=1.0, lambda_focal=1.0, reduction="mean",
        )
        self.post_pred = AsDiscrete(threshold=0.5)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")
        self.hd_metric = HausdorffDistanceMetric(
            include_background=True, distance_metric="euclidean",
            reduction="mean", get_not_nans=False
        )

        # Data containers
        self.images: List[str] = []
        self.masks: List[str] = []
        self.base_data = None                # Subjects (3D/Cascaded) or MONAI dict list (2D volumes)
        self.train_transforms = None         # per-mode
        self.val_transforms = None           # per-mode

    # ---------------------------
    # Public entrypoint
    # ---------------------------
    def run(self):        
        """Prepare data & transforms, build loaders for the selected fold, and train."""
        self._prepare_data_and_transforms()
        
        print(f"Running Segmentation on {self.device}")
        print(f"[QC] Items: {len(self.base_data)}")
        
        if len(self.base_data) == 0:
            raise RuntimeError("No items found in base_data. Check JSON split and file paths.")
        
        # ---- QC: TorchIO subjects (3D / Cascaded3D) ----
        if self.mode in ("3D", "Cascaded3D"):
            sub0 = self.base_data[0]  # TorchIO Subject
            # Resolve file paths from TorchIO ScalarImage/LabelMap
            def _get_path(img_or_path):
                """
                TorchIO ScalarImage/LabelMap may store:
                  - str or Path for single-channel
                  - list/tuple of str/Path for multi-channel
                Return a string path for the first entry.
                """
                p = getattr(img_or_path, "path", img_or_path)
                # Single path
                if isinstance(p, (str, Path)):
                    return str(p)
                # Multiple paths (e.g., cascaded input)
                if isinstance(p, (list, tuple)) and len(p) > 0:
                    first = p[0]
                    return str(first if isinstance(first, (str, Path)) else first)
                raise TypeError(f"Unsupported path type: {type(p)}")
        
            img_path = _get_path(sub0["image"])
            lbl_path = _get_path(sub0["label"]) if "label" in sub0 and sub0["label"] is not None else None
        
            print("[QC] image path:", img_path)
            print("[QC] label path:", lbl_path if lbl_path is not None else "(none)")
        
            # Use the *validation* transform chain for deterministic QC (or train if you prefer)
            qc_tf = self.train_transforms if self.train_transforms is not None else self.val_transforms
            sub0_t = qc_tf(sub0) if qc_tf is not None else sub0
        
            x = sub0_t["image"][tio.DATA]               # (B=1, C, H, W, D) for TorchIO loader; here it's (1,C,H,W,D) or (C,H,W,D)
            if x.dim() == 4:  # (C,H,W,D)
                x_print = x
            else:             # (1,C,H,W,D)
                x_print = x[0]
            print("[QC] image tensor:", x_print.dtype, tuple(x_print.shape))
        
            if "label" in sub0_t and sub0_t["label"] is not None:
                y = sub0_t["label"][tio.DATA]
                y_print = y if y.dim() == 4 else y[0]
                print("[QC] label tensor:", y_print.dtype, tuple(y_print.shape))
                # uniques in processed space (after any merging)
                y_uni = torch.unique(y_print).detach().cpu().tolist()
                print("[QC] label uniques (processed space):", y_uni)
            else:
                print("[QC] label tensor: (none)")
        else:
            raise ValueError("mode must be '2D', '3D', or 'Cascaded3D'")

        # Build KFold indices (on base_data; for 2D we split at VOLUME level and then slice inside)
        indices = np.arange(len(self.base_data))
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # Use only the user-selected fold (1-based)
        sel = self.fold_to_run if 1 <= self.fold_to_run <= self.n_folds else 1
        if sel != self.fold_to_run:
            print(f"[Info] Requested fold={self.fold_to_run} out of [1,{self.n_folds}]. Using fold={sel}.")
        train_idx, val_idx = list(kf.split(indices))[sel - 1]

        # Build loaders depending on mode
        if self.mode in ("3D", "Cascaded3D"):
            # TorchIO full-volume loaders
            train_loader, val_loader = make_loaders(
                train_indices=train_idx, val_indices=val_idx,
                base_data=self.base_data,
                train_transform=self.train_transforms,
                val_transform=self.val_transforms,
                train_batch_size=self.train_batch_size,
                val_batch_size=self.val_batch_size,
                num_workers=self.num_workers,
            )
            in_channels = 2 if self.mode == "Cascaded3D" else 1
        else:
            raise ValueError("mode must be '2D', '3D', or 'Cascaded3D'")
            
        # Peek at 1 batch
        batch_sample = next(iter(train_loader))
        if self.mode in ("3D", "Cascaded3D"):
            img_tensor = batch_sample["image"][tio.DATA]
            mask_tensor = batch_sample["label"][tio.DATA]
        else:
            img_tensor = batch_sample["image"]
            mask_tensor = batch_sample["label"]
        
        print(f"[QC] First train batch - image shape: {tuple(img_tensor.shape)}, dtype: {img_tensor.dtype}")
        print(f"[QC] First train batch - mask shape:  {tuple(mask_tensor.shape)}, dtype: {mask_tensor.dtype}")
        print(f"[QC] Image intensity range: {img_tensor.min().item():.2f} → {img_tensor.max().item():.2f}")
        print(f"[QC] Mask unique labels: {torch.unique(mask_tensor)}")

        # Build UNet for the chosen mode
        spatial_dims = 3
        model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=1,  # binary (sigmoid)
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=self.num_res_units,
            act="PRELU",
            norm="INSTANCE",
        ).to(self.device)
        
        print(f"[QC] Model expects in_channels={in_channels}, got batch channels={img_tensor.shape[1]}")

        # Optimizer + poly LR schedule
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5, amsgrad=True)
        iters_per_epoch = max(len(train_loader), 1)
        total_iters = self.max_epochs * iters_per_epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_poly_lambda(self.poly_power, total_iters))

        # Paths
        best_model_name = f"best_model_segmentation_{self.mode}_Unet_{self.date_str}_fold{sel}_test{self.test_version}.pth"
        ckpt_path = str(self.save_dir / best_model_name)

        # Train loop
        best_dice = -1.0
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            epoch_loss, n_batches = 0.0, 0

            for batch in tqdm(train_loader, desc=f"[{self.mode}] Epoch {epoch}/{self.max_epochs}", leave=False):
                if self.mode in ("3D", "Cascaded3D"):
                    # TorchIO batch: tensors under tio.DATA, shape (B,C,H,W,D)
                    images = batch["image"][tio.DATA].to(self.device)
                    labels = batch["label"][tio.DATA].to(self.device).float()
                else:
                    # 2D dict batch: tensors at keys directly, shape (B,1,H,W)
                    images = batch["image"].to(self.device)
                    labels = batch["label"].to(self.device).float()

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = self.loss_function(logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss /= max(n_batches, 1)

            # ---- Validation
            mean_dice, mean_hd, mean_val_loss = self._validate_one_epoch(model, val_loader)

            lr = optimizer.param_groups[0]["lr"]
            print(f"{self.mode} | Epoch {epoch:03d}/{self.max_epochs} "
                  f"| lr {lr:.2e} "
                  f"| train_loss {epoch_loss:.4f} "
                  f"| val_loss {mean_val_loss:.4f} "
                  f"| Dice {mean_dice:.4f} | HD {mean_hd:.2f}")

            # Save best by Dice
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "dice": best_dice,
                }, ckpt_path)
                print(f"  ↳ Saved best (Dice {best_dice:.4f}) → {ckpt_path}")

    # ---------------------------
    # Internals
    # ---------------------------
    def _prepare_data_and_transforms(self):
        """
        Load JSON → build image/mask paths → create base dataset and transforms
        (per mode). For Cascaded3D we expect **2-channel** image input prepared offline.
        """
        data = load_json_config(self.json_path)

        # Common (images + GT masks) from JSON
        self.images, self.masks = build_segmentation_paths(data, split="train")
        check_files_exist(self.images); check_files_exist(self.masks)

        if self.mode == "3D":
            # TorchIO subjects + 3D transforms
            self.train_transforms = get_tio_seg3d_transforms(
                stage="train",
                spacing=self.spacing_3d,
                roi=self.roi_3d,
                ensure_mult=16,
                merge_labels={1: 0, 2: 1, 3: 1, 4: 1}, # Myocardoim label only
            )
            self.val_transforms = get_tio_seg3d_transforms(
                stage="val",
                spacing=self.spacing_3d,
                roi=self.roi_3d,
                ensure_mult=16,
                merge_labels={1: 0, 2: 1, 3: 1, 4: 1},
            )
            self.base_data = make_torchio_subjects(self.images, self.masks)
        else:
            raise ValueError("mode must be 3D'")

    @torch.no_grad()
    def _validate_one_epoch(self, model, val_loader):
        """Compute Dice/HD/val_loss."""
        model.eval()
        self.dice_metric.reset()
        self.hd_metric.reset()
        val_loss_accum, n_vox = 0.0, 0

        for batch in val_loader:
            if self.mode in ("3D", "Cascaded3D"):
                images = batch["image"][tio.DATA].to(self.device)
                labels = batch["label"][tio.DATA].to(self.device).float()
            else:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device).float()

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = self.post_pred(probs)

            self.dice_metric(y_pred=preds, y=labels)
            self.hd_metric(y_pred=preds, y=labels)

            val_batch_loss = self.loss_function(logits, labels)
            val_loss_accum += val_batch_loss.item() * labels.numel()
            n_vox += labels.numel()

        mean_dice = self.dice_metric.aggregate().item()
        mean_hd = self.hd_metric.aggregate().item()
        mean_val_loss = val_loss_accum / max(n_vox, 1)
        return mean_dice, mean_hd, mean_val_loss
