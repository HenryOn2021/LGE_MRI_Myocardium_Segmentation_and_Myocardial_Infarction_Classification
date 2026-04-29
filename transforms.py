# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 11:52:57 2025

Modular transform factories for your project:

  • 3D segmentation  : TorchIO (Subjects with 'image' and 'label')
  • 2D segmentation  : MONAI (dictionary-based transforms: *d)
  • Classification   : MONAI (VANILLA transforms: NO 'd' suffix)

Why split?
- Segmentation (2D/3D) must apply the same spatial ops to image AND label → dict pipelines.
- Classification only transforms the image tensor → vanilla (non-d) pipelines are simpler & faster.

Usage:
    from transforms import (
        get_tio_seg3d_transforms,
        get_monai_seg2d_transforms,
        get_monai_classification_transforms,  # vanilla (non-d)
    )

@author: Henry
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, Literal

# -----------------------------
# TorchIO (3D segmentation)
# -----------------------------
import torchio as tio
import torch

def cast_float32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32)

def cast_uint8(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.uint8)

class EnsureDTypes(tio.Transform):
    def __init__(self, image_key: str = "image", label_key: str = "label"):
        super().__init__()
        self.image_key = image_key
        self.label_key = label_key

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # image → float32
        if self.image_key in subject:
            t = subject[self.image_key].data
            if t.dtype != torch.float32:
                subject[self.image_key].set_data(t.to(torch.float32))
        # label → uint8
        if self.label_key in subject:
            t = subject[self.label_key].data
            if t.dtype != torch.uint8:
                subject[self.label_key].set_data(t.to(torch.uint8))
        return subject

def get_tio_seg3d_transforms(
    stage: Literal["train", "val", "test"],
    spacing: Tuple[float, float, float] = (1.0, 1.0, 8.0),
    roi: Tuple[int, int, int] = (512, 512, 16),
    ensure_mult: Optional[int] = 16,
    merge_labels: Optional[Dict[int, int]] = None,
    use_bias: bool = True,
    aug_prob: float = 0.5,
) -> tio.Compose:
    """
    TorchIO transforms for 3D segmentation on Subject(image=ScalarImage, label=LabelMap).
    Order: ToCanonical → Resample → Crop/Pad → EnsureMultiple → ZNorm → (Augs) → Dtype casts.
    """
    tfms = []
    if merge_labels:
        tfms.append(tio.RemapLabels(merge_labels))  # e.g., {2:1, 3:1, 4:1}

    tfms += [
        tio.ToCanonical(),
        tio.Resample(spacing, image_interpolation="bspline"),
        tio.CropOrPad(roi),
    ]
    if ensure_mult is not None:
        tfms.append(tio.EnsureShapeMultiple(ensure_mult, method="pad"))

    tfms.append(tio.ZNormalization(masking_method=tio.ZNormalization.mean))

    if stage == "train":
        augs = [
            tio.RandomFlip(axes=(0,), flip_probability=0.5),
            tio.RandomFlip(axes=(1,), flip_probability=0.5),
            tio.RandomAffine(scales=(0.9, 1.1), degrees=(0, 0, 23), p=aug_prob),
            tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.2),
            tio.RandomNoise(std=(0, 0.1), p=0.2),
            tio.RandomBlur(std=(0.5, 1.0), p=0.2),
        ]
        if use_bias:
            augs.insert(2, tio.RandomBiasField(coefficients=0.3, order=3, p=0.5))
        tfms += augs

    # FINAL hard casts (picklable; no lambdas inside the factory)
    tfms += [
        tio.Lambda(cast_float32, types_to_apply=(tio.INTENSITY,)),
        tio.Lambda(cast_uint8,   types_to_apply=(tio.LabelMap,)),
        #EnsureDTypes(),  # extra safety, fine to keep
    ]

    return tio.Compose(tfms)