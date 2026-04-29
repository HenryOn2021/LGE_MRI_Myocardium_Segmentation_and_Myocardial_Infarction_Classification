# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 14:12:50 2025

@author: Henry
"""

import os
import random
import numpy as np
import nibabel as nib
import torch
from monai.utils import set_determinism
from monai.transforms import MapTransform, Spacing, LoadImage
from tqdm import tqdm

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for CPU and GPU computations where possible.

    Args:
        seed (int): Random seed value. Default is 42.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for MONAI
    set_determinism(seed=seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed (important for some dataloading randomness)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"🔒 Seed set to {seed} for reproducibility.")
    
class MergeForegroundClasses(MapTransform):
    """
    Merge all non-background (label > 0) classes into a single foreground (1).
    Robust to one-hot or index labels; keeps channel dim.
    """
    def __init__(self, keys=("label",)):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            x = d[k]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
            # collapse potential one-hot (C>1) to indices
            if x.dim() >= 3 and x.shape[0] > 1:
                x = torch.argmax(x, dim=0, keepdim=True)
            x = (x > 0).to(torch.uint8)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            d[k] = x
        return d
    
def inspect_resampled_shapes(nifti_dir, nifti_paths, target_spacing=(1.0, 1.0, 8.0), interp_mode='bilinear'):
    """
    Inspect resampled shape and voxel spacing info for a list of NIfTI images.
    
    Args:
        nifti_paths (list): List of paths to 3D NIfTI images.
        target_spacing (tuple): Desired voxel spacing to resample to.
        interp_mode (str): Interpolation mode for resampling ('bilinear', 'nearest', etc.).
        
    Returns:
        dict: Contains max_shape and voxel spacing stats.
    """
    load = LoadImage(image_only=True)

    seen_spacings = set()
    spacing_x = []
    spacing_y = []
    spacing_z = []
    max_shape = [0, 0, 0]
    
    for path in tqdm(nifti_paths, desc="Inspecting volumes"):        
        spacing_transform = Spacing(pixdim=target_spacing, mode=interp_mode)

        # Load and resample
        full_path = os.path.join(nifti_dir, path)
        img = load(full_path)  # shape: [H, W, D]
        spacing = tuple(img.pixdim.tolist())
        spacing_rounded = tuple((np.round(img.pixdim, 3)).tolist())
        
        if spacing_rounded in seen_spacings:
            continue
        seen_spacings.add(spacing_rounded)

        spacing_x.append(spacing[0])
        spacing_y.append(spacing[1])
        spacing_z.append(spacing[2])
        
        resampled = spacing_transform(img)
        
        resampled_shape = resampled.shape

        # Update max shape
        max_shape = [max(max_shape[i], resampled_shape[i]) for i in range(3)]

        print(f"✔ {os.path.basename(path)} | Spacing: {spacing_rounded} → Resampled shape: {resampled_shape}")

    print("\n📏 Voxel Spacing Range per Axis:")
    print(f"   → X: min={min(spacing_x):.3f}, max={max(spacing_x):.3f}")
    print(f"   → Y: min={min(spacing_y):.3f}, max={max(spacing_y):.3f}")
    print(f"   → Z: min={min(spacing_z):.3f}, max={max(spacing_z):.3f}")
    print(f"📐 Max resampled shape (D, H, W): {max_shape}")

    return {
        "max_shape": max_shape,
        "spacing_range": {
            "x": {"min": min(spacing_x), "max": max(spacing_x)},
            "y": {"min": min(spacing_y), "max": max(spacing_y)},
            "z": {"min": min(spacing_z), "max": max(spacing_z)},
        },
        "unique_spacings": list(seen_spacings)
    }