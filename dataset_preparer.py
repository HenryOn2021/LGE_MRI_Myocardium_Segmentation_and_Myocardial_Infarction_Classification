# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 09:58:40 2025

@author: Henry
"""

import os
import json
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import inspect_resampled_shapes

class DatasetPreparer_DataSplit:
    def __init__(self, root_dir):
        """
        Class to prepare and save dataset configuration for segmentation/classification tasks.
        Args:
            root_dir (str): The base directory containing all dataset folders.
        """
        self.root_dir = root_dir
        self.repository_dir = os.path.join(root_dir, 'repository')
        os.makedirs(self.repository_dir, exist_ok=True)

        # Define dataset subdirectories
        self.emidec_dataset_dir = os.path.join(root_dir, 'emidec_data')
        self.imperial_dataset_dir = os.path.join(root_dir, 'imperial_data')

        # Internal dict to store output config
        self.dataset_dict = {}

    def prepare(self):
        # === Load EMIDEC dataset info ===
        emidec_images_dir = os.path.join(self.emidec_dataset_dir, 'emidec_train_images')
        emidec_images_list = os.listdir(emidec_images_dir)
        emidec_labels = [0 if "Case_N" in name else 1 for name in emidec_images_list]

        # === Load Imperial dataset info and labels from CSV ===
        imperial_images_dir = os.path.join(self.imperial_dataset_dir, 'self_annotated_images')
        imperial_images_list = os.listdir(imperial_images_dir)
        imperial_labels_path = os.path.join(self.root_dir, 'Monai_res', 'imperial_classif_labels_henry_3D.csv')
        df = pd.read_csv(imperial_labels_path)

        # Convert imperial image names to match CSV keys
        imperial_images_list_seg = [name.replace('.nii', '_seg.nii') for name in imperial_images_list]
        label_df = df.set_index('image_name')['label'].to_dict()
        imperial_labels = [label_df[name] for name in imperial_images_list_seg]

        # === Split into train/test sets ===
        emidec_train_x, emidec_test_x, emidec_train_y, emidec_test_y = train_test_split(
            emidec_images_list, emidec_labels, train_size=0.8, stratify=emidec_labels, random_state=42
        )
        imperial_train_x, imperial_test_x, imperial_train_y, imperial_test_y = train_test_split(
            imperial_images_list, imperial_labels, train_size=0.8, stratify=imperial_labels, random_state=42
        )

        # Combine EMIDEC and Imperial for final train/test
        train_x = emidec_train_x + imperial_train_x
        train_y = emidec_train_y + imperial_train_y
        test_x = emidec_test_x + imperial_test_x
        test_y = emidec_test_y + imperial_test_y

        # Count positive/negative for logging
        train_count = Counter(train_y)
        test_count = Counter(test_y)
        emidec_count = Counter(emidec_labels)
        imperial_count = Counter(imperial_labels)
        
        # Check resampled volume dimension
        inspect_resampled_shapes_emidec = inspect_resampled_shapes(emidec_images_dir, emidec_images_list, target_spacing=(1.0, 1.0, 8.0))
        inspect_resampled_shapes_imperial = inspect_resampled_shapes(imperial_images_dir, imperial_images_list, target_spacing=(1.0, 1.0, 8.0))

        # === Create the final dictionary ===
        self.dataset_dict = {
            'config': {
                'root_dir': self.root_dir,
                'train/test_split': '80:20',
                'emidec_data_dir': self.emidec_dataset_dir,
                'emidec_data_info': inspect_resampled_shapes_emidec,
                'imperial_data_dir': self.imperial_dataset_dir,
                'imperial_data_info': inspect_resampled_shapes_imperial,
                'repository_dir': self.repository_dir,
                'train_POS_NEG': f"{train_count[1]}P_{train_count[0]}N",
                'test_POS_NEG': f"{test_count[1]}P_{test_count[0]}N",
                'emidec_POS_NEG': f"{emidec_count[1]}P_{emidec_count[0]}N",
                'imperial_POS_NEG': f"{imperial_count[1]}P_{imperial_count[0]}N"
            },
            'train': {
                name: {"3D": label, "2D_5mm": None, "2D_8mm": None}
                for name, label in zip(train_x, train_y)
            },
            'test': {
                name: {"3D": label, "2D_5mm": None, "2D_8mm": None}
                for name, label in zip(test_x, test_y)
            },
            'ROI_coords': {x: None for x in (train_x + test_x)}
        }

        # === Save the dictionary to JSON file ===
        out_path = os.path.join(self.repository_dir, 'dataset_config_train_test.json')
        with open(out_path, 'w') as f:
            json.dump(self.dataset_dict, f, indent=4)

        print(f"✅ Dataset config saved to {out_path}")