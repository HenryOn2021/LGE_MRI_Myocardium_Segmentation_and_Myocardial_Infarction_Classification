#!/usr/bin/env bash
python main.py \
  --stage seg_train \
  --mode 3D \
  --root_dir /path/to/project \
  --fold 1 \
  --epochs 500 \
  --train_bs 8 \
  --roi_3d 224,224,18
