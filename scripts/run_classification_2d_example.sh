#!/usr/bin/env bash
python main.py \
  --stage classif_full \
  --mode 2D \
  --classif_model resnet50 \
  --images_dir_emidec /path/to/emidec_2d_images \
  --images_dir_imperial /path/to/imperial_2d_images \
  --labels_csv_emidec /path/to/emidec_labels.csv \
  --labels_csv_imperial /path/to/imperial_labels.csv \
  --out_dir outputs/classification_2d \
  --classif_batch_size 8 \
  --lr 3e-4
