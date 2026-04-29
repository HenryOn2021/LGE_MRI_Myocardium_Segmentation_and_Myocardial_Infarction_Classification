# Segmentation Workflow

The segmentation branch supports fold-aware training and native-space inference/evaluation.

## Training

```bash
python main.py \
  --stage seg_train \
  --mode 3D \
  --root_dir /path/to/project \
  --fold 1 \
  --epochs 500 \
  --train_bs 8 \
  --roi_3d 224,224,18
```

## Evaluation

```bash
python main.py \
  --stage seg_eval \
  --mode 3D \
  --root_dir /path/to/project \
  --date_str 100825 \
  --n_folds 5
```

The inference module saves per-fold predictions, probability maps, ensemble predictions and native-space metric summaries where configured by the original pipeline.
