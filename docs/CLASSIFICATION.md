# MI Classification Workflow

The classification branch performs binary MI presence/absence classification using either 2D slices or 3D volumes.

## 2D Models

Supported architectures:

- DenseNet121/169/201/264
- ResNet18/34/50/101/152

Example:

```bash
python main.py \
  --stage classif_full \
  --mode 2D \
  --classif_model resnet50 \
  --images_dir_emidec /path/to/emidec_2d_images \
  --images_dir_imperial /path/to/imperial_2d_images \
  --labels_csv_emidec /path/to/emidec_labels.csv \
  --labels_csv_imperial /path/to/imperial_labels.csv \
  --out_dir outputs/classification_2d
```

## 3D Models

Supported architectures:

- DenseNet121/169/201/264

Example:

```bash
python main.py \
  --stage classif_full \
  --mode 3D \
  --classif_model densenet121 \
  --images_dir_emidec /path/to/emidec_3d_images \
  --images_dir_imperial /path/to/imperial_3d_images \
  --labels_csv_emidec /path/to/emidec_labels.csv \
  --labels_csv_imperial /path/to/imperial_labels.csv \
  --out_dir outputs/classification_3d
```

## Run Modes

The unified entry point maps stages as follows:

| Stage | Underlying classification mode |
|---|---|
| `classif_train` | `train` |
| `classif_eval` | `test` |
| `classif_full` | `full` |

## Outputs

The classification scripts save model checkpoints, metric summaries, ROC/PR curves and confusion matrices under:

```text
<out_dir>/<model_name>/
```
