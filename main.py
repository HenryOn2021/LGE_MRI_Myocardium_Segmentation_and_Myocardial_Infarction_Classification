# -*- coding: utf-8 -*-
"""
Unified command-line entry point for the LGE cardiac MRI AI pipeline.

The repository contains two major workflows:

1. Segmentation
   - 2D / 3D / Cascaded3D U-Net style training and inference.
   - Native-space inference/evaluation using MONAI/SimpleITK utilities.

2. Binary MI classification
   - 2D slice-based MI classification with DenseNet/ResNet model choices.
   - 3D volume-based MI classification with DenseNet model choices.
   - Patient-wise train/validation/test handling and standardised outputs.

Example usage
-------------
Segmentation training:
    python main.py --stage seg_train --mode 3D --root_dir /path/to/project --fold 1

Segmentation evaluation:
    python main.py --stage seg_eval --mode 3D --root_dir /path/to/project --date_str 100825

2D classification, full train + test:
    python main.py --stage classif_full --mode 2D --classif_model resnet50 \
        --images_dir_emidec /path/to/emidec_2d_images \
        --images_dir_imperial /path/to/imperial_2d_images \
        --labels_csv_emidec /path/to/emidec_labels.csv \
        --labels_csv_imperial /path/to/imperial_labels.csv \
        --out_dir outputs/classification_2d

3D classification, full train + test:
    python main.py --stage classif_full --mode 3D --classif_model densenet121 \
        --images_dir_emidec /path/to/emidec_3d_images \
        --images_dir_imperial /path/to/imperial_3d_images \
        --labels_csv_emidec /path/to/emidec_labels.csv \
        --labels_csv_imperial /path/to/imperial_labels.csv \
        --out_dir outputs/classification_3d
"""

import argparse
import sys
from pathlib import Path
from typing import List

from dataset_preparer import DatasetPreparer_DataSplit
from segmentation_myo import SegmentationTrainer
from segmentation_inference_myo import run_segmentation_inference


# -------------------------------------------------------------------------
# Helper parsers
# -------------------------------------------------------------------------
def _parse_float_triplet(s: str):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected 3 comma-separated floats, e.g. 1.0,1.0,8.0")
    return tuple(float(p) for p in parts)


def _parse_int_pair(s: str):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected 2 comma-separated ints, e.g. 224,224")
    return tuple(int(p) for p in parts)


def _parse_int_triplet(s: str):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected 3 comma-separated ints, e.g. 224,224,32")
    return tuple(int(p) for p in parts)


def _bool_to_int(flag: bool) -> str:
    return "1" if flag else "0"


# -------------------------------------------------------------------------
# Classification dispatchers
# -------------------------------------------------------------------------
def _run_classification_2d(args, classif_mode: str) -> None:
    """Call the original 2D classification script through its parser-compatible CLI."""
    import LGE_Classification_2D as classif_2d

    argv: List[str] = [
        "LGE_Classification_2D.py",
        "--images_dir_emidec", args.images_dir_emidec,
        "--images_dir_imperial", args.images_dir_imperial,
        "--labels_csv_emidec", args.labels_csv_emidec,
        "--labels_csv_imperial", args.labels_csv_imperial,
        "--out_dir", args.out_dir,
        "--model_name", args.classif_model,
        "--spatial_size", str(args.spatial_size_2d[0]), str(args.spatial_size_2d[1]),
        "--batch_size", str(args.classif_batch_size),
        "--num_workers", str(args.num_workers),
        "--max_epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--device", args.device,
        "--mode", classif_mode,
        "--enable_cams", _bool_to_int(args.enable_cams),
        "--cam_thresh", str(args.cam_thresh),
        "--topk_dice_figs", str(args.topk_dice_figs),
        "--strict_label_match", _bool_to_int(args.strict_label_match),
        "--save_per_fold", _bool_to_int(args.save_per_fold),
    ]

    if args.infarct_dir_emidec:
        argv.extend(["--infarct_dir_emidec", args.infarct_dir_emidec])
    if args.infarct_dir_imperial:
        argv.extend(["--infarct_dir_imperial", args.infarct_dir_imperial])

    old_argv = sys.argv
    try:
        sys.argv = argv
        classif_2d.main()
    finally:
        sys.argv = old_argv


def _run_classification_3d(args, classif_mode: str) -> None:
    """Call the original 3D classification script through its parser-compatible CLI."""
    import LGE_Classification_test5_3D as classif_3d

    argv: List[str] = [
        "LGE_Classification_test5_3D.py",
        "--images_dir_emidec", args.images_dir_emidec,
        "--images_dir_imperial", args.images_dir_imperial,
        "--labels_csv_emidec", args.labels_csv_emidec,
        "--labels_csv_imperial", args.labels_csv_imperial,
        "--out_dir", args.out_dir,
        "--model_name", args.classif_model,
        "--spatial_size", str(args.spatial_size_3d[0]), str(args.spatial_size_3d[1]), str(args.spatial_size_3d[2]),
        "--batch_size", str(args.classif_batch_size),
        "--num_workers", str(args.num_workers),
        "--max_epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--device", args.device,
        "--mode", classif_mode,
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        classif_3d.main()
    finally:
        sys.argv = old_argv


# -------------------------------------------------------------------------
# Main CLI
# -------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="LGE cardiac MRI segmentation and MI classification pipeline"
    )

    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=[
            "data_prep",
            "seg_train",
            "seg_eval",
            "classif_train",
            "classif_eval",
            "classif_full",
        ],
        help="Pipeline stage to run.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="3D",
        choices=["2D", "3D", "Cascaded3D"],
        help="Pipeline dimensionality/mode. Use 2D or 3D for classification; 3D/Cascaded3D for segmentation.",
    )

    # ------------------------------------------------------------------
    # Global arguments
    # ------------------------------------------------------------------
    parser.add_argument("--root_dir", type=str, default=None, help="Project root directory for segmentation/data preparation.")
    parser.add_argument("--json_path", type=str, default=None, help="Dataset split JSON path for segmentation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=8)

    # ------------------------------------------------------------------
    # Segmentation arguments
    # ------------------------------------------------------------------
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=1, help="Fold index for segmentation training, using the original 1-based convention.")
    parser.add_argument("--epochs", type=int, default=500, help="Maximum training epochs.")
    parser.add_argument("--train_bs", type=int, default=8, help="Segmentation training batch size.")
    parser.add_argument("--val_bs", type=int, default=2, help="Segmentation validation batch size.")
    parser.add_argument("--num_res_units", type=int, default=None)
    parser.add_argument("--test_version", type=int, default=1)
    parser.add_argument("--poly_power", type=float, default=0.9)
    parser.add_argument("--spacing_3d", type=_parse_float_triplet, default=(1.0, 1.0, 8.0))
    parser.add_argument("--roi_3d", type=_parse_int_triplet, default=(224, 224, 18))
    parser.add_argument("--date_str", type=str, default=None, help="Date tag used by trained segmentation checkpoint folder.")
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--no_metrics", action="store_true")
    parser.add_argument("--do_ensemble", action="store_true", default=True)
    parser.add_argument("--emidec_pred_dir", type=str, default=None)
    parser.add_argument("--imperial_pred_dir", type=str, default=None)

    # ------------------------------------------------------------------
    # Classification arguments
    # ------------------------------------------------------------------
    parser.add_argument("--images_dir_emidec", type=str, default=None)
    parser.add_argument("--images_dir_imperial", type=str, default=None)
    parser.add_argument("--labels_csv_emidec", type=str, default=None)
    parser.add_argument("--labels_csv_imperial", type=str, default=None)
    parser.add_argument("--infarct_dir_emidec", type=str, default=None, help="Optional 2D infarct mask directory for CAM/Dice analysis.")
    parser.add_argument("--infarct_dir_imperial", type=str, default=None, help="Optional 2D infarct mask directory for CAM/Dice analysis.")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory for classification results.")
    parser.add_argument(
        "--classif_model",
        type=str,
        default="densenet121",
        choices=[
            "densenet121", "densenet169", "densenet201", "densenet264",
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
        ],
        help="Classification architecture. ResNet variants are supported for 2D classification only.",
    )
    parser.add_argument("--spatial_size_2d", type=_parse_int_pair, default=(224, 224))
    parser.add_argument("--spatial_size_3d", type=_parse_int_triplet, default=(224, 224, 32))
    parser.add_argument("--classif_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--enable_cams", action="store_true", help="Enable 2D GradCAM/GuidedGradCAM outputs.")
    parser.add_argument("--cam_thresh", type=float, default=0.3)
    parser.add_argument("--topk_dice_figs", type=int, default=30)
    parser.add_argument("--strict_label_match", action="store_true", default=True)
    parser.add_argument("--save_per_fold", action="store_true", default=True)

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    if args.stage == "data_prep":
        if args.root_dir is None:
            parser.error("data_prep requires --root_dir")
        print("[Stage] Dataset preparation")
        DatasetPreparer_DataSplit(root_dir=args.root_dir).prepare()
        return

    # ------------------------------------------------------------------
    # Segmentation training/evaluation
    # ------------------------------------------------------------------
    if args.stage in {"seg_train", "seg_eval"}:
        if args.root_dir is None:
            parser.error(f"{args.stage} requires --root_dir")

        json_path = args.json_path or str(Path(args.root_dir) / "repository" / "dataset_config_train_test.json")
        seg_mode = args.mode

        if args.stage == "seg_train":
            print(f"[Stage] Segmentation training | mode={seg_mode}")

            pred_getter = None
            if seg_mode == "Cascaded3D":
                if not args.emidec_pred_dir or not args.imperial_pred_dir:
                    parser.error("Cascaded3D training requires --emidec_pred_dir and --imperial_pred_dir")

                def pred_path_getter(fname: str, data_dict):
                    p1 = Path(args.emidec_pred_dir) / fname
                    p2 = Path(args.imperial_pred_dir) / fname
                    if p1.exists():
                        return str(p1)
                    if p2.exists():
                        return str(p2)
                    return str(p1)

                pred_getter = pred_path_getter

            trainer = SegmentationTrainer(
                root_dir=args.root_dir,
                json_path=json_path,
                mode=seg_mode,
                fold_to_run=args.fold,
                n_folds=args.n_folds,
                num_res_units=args.num_res_units,
                max_epochs=args.epochs,
                train_batch_size=args.train_bs,
                val_batch_size=args.val_bs,
                num_workers=args.num_workers,
                spacing_3d=args.spacing_3d,
                roi_3d=args.roi_3d,
                poly_power=args.poly_power,
                pred_path_getter=pred_getter,
                test_version=args.test_version,
            )
            trainer.run()
            return

        if args.stage == "seg_eval":
            if args.date_str is None:
                parser.error("seg_eval requires --date_str matching the trained model folder naming")

            print(f"[Stage] Segmentation inference/evaluation | mode={seg_mode}")
            cascaded_pred_root = None
            if seg_mode == "Cascaded3D":
                if args.emidec_pred_dir is None and args.imperial_pred_dir is None:
                    parser.error("Cascaded3D evaluation requires at least one predicted-mask directory")
                cascaded_pred_root = args.emidec_pred_dir or args.imperial_pred_dir

            run_tag = args.run_tag or f"lge_{seg_mode.lower()}_unet_{args.date_str}"
            run_segmentation_inference(
                root_dir=args.root_dir,
                json_path=json_path,
                mode=seg_mode,
                n_folds=args.n_folds,
                num_res_units=args.num_res_units,
                date_str=args.date_str,
                test_version=args.test_version,
                run_tag=run_tag,
                no_metrics=args.no_metrics,
                spacing_3d=args.spacing_3d,
                roi_3d=args.roi_3d,
                cascaded_pred_root=cascaded_pred_root,
                do_ensemble=args.do_ensemble,
            )
            return

    # ------------------------------------------------------------------
    # Classification training/evaluation/full workflow
    # ------------------------------------------------------------------
    if args.stage in {"classif_train", "classif_eval", "classif_full"}:
        required = [
            "images_dir_emidec", "images_dir_imperial",
            "labels_csv_emidec", "labels_csv_imperial",
        ]
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"{args.stage} requires: {', '.join('--' + m for m in missing)}")

        classif_mode = {
            "classif_train": "train",
            "classif_eval": "test",
            "classif_full": "full",
        }[args.stage]

        if args.mode == "2D":
            print(f"[Stage] 2D MI classification | model={args.classif_model} | run={classif_mode}")
            _run_classification_2d(args, classif_mode)
            return

        if args.mode == "3D":
            if args.classif_model.startswith("resnet"):
                parser.error("3D classification currently supports DenseNet variants only. Use --mode 2D for ResNet models.")
            print(f"[Stage] 3D MI classification | model={args.classif_model} | run={classif_mode}")
            _run_classification_3d(args, classif_mode)
            return

        parser.error("Classification supports --mode 2D or --mode 3D only.")

    parser.error("Unknown stage")


if __name__ == "__main__":
    main()
