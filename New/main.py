import argparse
import logging
import subprocess
import sys
from pathlib import Path
import importlib.util
from typing import Tuple  # <<< חשוב לפייתון ישן יותר


# -------------------------
# Paths / constants
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSING_SCRIPT = PROJECT_ROOT / "preprocessing" / "preprocessing.py"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
CLASSIFICATION_DIR = PROJECT_ROOT / "Classification"


# -------------------------
# Logging setup
# -------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )


logger = logging.getLogger(__name__)


# -------------------------
# Preprocessing runner
# -------------------------

def run_preprocessing(patient_id: str, force: bool) -> Tuple[Path, Path]:
    """
    Run preprocessing script for a given patient and return
    (data_path, splits_path).
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    out_data = PROCESSED_DATA_DIR / f"{patient_id}_classification_data.npy"
    out_splits = PROCESSED_DATA_DIR / f"{patient_id}_splits.json"

    cmd = [
        sys.executable,
        str(PREPROCESSING_SCRIPT),
        "--patient_id",
        patient_id,
        "--out_data",
        str(out_data),
        "--out_splits",
        str(out_splits),
    ]
    if force:
        cmd.append("--force")

    logger.info(
        "[INFO] Running preprocessing script: %s",
        " ".join(str(c) for c in cmd),
    )
    subprocess.run(cmd, check=True)

    return out_data, out_splits


# -------------------------
# Dynamic import of classification training
# -------------------------

def import_train_cross_validation():
    """
    Dynamically import train_cross_validation from
    Classification/train_classification.py
    """
    train_script = CLASSIFICATION_DIR / "train_classification.py"
    if not train_script.exists():
        raise FileNotFoundError(
            "Could not find train_classification.py at {}".format(train_script)
        )

    spec = importlib.util.spec_from_file_location(
        "train_classification", str(train_script)
    )
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load train_classification module spec")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    if not hasattr(module, "train_cross_validation"):
        raise AttributeError(
            "train_classification.py does not define 'train_cross_validation'"
        )

    logger.info(
        "[INFO] Imported train_cross_validation from '%s'.",
        str(train_script),
    )
    return module.train_cross_validation


# -------------------------
# Argument parsing
# -------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Main entry point for LFP speech classification experiment."
    )

    parser.add_argument(
        "--patient_id",
        type=str,
        required=True,
        help="Patient ID (e.g., Patient_03)",
    )

    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )

    parser.add_argument(
        "--augment_minor",
        action="store_true",
        help=(
            "If set, augment the minority class in TRAIN by simple feature-space "
            "augmentations."
        ),
    )

    parser.add_argument(
        "--force_preproc",
        action="store_true",
        help="If set, force re-run preprocessing (overwrite existing npy/splits).",
    )

    return parser


# -------------------------
# Main
# -------------------------

def main():
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args()

    # 1) Run preprocessing (or reuse existing) --------------------
    data_path = PROCESSED_DATA_DIR / f"{args.patient_id}_classification_data.npy"
    splits_path = PROCESSED_DATA_DIR / f"{args.patient_id}_splits.json"

    if args.force_preproc or not data_path.exists() or not splits_path.exists():
        data_path, splits_path = run_preprocessing(
            patient_id=args.patient_id,
            force=args.force_preproc,
        )
    else:
        logger.info(
            "[INFO] %s already exists. Use --force_preproc to overwrite.",
            str(data_path),
        )

    # 2) Import training function ---------------------------------
    train_cross_validation = import_train_cross_validation()

    # 3) Run training with cross-validation -----------------------
    logger.info(
        "[INFO] Starting training. data_path=%s",
        str(data_path),
    )

    # שים לב: לא מעבירים batch_size כי הפונקציה לא מקבלת אותו
    train_cross_validation(
        data_path=str(data_path),
        patient_id=args.patient_id,
        cv_folds=args.cv_folds,
        augment_minor=args.augment_minor,
    )


if __name__ == "__main__":
    main()
