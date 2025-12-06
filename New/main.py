import argparse
import importlib.util
import logging
import os
import sys

import numpy as np


LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="INFO: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def import_train_cross_validation():
    """
    Dynamically import Classification/train_classification.py
    and return the train_cross_validation function.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(base_dir, "Classification", "train_classification.py")

    if not os.path.exists(module_path):
        LOGGER.error(
            f"[FATAL] Could not find 'train_classification.py' at: {module_path}"
        )
        raise FileNotFoundError(module_path)

    spec = importlib.util.spec_from_file_location("train_classification", module_path)
    if spec is None or spec.loader is None:
        LOGGER.error("[FATAL] Failed to load spec for train_classification.py")
        raise RuntimeError("Failed to load train_classification module spec")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    if not hasattr(module, "train_cross_validation"):
        LOGGER.error(
            "[FATAL] 'train_classification.py' does not define train_cross_validation()"
        )
        raise AttributeError("train_cross_validation not found in module")

    LOGGER.info(
        f"[INFO] Imported train_cross_validation from '{module_path}'."
    )
    return module.train_cross_validation


def ensure_classification_data(patient_id: str, force_preproc: bool) -> str:
    """
    Ensure that processed_data/{patient_id}_classification_data.npy exists.
    If it exists and force_preproc=False -> use it.
    If it does not exist OR force_preproc=True -> try to call preprocessing,
    or raise a clear error telling the user what to do.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "processed_data")
    os.makedirs(processed_dir, exist_ok=True)

    data_path = os.path.join(
        processed_dir, f"{patient_id}_classification_data.npy"
    )

    if os.path.exists(data_path) and not force_preproc:
        LOGGER.info(
            f"[INFO] {data_path} already exists. Use --force_preproc to overwrite."
        )
        return data_path

    # If we reach here, either the file is missing or the user requested re-generation.
    LOGGER.info(
        f"[INFO] (Re)generating classification data for patient '{patient_id}'."
    )

    try:
        # You can adapt this import/function name to match your actual preprocessing API.
        from preprocessing.preprocessing import run_classification_preprocessing
    except ImportError:
        LOGGER.error(
            "[FATAL] Could not import 'run_classification_preprocessing' from "
            "'preprocessing.preprocessing'. Please regenerate the "
            f"{patient_id}_classification_data.npy file manually."
        )
        raise

    # Call the preprocessing function
    run_classification_preprocessing(patient_id=patient_id)

    if not os.path.exists(data_path):
        LOGGER.error(
            f"[FATAL] After preprocessing, classification file was not found: {data_path}"
        )
        raise FileNotFoundError(data_path)

    return data_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--patient_id",
        required=True,
        help="Patient identifier, e.g. 'Patient_03'.",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=6,
        help="Number of cyclic cross-validation folds (default: 6).",
    )
    parser.add_argument(
        "--force_preproc",
        action="store_true",
        help=(
            "Force re-run of preprocessing for the classification data, "
            "overwriting any existing *_classification_data.npy file."
        ),
    )
    parser.add_argument(
        "--augment_minor",
        action="store_true",
        help=(
            "Enable on-the-fly data augmentation for minority classes in TRAIN. "
            "We still keep TRAIN balanced (or class-weighted if --use_class_weights)."
        ),
    )
    # *** IMPORTANT: here is the fix ***
    # Accept BOTH --use_class_weights AND --train_with_class_weights
    parser.add_argument(
        "--use_class_weights",
        "--train_with_class_weights",
        dest="use_class_weights",
        action="store_true",
        help=(
            "Use class-weighted loss instead of (or in addition to) downsampling. "
            "When enabled, all TRAIN samples are kept and CrossEntropyLoss uses "
            "per-class weights. Default (flag off) = perfect class balance via "
            "downsampling (no weights)."
        ),
    )

    parser.add_argument(
        "--val_per_class",
        type=int,
        default=None,
        help=(
            "Number of samples per class to allocate to VAL in each fold. "
            "If not provided, it is inferred from the smallest class and cv_folds."
        ),
    )
    parser.add_argument(
        "--test_per_class",
        type=int,
        default=None,
        help=(
            "Number of samples per class to allocate to TEST in each fold. "
            "In the cyclic scheme we use TEST = previous fold's VAL, so in practice "
            "this should be equal to val_per_class. If not set, it defaults to "
            "val_per_class."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device for training: 'cpu', 'cuda', 'cuda:0', etc. "
            "If not provided, will auto-detect."
        ),
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    LOGGER.info(f"[INFO] Starting training. patient_id={args.patient_id}")

    # Ensure classification data exists (or re-generate if requested).
    data_path = ensure_classification_data(
        patient_id=args.patient_id,
        force_preproc=bool(args.force_preproc),
    )
    LOGGER.info(f"[INFO] Starting training. data_path={data_path}")

    # Dynamically import the classification training function
    train_cross_validation = import_train_cross_validation()

    LOGGER.info(
        f"[INFO] Running cross-validation for patient_id={args.patient_id}"
    )

    # Call the classification cross-validation trainer
    train_cross_validation(
        data_path=data_path,
        patient_id=args.patient_id,
        num_folds=int(args.cv_folds),
        augment_minor=bool(args.augment_minor),
        use_class_weights=bool(args.use_class_weights),
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        device=args.device,
    )


if __name__ == "__main__":
    main()
