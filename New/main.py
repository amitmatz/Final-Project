# main.py
import argparse
import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

# --- Logging setup ---
LOG_FORMAT = "%(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- Shared paths ---
from defines import PROCESSED_DATA_DIR

# --- Training module ---
# Make sure the folder name 'Classification' matches your actual folder name.
from Classification.train_classification import TrainConfig, train


def _run_preprocessing(patient_id: str, force: bool = False) -> None:
    """
    Run the preprocessing *script* preprocessing/preprocessing.py as a separate
    Python process, passing all required arguments via the command line.

    This does NOT import any function from preprocessing; it just executes the file.
    """
    # Path to preprocessing script
    root = Path(__file__).parent.resolve()
    script_path = root / "preprocessing" / "preprocessing.py"

    if not script_path.exists():
        logger.info("preprocessing.py not found at %s. Skipping preprocessing.", script_path)
        return

    # Output paths must match what training expects
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_data_path = PROCESSED_DATA_DIR / f"{patient_id}_classification_data.npy"
    out_splits_path = PROCESSED_DATA_DIR / f"{patient_id}_splits.json"

    # Build command line to run the script
    cmd: List[str] = [
        sys.executable,
        str(script_path),
        "--patient_id",
        patient_id,
        "--out_data",
        str(out_data_path),
        "--out_splits",
        str(out_splits_path),
    ]
    if force:
        cmd.append("--force")

    logger.info("Running preprocessing script: %s", " ".join(cmd))

    # If preprocessing fails, raise an error so we see why.
    subprocess.check_call(cmd)


def build_paths(patient_id: str) -> Tuple[str, str]:
    """
    Compute default data/splits paths used by TrainConfig and logs.
    Uses the same PROCESSED_DATA_DIR as defines.py to stay consistent.
    """
    processed_dir = PROCESSED_DATA_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    data_path = processed_dir / f"{patient_id}_classification_data.npy"
    splits_path = processed_dir / f"{patient_id}_splits.json"
    return str(data_path), str(splits_path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Final Project — Classification Trainer")
    p.add_argument("--patient_id", required=True, help="Patient ID, e.g., Patient_03")
    p.add_argument("--force_preproc", action="store_true", help="Force preprocessing before training")
    p.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds (default: 5)")

    # Optional overrides
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)  # mapped to max_epochs
    p.add_argument("--lr", type=float, default=None)
    p.add_argument(
        "--keep_channels",
        type=str,
        default=None,
        help="Comma-separated channel indices to keep, e.g. '0,1,2'. If omitted, use all.",
    )

    return p.parse_args(argv)


def maybe_int_list(csv_str: Optional[str]) -> Optional[List[int]]:
    if not csv_str:
        return None
    out = []
    for tok in csv_str.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(int(tok))
    return out if out else None


def main(argv=None):
    args = parse_args(argv)

    # Build data/splits default paths (must match preprocessing output)
    data_path, splits_path = build_paths(args.patient_id)

    # Optionally run preprocessing as a separate script
    if args.force_preproc:
        _run_preprocessing(args.patient_id, force=True)

    # Sanity checks (not fatal — Train will raise clearer error if missing)
    if not os.path.isfile(data_path):
        logger.warning("Expected data file not found: %s", data_path)
    if not os.path.isfile(splits_path):
        logger.warning("Expected splits file not found: %s", splits_path)

    # Build TrainConfig with ONLY known-safe keys to avoid TypeError on unexpected kwargs
    cfg_kwargs: Dict[str, Any] = {
        "data_path": data_path,
        "splits_path": splits_path,
        "cv_folds": args.cv_folds,
    }

    # Optional overrides (add only if user provided a value)
    if args.batch_size is not None:
        cfg_kwargs["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg_kwargs["max_epochs"] = args.epochs
    if args.lr is not None:
        cfg_kwargs["lr"] = args.lr
    keep_ch = maybe_int_list(args.keep_channels)
    if keep_ch is not None:
        cfg_kwargs["keep_channels"] = keep_ch

    # Construct config
    try:
        tr_cfg = TrainConfig(**cfg_kwargs)
    except TypeError as e:
        # Fallback: try with the absolute minimal set if your TrainConfig is stricter
        logger.warning("TrainConfig(**kwargs) raised %s. Retrying with minimal args.", e)
        tr_cfg = TrainConfig(data_path=data_path, splits_path=splits_path, cv_folds=args.cv_folds)
        # Apply optional fields if attributes exist
        if keep_ch is not None and hasattr(tr_cfg, "keep_channels"):
            setattr(tr_cfg, "keep_channels", keep_ch)
        if args.batch_size is not None and hasattr(tr_cfg, "batch_size"):
            setattr(tr_cfg, "batch_size", args.batch_size)
        if args.epochs is not None and hasattr(tr_cfg, "max_epochs"):
            setattr(tr_cfg, "max_epochs", args.epochs)
        if args.lr is not None and hasattr(tr_cfg, "lr"):
            setattr(tr_cfg, "lr", args.lr)

    # Kick off training
    logger.info("Starting training. data_path=%s", tr_cfg.data_path)
    train(tr_cfg)


if __name__ == "__main__":
    main()
