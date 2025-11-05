# main.py
import argparse
from pathlib import Path
from typing import List, Dict, Any
from inspect import signature

from preprocessing.preprocessing import process_patient, PreprocConfig
from Classification.train_classification import TrainConfig, train


# ---------- small utils ----------

def parse_tau_grid(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return [0.0]
    parts = [p for p in s.split(",") if p != ""]
    return [float(p) for p in parts]


def filter_kwargs_for_ctor(ctor, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only kwargs that actually appear in the constructor's signature.
    If anything goes wrong, return {} so the caller can try zero-arg construction.
    """
    try:
        params = signature(ctor).parameters
        return {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        return {}


def try_construct(ctor, **kwargs):
    """
    Try to construct an object from `ctor` with only the supported kwargs.
    If that fails, try zero-arg construction. If that also fails, re-raise.
    """
    filt = filter_kwargs_for_ctor(ctor, kwargs)
    try:
        return ctor(**filt)
    except Exception:
        try:
            return ctor()  # maybe it's a no-arg dataclass or simple object
        except Exception as e:
            raise e


def safe_process_patient(pre_cfg, patient_id: str, force: bool):
    """
    Call process_patient with whatever signature exists locally.
    Tries common signatures to stay compatible.
    """
    print(f"[INFO] Running preprocessing for {patient_id} ...")

    # (patient_id, pre_cfg, force=...)
    try:
        return process_patient(patient_id, pre_cfg, force=force)
    except TypeError:
        pass
    # (pre_cfg, patient_id, force=...)
    try:
        return process_patient(pre_cfg, patient_id, force=force)
    except TypeError:
        pass
    # (patient_id, pre_cfg)
    try:
        return process_patient(patient_id, pre_cfg)
    except TypeError:
        pass
    # (pre_cfg) only
    return process_patient(pre_cfg)


# ---------- CLI ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # required
    p.add_argument("--patient_id", required=True, help="e.g., Patient_03")

    # device
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    # preprocessing (נעשה פילטרינג דינמי בהמשך)
    p.add_argument("--force_preproc", action="store_true")
    p.add_argument("--apply_notch50", action="store_true")
    p.add_argument("--bp_low_hz", type=int, default=1)
    p.add_argument("--bp_high_hz", type=int, default=150)
    p.add_argument("--window_ms", type=int, default=1000)
    p.add_argument("--other_margin_ms", type=int, default=300)
    p.add_argument("--other_ratio", type=float, default=1.0)

    # training hparams
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", choices=["cosine", "plateau"], default="cosine")

    # losses & balancing
    p.add_argument("--use_focal", action="store_true")
    p.add_argument("--gamma_focal", type=float, default=1.5)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--cb_beta", type=float, default=None)  # preferred name
    p.add_argument("--beta_cb", type=float, default=None)  # alias

    # τ grid
    p.add_argument("--tau_grid", type=str, default="0,0.25,0.5,0.75,1.0")

    # model
    p.add_argument("--lstm_layers", type=int, default=2)
    p.add_argument("--use_dilation", action="store_true")
    p.add_argument("--conv_norm", choices=["batch", "group"], default="group")
    p.add_argument("--k1", type=int, default=17)
    p.add_argument("--k2", type=int, default=5)
    p.add_argument("--use_mha", action="store_true")  # in case your train supports it

    # regularization & augmentation
    p.add_argument("--mixup_alpha", type=float, default=0.0)
    p.add_argument("--time_mask_prob", type=float, default=0.0)
    p.add_argument("--time_mask_max_ms", type=int, default=120,
                   help="max time-mask width in milliseconds (fs≈2000Hz)")
    p.add_argument("--time_mask_max_frac", type=float, default=None)  # legacy, ignored if ms provided
    p.add_argument("--channel_drop_prob", type=float, default=None)
    p.add_argument("--chan_drop_prob", type=float, default=None)  # alias

    # exact balanced single batch per epoch
    p.add_argument("--balanced_one_batch", action="store_true")

    # CV: always on, default 5 folds
    p.add_argument("--cv_folds", type=int, default=5)

    return p


def main():
    args = build_parser().parse_args()

    # ---------- Preprocessing config (robust to signature changes) ----------
    pre_cfg_kwargs = dict(
        apply_notch50=args.apply_notch50,
        bp_low_hz=args.bp_low_hz,
        bp_high_hz=args.bp_high_hz,
        window_ms=args.window_ms,
        other_margin_ms=args.other_margin_ms,
        other_ratio=args.other_ratio,
        patient_id=args.patient_id,  # in case your PreprocConfig supports it
    )
    pre_cfg = try_construct(PreprocConfig, **pre_cfg_kwargs)

    processed_dir = Path("processed_data")
    data_path = processed_dir / f"{args.patient_id}_classification_data.npy"

    if args.force_preproc or not data_path.exists():
        safe_process_patient(pre_cfg, args.patient_id, force=args.force_preproc)
    else:
        print(f"[INFO] Found existing data at: {data_path} (skip preprocessing)")

    # ---------- Train config (robust to signature changes) ----------
    # handle alias for class-balanced beta
    cb_beta_val = args.cb_beta if args.cb_beta is not None else (
        args.beta_cb if args.beta_cb is not None else 0.9999
    )
    # handle alias for channel drop prob
    ch_drop = args.channel_drop_prob if args.channel_drop_prob is not None else (
        args.chan_drop_prob if args.chan_drop_prob is not None else 0.0
    )
    tau_grid = parse_tau_grid(args.tau_grid)

    train_cfg_kwargs = dict(
        patient_id=args.patient_id,
        device=args.device,

        # training
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,

        # model
        lstm_layers=args.lstm_layers,
        conv_norm=args.conv_norm,
        k1=args.k1,
        k2=args.k2,
        use_dilation=args.use_dilation,
        use_mha=args.use_mha,

        # loss
        use_focal=args.use_focal,
        gamma_focal=args.gamma_focal,
        label_smoothing=args.label_smoothing,
        cb_beta=cb_beta_val,

        # batch policy
        balanced_one_batch=args.balanced_one_batch,

        # augmentation / regularization
        time_mask_prob=args.time_mask_prob,
        time_mask_max_ms=args.time_mask_max_ms,
        time_mask_max_frac=args.time_mask_max_frac,
        channel_drop_prob=ch_drop,
        mixup_alpha=args.mixup_alpha,

        # τ / CV
        tau_grid=tau_grid,
        cv_folds=args.cv_folds,
    )
    tr_cfg = try_construct(TrainConfig, **train_cfg_kwargs)

    # ---------- Train (includes CV inside your train()) ----------
    train(tr_cfg)


if __name__ == "__main__":
    main()
