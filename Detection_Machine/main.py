# main.py
# Command-line flags:
#   --patient_id <ID>          : Choose which patient to run on (default: defines.PATIENT_ID)
#   --force_preproc            : Re-run preprocessing ONLY for this patient
#                                (deletes PROCESSED_DATA_DIR/<patient_id> and rebuilds it)
#   --skip_preproc             : Do not run preprocessing (even if this is the first run)
#   --skip_train_detection     : Skip detection training stage
#   --FE                       : Enable feature engineering (band-pass + smoothing) in detection model
#   --fe_low <float>           : Low cutoff frequency (Hz) for FE band-pass (used only with --FE)
#   --fe_high <float>          : High cutoff frequency (Hz) for FE band-pass (used only with --FE)
#   --use_focal                : Use Focal Loss instead of standard BCE loss
#   --threshold_metric {f1,accuracy}
#                              : Metric for choosing decision threshold on validation set (default: accuracy)
#   --smooth_k <int>           : Window size (in windows) for post-hoc smoothing on test predictions;
#                                1 = no smoothing (default: 1)

import argparse
import json
import os
import shutil

import Final_Project.Detection_machine.Detection.train_detection
from Final_Project.Detection_machine.preprocessing.preprocessing import process_all_patients
from Final_Project.Detection_machine.defines import PATIENTS_CONFIG_PATH, PROCESSED_DATA_DIR, PATIENT_ID as DEFAULT_PATIENT_ID


def main():
    parser = argparse.ArgumentParser(description="Final Project pipeline runner")

    parser.add_argument(
        "--patient_id",
        type=str,
        default=None,
        help="Patient ID to use (default from defines.PATIENT_ID)"
    )
    parser.add_argument(
        "--force_preproc",
        action="store_true",
        help="Force re-run preprocessing ONLY for the selected patient"
    )
    parser.add_argument(
        "--skip_preproc",
        action="store_true",
        help="Skip preprocessing stage entirely (even on first run)"
    )
    parser.add_argument(
        "--skip_train_detection",
        action="store_true",
        help="Skip detection training stage"
    )
    parser.add_argument(
        "--FE",
        dest="use_fe",
        action="store_true",
        help="Enable feature engineering (band-pass + smoothing) in detection model"
    )
    parser.add_argument(
        "--fe_low",
        type=float,
        default=1.0,
        help="Low cut frequency for FE band-pass (Hz), used with --FE (default: 1.0)"
    )
    parser.add_argument(
        "--fe_high",
        type=float,
        default=150.0,
        help="High cut frequency for FE band-pass (Hz), used with --FE (default: 150.0)"
    )
    parser.add_argument(
        "--use_focal",
        action="store_true",
        help="Use Focal Loss instead of BCEWithLogitsLoss (default: off)"
    )
    parser.add_argument(
        "--threshold_metric",
        type=str,
        choices=["f1", "accuracy"],
        default="accuracy",
        help="Metric for selecting decision threshold on validation set (default: accuracy)"
    )
    parser.add_argument(
        "--smooth_k",
        type=int,
        default=1,
        help="Smoothing window size (number of windows) for test predictions; 1 = no smoothing (default: 1)"
    )

    args = parser.parse_args()

    patient_id = args.patient_id or DEFAULT_PATIENT_ID
    print(f"[INFO] Using patient: {patient_id}")
    print("----start of run stage----")

    # ============================
    # Preprocessing stage
    # ============================
    if args.force_preproc:
        # Remove ONLY this patient's processed_data subdir
        patient_proc_dir = os.path.join(PROCESSED_DATA_DIR, patient_id)
        if os.path.exists(patient_proc_dir):
            print(f"[clean] removing previous processed_data for {patient_id}: {patient_proc_dir}")
            shutil.rmtree(patient_proc_dir, ignore_errors=True)
        else:
            print(f"[clean] no existing processed_data for {patient_id} at {patient_proc_dir}")

        print("----start of preprocessing stage----")
        # Process ONLY this patient
        process_all_patients(patient_id=patient_id)
        print("----end of preprocessing stage----")
    else:
        if args.skip_preproc:
            print("[INFO] Skipping preprocessing stage (per --skip_preproc).")
        else:
            # Do NOT run preprocessing automatically anymore.
            # If data for this patient is missing, training will raise a clear error.
            print("[INFO] Preprocessing stage not run (no --force_preproc).")
            print("       If you need to regenerate processed data for this patient,")
            print("       run again with: --force_preproc")

    # ============================
    # Detection training stage
    # ============================
    if not args.skip_train_detection:
        print("----start of training detection stage----")
        with open(PATIENTS_CONFIG_PATH, "r") as f:
            config = json.load(f)

        Final_Project.Detection_machine.Detection.train_detection.train(
            config,
            patient_id=patient_id,
            use_fe=args.use_fe,
            fe_lowcut=args.fe_low,
            fe_highcut=args.fe_high,
            use_focal=args.use_focal,
            threshold_metric=args.threshold_metric,
            smooth_k=args.smooth_k,
        )
        print("----end of training detection stage----")
    else:
        print("[INFO] Skipping training detection stage (per --skip_train_detection).")

    print("----end of run stage----")


if __name__ == "__main__":
    main()
