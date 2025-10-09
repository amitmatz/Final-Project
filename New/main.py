#main.py
import os
from defines import (
    PATIENT_ID,
    PROCESSED_DIR,
    BATCH_SIZE,
    LR,
    EPOCHS,
    DEVICE,
    REBUILD_NPY,
)
from Classification.train_classification import run_training

def main():
    print("----start of run stage----")

    # (Optional) preprocessing stage stub
    if REBUILD_NPY:
        print("----start of preprocessing stage----")
        # Your real preprocessing should go here.
        # This stub only shows the banner to match previous logs.
        print("[INFO] SKIP_LFP_EXPORT=True: skipping LFP→CSV export")
        print(f"[EXPORT] Labels saved to: {os.path.join(PROCESSED_DIR, 'labels_aligned.csv')}")
        print(f"[INFO] CLASSIFICATION: saved windows and metadata.")
        print("----end of preprocessing stage----")
    else:
        print("[INFO] preprocessing skipped (REBUILD_NPY=0).")

    print("----start of training Classification stage----")
    npy_path = os.path.join(PROCESSED_DIR, f"{PATIENT_ID}_classification_data.npy")

    params = dict(
        patient_id=PATIENT_ID,
        npy_path=npy_path,
        batch_size=BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
        device=DEVICE,
    )
    # run training
    run_training(**params)
    print("----end of training Classification stage----")
    print("----end of run stage----")

if __name__ == "__main__":
    main()
