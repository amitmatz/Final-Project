# main.py
import json
import Detection.train_detection
from preprocessing.preprocessing import process_all_patients
from defines import PATIENTS_CONFIG_PATH
import sys, torch, platform
SKIP_PREPROCESS_STAGE = False
SKIP_TRAIN_DETECTION_STAGE = True
SKIP_BACKGROUNG_STAGE = True

if __name__ == "__main__":
    print("----start of run stage----")

    if not SKIP_BACKGROUNG_STAGE:
        print("EXE:", sys.executable)
        print("TORCH:", torch.__version__, "| PY:", platform.python_version())
        print("CUDA available:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("GPU name:", torch.cuda.get_device_name(0))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Chosen device:", device)

    if not SKIP_PREPROCESS_STAGE:
        print("----start of preprocessing stage----")
        process_all_patients()
        print("----end of preprocessing stage----")

    if not SKIP_TRAIN_DETECTION_STAGE:
        print("----start of training detection stage----")
        with open(PATIENTS_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        Detection.train_detection.train(config)
        print("----end of training detection stage----")
    print("----end of run stage----")