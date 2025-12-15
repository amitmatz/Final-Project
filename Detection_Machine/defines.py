import os

# Absolute base path for the raw data (where Patient_01 folder is located)
BASE_DATA_PATH = r"G:\האחסון שלי\FinalProject\Data"

# Project base directory (the root of your Final_Project folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the patients configuration JSON file
PATIENTS_CONFIG_PATH = os.path.join(BASE_DIR, "config", "patients_config.json")

# Default patient ID (used if --patient_id is not given)
PATIENT_ID = "Patient_01"

# Directory where all processed data for the project will be stored
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")

# Convenience paths for Patient_01 (optional)
OFFSET_FILE = os.path.join(
    BASE_DATA_PATH,
    "Patient_01",
    "sound_w_times.mat"
)

LABELS_FILE = os.path.join(
    BASE_DATA_PATH,
    "Patient_01",
    "Labels",
    "Detection_Labels",
    "Patient1_Labels_Detection.txt"
)
