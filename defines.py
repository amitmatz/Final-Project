import os

# נתיב לקבצים הגולמיים
BASE_DATA_PATH = "G:/My Drive/FinalProject/Data/"

# נתיב הבסיס של הפרויקט - תמיד יחושב לפי מיקום הקובץ הנוכחי
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# נתיב לקובץ הקונפיגורציה של הפציינטים
PATIENTS_CONFIG_PATH = os.path.join(BASE_DIR,"config", "patients_config.json")

# Choose which patient to process (used for building processed data directory path)
PATIENT_ID = "Patient_01"  # ניתן לשנות לפי המטופל הרצוי

# # Directory for processed data of the chosen patient
# PROCESSED_DATA_DIR = os.path.join(BASE_DATA_PATH, PATIENT_ID, "Processed")

# היכן לשמור קבצי פלט של דאטה מעובד - נתיב מוחלט
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")


# נתיבים לקבצים
OFFSET_FILE = r'G:/My Drive/FinalProject/Data/Patient_01/sound_w_times.mat'
LABELS_FILE = r'G:/My Drive/FinalProject/Data/Patient_01/Labels/Detection_Labels/Patient1_Labels_Detection.txt'

# # Define label names and their numeric representation for speech detection
# SPEECH_LABEL = "Speech"
# NO_SPEECH_LABEL = "NoSpeech"
# SPEECH_LABEL_ID = 1
# NO_SPEECH_LABEL_ID = 0
#
# # Mapping from label string to numeric value
# LABEL_TO_INT = {
#     SPEECH_LABEL: SPEECH_LABEL_ID,
#     NO_SPEECH_LABEL: NO_SPEECH_LABEL_ID
# }

