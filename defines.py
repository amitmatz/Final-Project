import os

# נתיב הבסיס של הפרויקט - תמיד יחושב לפי מיקום הקובץ הנוכחי
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# נתיב לקבצים הגולמיים
BASE_DATA_PATH = "G:/My Drive/FinalProject/Data/"

# נתיב לקובץ הקונפיגורציה של הפציינטים
PATIENTS_CONFIG_PATH = os.path.join("config", "patients_config.json")

# היכן לשמור קבצי פלט של דאטה מעובד - נתיב מוחלט
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")

# נתיב מלא לקובץ המעובד עבור Patient_01 (לשימוש אם תרצה גישה ישירה)
PROCESSED_PATH = os.path.join(PROCESSED_DATA_DIR, "Patient_01_data.npy")
