# Final-Project

## Introduction
This project decodes speech from Local Field Potential (LFP) brain signals using deep learning techniques.

---

## Project Structure

```
├──Final-Project/            
    ├── Data/                     # Synchronized data from Google Drive
    │
    ├── config/ 
    │   └── patients_config.json  # הגדרת הפציינטים 
    │
    ├── preprocessing/            # עיבוד מקדים לנתונים
    │   └── preprocessing.py
    │
    ├── Detection/                # זיהוי נוכחות דיבור
    │   └── train_detection.py       # מחברת לזיהוי אם יש ניסיון דיבור בכלל
    │
    ├── classification/           # סיווג ההברות / מילים
    │   ├── model/
    │   │   └── lstm_model.py     # הגדרת רשת LSTM
    │   ├── train.py              # אימון המודל
    │   ├── validate.py           # בדיקה על סט וולידציה
    │   ├── param_optimizer.py    # אופטימיזציה של היפר-פרמטרים עם Optuna
    │   └── utils.py              # פונקציות עזר כלליות
    ├── defines.py                # משתנים גלובליים והגדרות
    ├── main.py                   # קובץ ראשי שמנהל את התהליך
    ├── README.md                 # הסבר על הפרויקט + איך להריץ
    └── requirements.txt          # רשימת ספריות Python
```
---
## Data Organization
```
├── Data/                     # Synchronized data from Google Drive
│   ├── Patient_01/
│   │   ├── LFP_signals/
│   │   │   ├── CSC1.mat
│   │   │   ├── CSC2.mat
│   │   │   ├── ...
│   │   │   └── CSC40.mat                
│   │   └── Labels.txt        # תוויות מילות דיבור
│   ├── Patient_02/
│   └── ... 
```
Each patient folder contains 40 LFP `.mat` files and a `Labels.txt` file.

---

## Setup

1. Install Python libraries:

```bash
pip install -r requirements.txt