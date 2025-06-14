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
    ├── detection/                # זיהוי נוכחות דיבור
    │   └── detection.ipynb       # מחברת לזיהוי אם יש ניסיון דיבור בכלל
    │
    ├── classification/           # סיווג ההברות / מילים
    │   ├── model/
    │   │   └── lstm_model.py     # הגדרת רשת LSTM
    │   ├── train.py              # אימון המודל
    │   ├── validate.py           # בדיקה על סט וולידציה
    │   ├── param_optimizer.py    # אופטימיזציה של היפר-פרמטרים עם Optuna
    │   └── utils.py              # פונקציות עזר כלליות
    │
    ├── results/                  # תוצאות האימון והבדיקה
    │   ├── plots/                # גרפים: accuracy, loss, confusion matrix
    │   │   ├── patient_01_accuracy_plot.png
    │   │   ├── patient_01_loss_plot.png
    │   │   └── ...
    │   ├── best_params/          # תצורת פרמטרים מיטבית לכל פציינט
    │   │   ├── patient_01_optuna.json
    │   │   └── ...
    │   ├── patient_results.csv   # טבלת סיכום לכל הפציינטים
    │   └── logs/                 # קבצי לוג של הרצות
    │
    ├── saved_models/             # מודלים מאומנים (.pth)
    │   ├── patient_01_model.pth
    │   └── ...
    │
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