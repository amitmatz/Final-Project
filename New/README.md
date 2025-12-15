# Final Project – Decoding Speech from LFP Signals using Deep Learning

פרויקט גמר: פענוח אותות מוחיים (LFP) הקשורים לדיבור באמצעות למידה עמוקה.  
המערכת כוללת שלבי **preprocessing** (עיבוד נתונים גולמיים) ו־**classification** (סיווג איזו מילה נאמרה), עם אימון באמצעות **Cross-Validation מחזורי (Cyclic CV)**.

---

## Project Structure

project_root/
│
├── config/  
│   └── patients_config.json          # הגדרות לכל פציינט: נתיבים יחסיים, פרמטרים, יחס OTHER  
│
├── Data/                             # נתוני הקלט (LFP + Labels) – מקומי / מסונכרן  
│
├── preprocessing/  
│   └── preprocessing.py              # עיבוד raw → windows ושמירת processed_data  
│
├── Classification/  
│   ├── model.py                      # מודל LSTMClassifier (משמש כברירת מחדל אם קיים)  
│   └── train_classification.py       # אימון מודל + CV + מדדים + Confusion Matrix  
│
├── processed_data/                   # פלט preprocessing: קבצי npy/json לכל פציינט  
│   ├── {patient}_classification_data.npy  
│   └── {patient}_classification_data_splits.json  
│
├── defines.py                        # קבועים + גילוי בסיסי דאטה (כולל Google Drive)  
├── main.py                           # Pipeline: preprocessing (אם צריך) + training  
├── requirements.txt                  # חבילות Python  
└── README.md                         # תיעוד הפרויקט

---

## Environment Setup

1. מומלץ לעבוד עם `venv` או `conda`.
2. התקנת תלותים:

```bash
pip install -r requirements.txt



