# Final-Project
```
/project_root
│
├── LICENSE                  # אופציונליי
├── README.md                # הסבר על הפרויקט + איך להריץ
├── requirements.txt         # רשימת ספריות Python
│
├── data/                    # נתוני קלט גולמיים (LFP)
│   ├── patient_01/
│   │   ├── lfp_signals.txt   # אותות LFP חתוכים
│   │   └── labels.txt        # תוויות מילות דיבור
│   └── ... (פציינטים נוספים בעתיד)
│
├── preprocessing/           # עיבוד מקדים לנתונים
│   └── preprocessing.py
│
├── detection/               # זיהוי נוכחות דיבור
│   └── detection.ipynb      # מחברת לזיהוי אם יש ניסיון דיבור בכלל
│
├── classification/          # סיווג ההברות / מילים
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
└── main.py                   # קובץ ראשי שמנהל את התהליך
```