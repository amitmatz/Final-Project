# Final Project – Decoding Speech from LFP Signals using Deep Learning

פרויקט גמר: פענוח אותות מוחיים (LFP) הקשורים לדיבור באמצעות למידה עמוקה.  
המערכת כוללת שלבי **preprocessing** (עיבוד נתונים גולמיים) ו־**classification** (סיווג איזו מילה נאמרה).  

---

## Project Structure

project_root/
│
├── config/
│ └── patients_config.json # קובץ הגדרות למטופלים (נתיבים, שמות קבצים, פרמטרים)
│
├── data/
│ ├── raw/ # נתונים מקוריים (אותות LFP, תוויות טקסט)
│ ├── processed/ # פלט מה-preprocessing (.npy לכל פציינט)
│ └── external/ # נתונים חיצוניים לניסוי (אם קיימים)
│
├── preprocessing/
│ └── preprocessing.py # סקריפט לעיבוד raw → processed
│
├── classification/
│ ├── dataset.py # מחלקת Dataset + collate_fn לפיצול ל-train/val/test
│ ├── model.py # מימוש מודלים (BiLSTM / אחרים)
│ ├── train.py # אימון המודל + שמירת checkpoints
│ ├── validate.py # הערכת המודל על test set
│ ├── utils.py # פונקציות עזר (מדדים, גרפים, כלים)
│ ├── results/ # תוצאות אימון (גרפים, מטריצות בלבול)
│ ├── logs/ # לוגים של TensorBoard / הרצות
│ └── saved_models/ # מודלים מאומנים (.pth)
│

├── defines.py # קבועים ונתיבי בסיס לשימוש בכל הסקריפטים
├── main.py # קובץ ראשי להרצת pipeline מקצה לקצה
├── requirements.txt # רשימת חבילות Python (numpy, torch, sklearn, וכו')
└── README.md # תיעוד הפרויקט