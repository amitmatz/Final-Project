classification/
├── data/                      ← קבצי הקלט לאחר detection (LFP + תוויות)
│   ├── matrices_0.pt          ← אותות לפציינט 0
│   └── labels_0.pt            ← תוויות תואמות (0=האריה, 1=אהב, 2=תות)
│
├── plots/                     ← גרפים ותוצרים גרפיים
│   ├── loss_accuracy_p0.png   ← גרף אימון
│   └── confusion_matrix_p0.png← מטריצת בלבול
│
├── model.py                   ← הגדרת המודל LSTM+Attention
├── utils.py                   ← פונקציות עזר: טעינה, אימון, בדיקה, גרפים
└── main.py                    ← קובץ ריצה מרכזי: train + test