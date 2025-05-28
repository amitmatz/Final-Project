classification/
├── data/
│   ├── matrices_X.pt        ← קטעי אותות LFP שזהוו כדיבור בלבד
│   └── labels_X.pt          ← התוויות ("האריה"=0, "אהב"=1, "תות"=2)
│
├── plots/                   ← תמונות של loss, accuracy, confusion matrix
│
├── model.py                 ← מודל LSTM + Attention + Fully Connected
├── utils.py                 ← טעינה, עיבוד נתונים, אימון, בדיקה, גרפים
└── main.py                  ← תסריט ראשי להרצת המודל על פציינטים