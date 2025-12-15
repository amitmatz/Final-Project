import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix values as fractions (not percents)
# Columns: [Pred = Speech, Pred = No-Speech]
# Rows:    [True Speech, True No-Speech]
cm = np.array([
    [0.9081, 0.0919],  # True Speech  -> TP, FN
    [0.3422, 0.6578],  # True No-Speech -> FP, TN
])

# Axis labels
x_labels = ["Speech", "No-Speech"]          # Predicted
y_labels = ["True Speech", "True No-Speech"]  # True

# Build annotation strings with TP / FN / FP / TN + percent
cm_percent = cm * 100.0
annot_labels = np.empty_like(cm_percent, dtype=object)
annot_labels[0, 0] = f"(TP) {cm_percent[0, 0]:.2f}%"
annot_labels[0, 1] = f"(FN) {cm_percent[0, 1]:.2f}%"
annot_labels[1, 0] = f"(FP) {cm_percent[1, 0]:.2f}%"
annot_labels[1, 1] = f"(TN) {cm_percent[1, 1]:.2f}%"

plt.figure(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=annot_labels,
    fmt="",
    cmap="Blues",
    xticklabels=x_labels,
    yticklabels=y_labels,
    vmin=0.0,
    vmax=1.0,
    cbar=True
)

plt.xlabel("Predicted label", fontsize=11)
plt.ylabel("True label", fontsize=11)
plt.title("Speech / No-Speech confusion matrix", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.show()

# Optional: save to file instead of / בנוסף ל- plt.show()
# plt.savefig("confusion_matrix_speech_detection.png", dpi=300)
