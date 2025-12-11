import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cm = np.array([
    [0.358, 0.259, 0.383],  # True: HAARYE
    [0.148, 0.543, 0.309],  # True: OTHER
    [0.272, 0.333, 0.395],  # True: TUT
])

labels = ["HAARYE", "OTHER", "TUT"]

cm_percent = cm * 100.0
annot_labels = np.array([[f"{v:.1f}%" for v in row] for row in cm_percent])

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=annot_labels,
    fmt="",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    vmin=0.0,
    vmax=1.0,
    cbar=True
)

plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("True \\ Pred", fontsize=12)
plt.title("Confusion matrix (HAARYE / OTHER / TUT)", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.show()
