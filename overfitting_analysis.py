import matplotlib.pyplot as plt
import numpy as np

# ================================
# Permuted-label results (from your outputs)
# ================================
models = [
    "LogReg (perm)",
    "MLP (no early stop)",
    "MLP (with early stop)",
    "SVM (perm)"
]

train_acc = [
    0.2564,  # Logistic Regression
    0.7552,  # MLP, early_stopping=False
    0.3422,  # MLP, early_stopping=True
    0.1289,  # SVM
]

test_acc = [
    0.1574,  # Logistic Regression
    0.2083,  # MLP, early_stopping=False
    0.2130,  # MLP, early_stopping=True
    0.1389,  # SVM
]

# ================================
# Bar plot
# ================================
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(x - width/2, train_acc, width, label="Train Accuracy", color="#1f77b4")
ax.bar(x + width/2, test_acc,  width, label="Test Accuracy",  color="#ff7f0e")

# Formatting
ax.set_ylabel("Accuracy")
ax.set_title("Permuted-Label Test: Train vs Test Accuracy")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.set_ylim(0, 1.0)
ax.grid(True, axis="y", linestyle="--", alpha=0.5)
ax.legend()

# Annotate bars
for i, (tr, te) in enumerate(zip(train_acc, test_acc)):
    ax.text(i - width/2, tr + 0.02, f"{tr:.2f}", ha="center", fontsize=9)
    ax.text(i + width/2, te + 0.02, f"{te:.2f}", ha="center", fontsize=9)

plt.tight_layout()
plt.show()
