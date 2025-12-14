import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def remove_spines(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)

# ===========================================================
# CONFIG
# ===========================================================
CSV_PATH = "landmarks_reduced.csv"
LABELS = [0,1,2,3,4,5]
C_VALUES = [0.0001, 0.001, 0.1, 1.0, 10.0, 100.0]
K_FOLDS = 10
RANDOM_SEED = 42

# ===========================================================
# Load dataset
# ===========================================================
df = pd.read_csv(CSV_PATH)
df = df[df["label_fingers"].isin(LABELS)]

feature_cols = [c for c in df.columns if c.startswith(("x","y","z"))]
X = df[feature_cols].values
y = df["label_fingers"].values

print("\n===== DATASET SUMMARY =====")
print(df["label_fingers"].value_counts().sort_index())

# ===========================================================
# STEP 1 — Train / Validation / Test split
#   60% train, 20% val, 20% test
# ===========================================================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_SEED
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,   # 0.25 × 0.80 = 0.20
    stratify=y_temp,
    random_state=RANDOM_SEED
)

print("\nSplit sizes:")
print("Train:", len(X_train))
print("Val:  ", len(X_val))
print("Test: ", len(X_test))

# ===========================================================
# STEP 2 — Hyperparameter sweep (TRAIN → VAL)
# ===========================================================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

sweep_results = []
best_C = None
best_val_acc = -1

for C in C_VALUES:
    model = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=2000
    )
    model.fit(X_train_s, y_train)

    y_val_pred = model.predict(X_val_s)
    acc = accuracy_score(y_val, y_val_pred)

    sweep_results.append({"C": C, "val_accuracy": acc})

    if acc > best_val_acc:
        best_val_acc = acc
        best_C = C

sweep_df = pd.DataFrame(sweep_results)

# ===========================================================
# Validation Accuracy vs C (Line Plot)
# ===========================================================
fig, ax = plt.subplots(figsize=(7,5))

ax.plot(
    sweep_df["C"],
    sweep_df["val_accuracy"],
    marker="o",
    linestyle="-",
    linewidth=2
)

ax.set_xscale("log")
ax.set_xlabel("Regularization Strength (C)")
ax.set_ylabel("Validation Accuracy")
ax.set_title("Logistic Regression: Validation Accuracy vs C")

ax.grid(True, linestyle="--", alpha=0.5)

# # Remove borders/spines for clean look
# for spine in ax.spines.values():
#     spine.set_visible(False)

plt.tight_layout()
# plt.show()

print(f"\nBest C selected from validation: {best_C:.4g}")

# ===========================================================
# STEP 3 — Final model (TRAIN + VAL → TEST)
# ===========================================================
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.hstack([y_train, y_val])

scaler = StandardScaler()
X_trainval_s = scaler.fit_transform(X_trainval)
X_test_s     = scaler.transform(X_test)

final_model = LogisticRegression(
    C=best_C,
    solver="lbfgs",
    max_iter=2000
)

final_model.fit(X_trainval_s, y_trainval)
y_test_pred = final_model.predict(X_test_s)

acc  = accuracy_score(y_test, y_test_pred)
f1   = f1_score(y_test, y_test_pred, average="macro")
prec = precision_score(y_test, y_test_pred, average="macro")
rec  = recall_score(y_test, y_test_pred, average="macro")

print("\n===== FINAL TEST METRICS =====")
print(f"Accuracy:        {acc:.4f}")
print(f"Macro F1:        {f1:.4f}")
print(f"Macro Precision:{prec:.4f}")
print(f"Macro Recall:   {rec:.4f}")

# ---------------- Confusion Matrix ----------------
cm = confusion_matrix(y_test, y_test_pred, labels=LABELS)
cm_percent = cm.astype(float) / cm.sum(axis=1)[:, None] * 100

fig, ax = plt.subplots(figsize=(6,5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=np.round(cm_percent, 1),
    display_labels=LABELS
)

disp.plot(
    cmap="Blues",
    values_format=".1f",
    ax=ax,
    colorbar=True
)

ax.set_title("Logistic Regression Confusion Matrix (%) — Test Set")

# 🔵 REMOVE BORDER
remove_spines(ax)

plt.tight_layout()
# plt.show()
# plt.show()

# ===========================================================
# STEP 4 — 10-Fold CV (Frozen hyperparameters → CI)
# ===========================================================
skf = StratifiedKFold(
    n_splits=K_FOLDS,
    shuffle=True,
    random_state=RANDOM_SEED
)

cv_accuracies = []

for train_idx, val_idx in skf.split(X, y):
    X_tr, X_va = X[train_idx], X[val_idx]
    y_tr, y_va = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)

    model = LogisticRegression(
        C=best_C,
        solver="lbfgs",
        max_iter=2000
    )

    model.fit(X_tr, y_tr)
    y_va_pred = model.predict(X_va)

    cv_accuracies.append(accuracy_score(y_va, y_va_pred))

cv_accuracies = np.array(cv_accuracies)
mean_acc = cv_accuracies.mean()
std_acc  = cv_accuracies.std()
ci_95 = 1.96 * std_acc

print("\n===== 10-FOLD CV RESULTS =====")
print(f"Mean Accuracy: {mean_acc:.4f}")
print(f"95% CI:        [{mean_acc-ci_95:.4f}, {mean_acc+ci_95:.4f}]")

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ===========================================================
# CI Bar Plot — Mean Accuracy with 95% CI
# ===========================================================
plt.figure(figsize=(5,5))

plt.bar(
    ["Logistic Regression"],
    [mean_acc],
    yerr=[ci_95],
    capsize=8,
    color="#4CAF50",
    alpha=0.8
)

# Add headroom so CI is clearly visible
y_max = max(1.0, mean_acc + ci_95 + 0.05)
plt.ylim(0, y_max)
# plt.tight_layout()
plt.ylabel("Accuracy")
plt.title("10-Fold Cross-Validation Accuracy\n(Mean ± 95% CI)")
plt.grid(axis="y", linestyle="-", alpha=0.5)

# -----------------------------------------------------------
# Legend (proxy artists)
# -----------------------------------------------------------
legend_elements = [
    Patch(facecolor="#4CAF50", edgecolor="black", label="Mean Accuracy"),
    Line2D([0], [0], color="black", linewidth=2, label="95% Confidence Interval")
]

plt.legend(handles=legend_elements, loc="lower right")

plt.show()

def min_class_count(y):
    return pd.Series(y).value_counts().min()

# ===========================================================
# STEP 5 — Dataset Size Sensitivity (Learning Curve)
# ===========================================================
print("\n===== DATASET SIZE SENSITIVITY ANALYSIS =====")

# Percentages of dataset to evaluate

learning_curve_results = []

dataset_fracs = np.linspace(0.01, 1.0, 25)

for frac in dataset_fracs:
    print(f"Using {int(frac*100)}% of dataset")

    # ---------------------------------------
    # Subsample fraction (handle 100%)
    # ---------------------------------------
    if frac < 1.0:
        X_frac, _, y_frac, _ = train_test_split(
            X,
            y,
            train_size=frac,
            stratify=y,
            random_state=RANDOM_SEED
        )
    else:
        X_frac, y_frac = X.copy(), y.copy()

    # ---------------------------------------
    # CHECK: can we stratify further?
    # ---------------------------------------
    if min_class_count(y_frac) < 2:
        print("  ⚠️ Skipping — not enough samples per class")
        continue

    # ---------------------------------------
    # 80 / 20 internal split
    # ---------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_frac,
        y_frac,
        test_size=0.20,
        stratify=y_frac,
        random_state=RANDOM_SEED
    )

    # ---------------------------------------
    # Train + evaluate
    # ---------------------------------------
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    model = LogisticRegression(
        C=best_C,
        solver="lbfgs",
        max_iter=2000
    )

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc = accuracy_score(y_te, y_pred)

    learning_curve_results.append({
        "dataset_fraction": frac,
        "dataset_percent": frac * 100,
        "accuracy": acc,
        "n_samples": len(X_frac)
    })

# Convert to DataFrame
lc_df = pd.DataFrame(learning_curve_results)

# ===========================================================
# Save results to CSV
# ===========================================================
CSV_OUT = "logistic_regression_learning_curve.csv"
lc_df.to_csv(CSV_OUT, index=False)
print(f"\nLearning curve data saved to: {CSV_OUT}")
# ===========================================================


