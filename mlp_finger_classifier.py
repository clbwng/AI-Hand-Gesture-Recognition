import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ===========================================================
# CONFIG
# ===========================================================
CSV_PATH = "landmarks_reduced.csv"
LABELS = [0,1,2,3,4,5]

HIDDEN_SIZES = [(16,), (32,), (64,), (64,32), (128,64)]
ALPHAS = [1e-4, 5e-4, 1e-3]
LEARNING_RATES = [5e-4, 1e-3, 5e-3]

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
# STEP 1 — Train / Validation / Test split (60 / 20 / 20)
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
best_config = None
best_val_acc = -1

for hs in HIDDEN_SIZES:
    for alpha in ALPHAS:
        for lr in LEARNING_RATES:

            model = MLPClassifier(
                hidden_layer_sizes=hs,
                activation="relu",
                solver="adam",
                alpha=alpha,
                learning_rate_init=lr,
                max_iter=400,
                batch_size=32,
                shuffle=True,
                random_state=RANDOM_SEED,
                early_stopping=True,
                n_iter_no_change=10
            )

            model.fit(X_train_s, y_train)
            y_val_pred = model.predict(X_val_s)
            acc = accuracy_score(y_val, y_val_pred)

            sweep_results.append({
                "hidden": str(hs),
                "alpha": alpha,
                "lr": lr,
                "val_accuracy": acc
            })

            if acc > best_val_acc:
                best_val_acc = acc
                best_config = (hs, alpha, lr)

sweep_df = pd.DataFrame(sweep_results)

print("\nBest MLP config from validation:")
print(f"Hidden={best_config[0]}, Alpha={best_config[1]}, LR={best_config[2]}")

# ===========================================================
# CONTOURF — Alpha × Learning Rate (fixed hidden size)
# ===========================================================
fixed_hidden = best_config[0]
accuracy_grid = np.zeros((len(ALPHAS), len(LEARNING_RATES)))

for i, alpha in enumerate(ALPHAS):
    for j, lr in enumerate(LEARNING_RATES):
        row = sweep_df[
            (sweep_df["hidden"] == str(fixed_hidden)) &
            (sweep_df["alpha"] == alpha) &
            (sweep_df["lr"] == lr)
        ]
        accuracy_grid[i, j] = row["val_accuracy"].values[0]

XX, YY = np.meshgrid(
    np.arange(len(LEARNING_RATES)),
    np.arange(len(ALPHAS))
)

plt.figure(figsize=(8,6))
contour = plt.contourf(
    XX, YY, accuracy_grid,
    levels=40, cmap="viridis"
)
plt.colorbar(contour, label="Validation Accuracy")

CS = plt.contour(
    XX, YY, accuracy_grid,
    levels=[0.7, 0.8, 0.85, 0.9, 0.95],
    colors="black"
)
plt.clabel(CS, inline=True, fontsize=9, fmt="%.2f")

plt.xticks(
    np.arange(len(LEARNING_RATES)),
    [f"{lr:.1e}" for lr in LEARNING_RATES],
    rotation=45
)
plt.yticks(
    np.arange(len(ALPHAS)),
    [f"{a:.1e}" for a in ALPHAS]
)

plt.xlabel("Learning Rate")
plt.ylabel("Alpha (L2 Regularization)")
plt.title(f"MLP Validation Accuracy Surface\nHidden Layers = {fixed_hidden}")
plt.tight_layout()
plt.show()

# ===========================================================
# STEP 3 — Final model (TRAIN + VAL → TEST)
# ===========================================================
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.hstack([y_train, y_val])

scaler = StandardScaler()
X_trainval_s = scaler.fit_transform(X_trainval)
X_test_s     = scaler.transform(X_test)

final_model = MLPClassifier(
    hidden_layer_sizes=best_config[0],
    alpha=best_config[1],
    learning_rate_init=best_config[2],
    activation="relu",
    solver="adam",
    max_iter=400,
    batch_size=32,
    shuffle=True,
    random_state=RANDOM_SEED,
    early_stopping=True,
    n_iter_no_change=10
)

final_model.fit(X_trainval_s, y_trainval)
y_test_pred = final_model.predict(X_test_s)

acc  = accuracy_score(y_test, y_test_pred)
f1   = f1_score(y_test, y_test_pred, average="macro")
prec = precision_score(y_test, y_test_pred, average="macro")
rec  = recall_score(y_test, y_test_pred, average="macro")

print("\n===== FINAL TEST METRICS (MLP) =====")
print(f"Accuracy:        {acc:.4f}")
print(f"Macro F1:        {f1:.4f}")
print(f"Macro Precision:{prec:.4f}")
print(f"Macro Recall:   {rec:.4f}")

# ---------------- Confusion Matrix ----------------
cm = confusion_matrix(y_test, y_test_pred, labels=LABELS)
cm_percent = cm.astype(float) / cm.sum(axis=1)[:, None] * 100

disp = ConfusionMatrixDisplay(
    confusion_matrix=np.round(cm_percent, 1),
    display_labels=LABELS
)

disp.plot(cmap="Blues", values_format=".1f")
plt.title("MLP Confusion Matrix (%) — Test Set")
plt.show()

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

    model = MLPClassifier(
        hidden_layer_sizes=best_config[0],
        alpha=best_config[1],
        learning_rate_init=best_config[2],
        activation="relu",
        solver="adam",
        max_iter=400,
        batch_size=32,
        shuffle=True,
        random_state=RANDOM_SEED,
        early_stopping=True,
        n_iter_no_change=10
    )

    model.fit(X_tr, y_tr)
    y_va_pred = model.predict(X_va)

    cv_accuracies.append(accuracy_score(y_va, y_va_pred))

cv_accuracies = np.array(cv_accuracies)
mean_acc = cv_accuracies.mean()
std_acc  = cv_accuracies.std()
ci_95 = 1.96 * std_acc

print("\n===== 10-FOLD CV RESULTS (MLP) =====")
print(f"Mean Accuracy: {mean_acc:.4f}")
print(f"95% CI:        [{mean_acc-ci_95:.4f}, {mean_acc+ci_95:.4f}]")

# ===========================================================
# CI Bar Plot
# ===========================================================
plt.figure(figsize=(5,5))

plt.bar(
    ["MLP"],
    [mean_acc],
    yerr=[ci_95],
    capsize=8,
    color="#2196F3",
    alpha=0.8
)

y_max = max(1.0, mean_acc + ci_95 + 0.05)
plt.ylim(0, y_max)

plt.ylabel("Accuracy")
plt.title("MLP 10-Fold CV Accuracy\n(Mean ± 95% CI)")
plt.grid(axis="y", linestyle="--", alpha=0.5)

legend_elements = [
    Patch(facecolor="#2196F3", edgecolor="black", label="Mean Accuracy"),
    Line2D([0], [0], color="black", linewidth=2, label="95% Confidence Interval")
]

plt.legend(handles=legend_elements)
plt.show()
