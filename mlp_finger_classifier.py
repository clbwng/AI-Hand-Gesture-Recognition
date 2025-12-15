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

HIDDEN_SIZES = [(64, 32)]                 # fixed for parity with LR
ALPHAS = [1e-6, 1e-4, 1e-2, 1e-1]
LEARNING_RATES = [1e-5, 1e-4, 1e-3, 1e-2]

K_FOLDS = 10
RANDOM_SEED = 42

LEARNING_CURVE_CSV = "mlp_learning_curve.csv"

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
# STEP 1 — Train / Val / Test split (60 / 20 / 20)
# ===========================================================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_SEED
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    stratify=y_temp,
    random_state=RANDOM_SEED
)

# ===========================================================
# STEP 2 — Hyperparameter sweep (TRAIN → VAL)
# ===========================================================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

results = []
best_cfg = None
best_val_acc = -1

for alpha in ALPHAS:
    for lr in LEARNING_RATES:
        model = MLPClassifier(
            hidden_layer_sizes=HIDDEN_SIZES[0],
            alpha=alpha,
            learning_rate_init=lr,
            activation="relu",
            solver="adam",
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=RANDOM_SEED
        )

        model.fit(X_train_s, y_train)
        acc = accuracy_score(y_val, model.predict(X_val_s))

        results.append({"alpha": alpha, "lr": lr, "val_accuracy": acc})

        if acc > best_val_acc:
            best_val_acc = acc
            best_cfg = (alpha, lr)

sweep_df = pd.DataFrame(results)

# ---------------- Heatmap ----------------
pivot = sweep_df.pivot(index="alpha", columns="lr", values="val_accuracy")

plt.figure(figsize=(7,5))
sns.heatmap(
    pivot,
    annot=True,
    fmt=".3f",
    cmap="Blues",
    cbar_kws={"label": "Validation Accuracy"}
)
plt.title("MLP Validation Accuracy (α × Learning Rate)")
plt.xlabel("Learning Rate")
plt.ylabel("Alpha (L2)")
plt.tight_layout()
# plt.show()

print("\nBest MLP hyperparameters:")
print(f"Alpha={best_cfg[0]}, Learning rate={best_cfg[1]}")

# ===========================================================
# STEP 3 — Final model (TRAIN+VAL → TEST)
# ===========================================================
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.hstack([y_train, y_val])

scaler = StandardScaler()
X_trainval_s = scaler.fit_transform(X_trainval)
X_test_s = scaler.transform(X_test)

final_model = MLPClassifier(
    hidden_layer_sizes=HIDDEN_SIZES[0],
    alpha=best_cfg[0],
    learning_rate_init=best_cfg[1],
    activation="relu",
    solver="adam",
    max_iter=400,
    early_stopping=True,
    random_state=RANDOM_SEED
)

final_model.fit(X_trainval_s, y_trainval)
y_test_pred = final_model.predict(X_test_s)

acc = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred, average="macro")
prec = precision_score(y_test, y_test_pred, average="macro")
rec = recall_score(y_test, y_test_pred, average="macro")

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
# plt.show()

# ===========================================================
# STEP 4 — 10-Fold CV (Frozen hyperparameters)
# ===========================================================
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
cv_accs = []

for tr, va in skf.split(X, y):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr])
    X_va = scaler.transform(X[va])

    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_SIZES[0],
        alpha=best_cfg[0],
        learning_rate_init=best_cfg[1],
        max_iter=400,
        early_stopping=True,
        random_state=RANDOM_SEED
    )

    model.fit(X_tr, y[tr])
    cv_accs.append(accuracy_score(y[va], model.predict(X_va)))

cv_accs = np.array(cv_accs)
mean_acc = cv_accs.mean()
ci_95 = 1.96 * cv_accs.std()

print("\n===== 10-FOLD CV RESULTS (MLP) =====")
print(f"Mean Accuracy: {mean_acc:.4f}")
print(f"95% CI: [{mean_acc-ci_95:.4f}, {mean_acc+ci_95:.4f}]")

# ---------------- CI Bar Plot ----------------
plt.figure(figsize=(5,5))
plt.bar(["MLP"], [mean_acc], yerr=[ci_95], capsize=8, color="#2196F3")
plt.ylim(0, max(1.0, mean_acc + ci_95 + 0.05))
plt.ylabel("Accuracy")
plt.title("MLP 10-Fold CV Accuracy\n(Mean ± 95% CI)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

def min_class_count(y):
    return pd.Series(y).value_counts().min()


# ===========================================================
# STEP 5 — Learning Curve (Fixed 20% Test Set, MLP)
# ===========================================================
from sklearn.model_selection import StratifiedShuffleSplit

rows = []

print("\n===== MLP LEARNING CURVE (FIXED TEST SET) =====")

# -----------------------------------------------------------
# FIXED train/test split (80 / 20) — reuse earlier split
# -----------------------------------------------------------
X_train_pool = X_trainval        # 80% pool (train + val)
y_train_pool = y_trainval

n_train_total = len(y_train_pool)

for pct in range(1, 100):  # 1% → 100%
    frac = pct / 100.0
    n_samples = max(1, int(round(frac * n_train_total)))

    # -------------------------------------------------------
    # Stratified subsample from TRAIN POOL ONLY
    # -------------------------------------------------------
    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=n_samples,
        random_state=RANDOM_SEED + pct,
    )

    for idx, _ in sss.split(X_train_pool, y_train_pool):
        X_sub = X_train_pool[idx]
        y_sub = y_train_pool[idx]

    # -------------------------------------------------------
    # Scale (fit on training subset only)
    # -------------------------------------------------------
    scaler = StandardScaler()
    X_sub_s = scaler.fit_transform(X_sub)
    X_test_s = scaler.transform(X_test)

    # -------------------------------------------------------
    # Train MLP with frozen best hyperparameters
    # -------------------------------------------------------
    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_SIZES[0],
        alpha=best_cfg[0],
        learning_rate_init=best_cfg[1],
        activation="relu",
        solver="adam",
        max_iter=400,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )

    model.fit(X_sub_s, y_sub)

    # -------------------------------------------------------
    # Evaluate on FIXED test set
    # -------------------------------------------------------
    acc = accuracy_score(y_test, model.predict(X_test_s))

    rows.append({
        "dataset_percent": float(pct),
        "accuracy": acc,
    })


    print(f"Train % = {pct:3d}% | Samples = {n_samples:4d} | Test Acc = {acc:.4f}")

# -----------------------------------------------------------
# Save CSV (exact schema requested)
# -----------------------------------------------------------
curve_df = pd.DataFrame(rows)
curve_df.to_csv(LEARNING_CURVE_CSV, index=False)

print(f"\nSaved MLP learning curve to: {LEARNING_CURVE_CSV}")

# ===========================================================
# STEP 6 — Permuted Label Sanity Check (MLP)
# ===========================================================
print("\n===== PERMUTED LABEL SANITY CHECK (MLP) =====")

rng = np.random.RandomState(RANDOM_SEED)

# -----------------------------------------------------------
# Permute labels (global permutation)
# -----------------------------------------------------------
y_perm = rng.permutation(y)

# -----------------------------------------------------------
# Same 60 / 20 / 20 split
# -----------------------------------------------------------
X_temp_p, X_test_p, y_temp_p, y_test_p = train_test_split(
    X, y_perm,
    test_size=0.20,
    stratify=y_perm,
    random_state=RANDOM_SEED
)

X_train_p, X_val_p, y_train_p, y_val_p = train_test_split(
    X_temp_p, y_temp_p,
    test_size=0.25,
    stratify=y_temp_p,
    random_state=RANDOM_SEED
)

# -----------------------------------------------------------
# Train on TRAIN + VAL (same as final model)
# -----------------------------------------------------------
X_trainval_p = np.vstack([X_train_p, X_val_p])
y_trainval_p = np.hstack([y_train_p, y_val_p])

scaler = StandardScaler()
X_trainval_p_s = scaler.fit_transform(X_trainval_p)
X_test_p_s     = scaler.transform(X_test_p)

perm_model = MLPClassifier(
    hidden_layer_sizes=HIDDEN_SIZES[0],
    alpha=best_cfg[0],
    learning_rate_init=best_cfg[1],
    activation="relu",
    solver="adam",
    max_iter=400,
    early_stopping=True,
    random_state=RANDOM_SEED
)

perm_model.fit(X_trainval_p_s, y_trainval_p)

# -----------------------------------------------------------
# Evaluate
# -----------------------------------------------------------
y_train_pred_p = perm_model.predict(X_trainval_p_s)
y_test_pred_p  = perm_model.predict(X_test_p_s)

train_acc_p = accuracy_score(y_trainval_p, y_train_pred_p)
test_acc_p  = accuracy_score(y_test_p, y_test_pred_p)

print(f"Training Accuracy (permuted labels): {train_acc_p:.4f}")
print(f"Test Accuracy     (permuted labels): {test_acc_p:.4f}")
print(f"Chance Level Accuracy:              {1.0 / len(LABELS):.4f}")

