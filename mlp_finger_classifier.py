import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import hashlib

# ===========================================================
# Configuration
# ===========================================================
CSV_PATH = "landmarks_reduced.csv"
LABELS = [0,1,2,3,4,5]

BASE_HIDDEN = (64, 32)
BASE_ALPHA  = 5e-4
BASE_LR     = 1e-3

RANDOM_STATE = 42


# ===========================================================
# Helper: shared evaluation metrics
# ===========================================================
def evaluate_model(y_true, y_pred, labels=LABELS):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro")
    rec  = recall_score(y_true, y_pred, average="macro")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_percent = cm.astype(float) / cm.sum(axis=1)[:, None] * 100

    return acc, f1, prec, rec, cm_percent


# ===========================================================
# STEP 1 — Load + Clean Dataset
# ===========================================================
df = pd.read_csv(CSV_PATH)
df = df[df["label_fingers"].isin(LABELS)]

feature_cols = [c for c in df.columns if c.startswith(("x","y","z"))]

def row_hash(row):
    return hashlib.md5(str(tuple(row[feature_cols])).encode()).hexdigest()

df["hash"] = df.apply(row_hash, axis=1)

print("\n===== DATASET SUMMARY =====")
print(df["label_fingers"].value_counts().sort_index())
print("Total samples:", len(df))
print("Unique landmark sets:", df["hash"].nunique(), "\n")

X = df[feature_cols].values
y = df["label_fingers"].values


# ===========================================================
# STEP 2 — Stratified Train/Test Split
# ===========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.80,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Train distribution:", pd.Series(y_train).value_counts().sort_index().to_dict())
print("Test distribution:",  pd.Series(y_test).value_counts().sort_index().to_dict(), "\n")


# ===========================================================
# STEP 3 — Feature Scaling (CRITICAL for MLP)
# ===========================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# ===========================================================
# Helper: run one MLP experiment
# ===========================================================
def run_mlp(hidden, alpha, lr):
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=lr,
        max_iter=400,
        batch_size=32,
        shuffle=True,
        random_state=RANDOM_STATE,
        early_stopping=True,
        n_iter_no_change=10
    )

    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    return evaluate_model(y_test, y_pred)


# ===========================================================
# STEP 4 — Experiment 1: Model Capacity
# ===========================================================
capacity_configs = [(8,), (16,), (32,), (64,), (64,32), (128,64)]
capacity_results = []

print("\n===== EXPERIMENT 1: MODEL CAPACITY =====\n")

for hs in capacity_configs:
    acc, f1, prec, rec, _ = run_mlp(hs, BASE_ALPHA, BASE_LR)

    capacity_results.append({
        "hidden": str(hs),
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec
    })

    print(f"Hidden={hs} → Acc={acc:.4f}, F1={f1:.4f}")

cap_df = pd.DataFrame(capacity_results)


# ===========================================================
# STEP 5 — Experiment 2: Regularization Strength
# ===========================================================
alpha_values = [1e-5, 1e-4, 5e-4, 1e-3, 1e-2]
alpha_results = []

print("\n===== EXPERIMENT 2: REGULARIZATION =====\n")

for alpha in alpha_values:
    acc, f1, prec, rec, _ = run_mlp(BASE_HIDDEN, alpha, BASE_LR)

    alpha_results.append({
        "alpha": alpha,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec
    })

    print(f"Alpha={alpha} → Acc={acc:.4f}, F1={f1:.4f}")

alpha_df = pd.DataFrame(alpha_results)


# ===========================================================
# STEP 6 — Experiment 3: Learning Rate
# ===========================================================
lr_values = [1e-4, 5e-4, 1e-3, 5e-3]
lr_results = []

print("\n===== EXPERIMENT 3: LEARNING RATE =====\n")

for lr in lr_values:
    acc, f1, prec, rec, _ = run_mlp(BASE_HIDDEN, BASE_ALPHA, lr)

    lr_results.append({
        "lr": lr,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec
    })

    print(f"LR={lr} → Acc={acc:.4f}, F1={f1:.4f}")

lr_df = pd.DataFrame(lr_results)


# ===========================================================
# STEP 7 — Plots
# ===========================================================

# Capacity plot
plt.figure(figsize=(7,5))
plt.plot(
    [sum(eval(h)) for h in cap_df["hidden"]],
    cap_df["accuracy"],
    marker="o"
)
plt.xlabel("Model Capacity (sum of hidden units)")
plt.ylabel("Accuracy")
plt.title("MLP Accuracy vs Model Capacity")
plt.grid(True)
plt.show()

# Regularization plot
plt.figure(figsize=(7,5))
plt.plot(alpha_df["alpha"], alpha_df["accuracy"], marker="o")
plt.xscale("log")
plt.xlabel("Alpha (L2 regularization)")
plt.ylabel("Accuracy")
plt.title("MLP Accuracy vs Regularization Strength")
plt.grid(True)
plt.show()

# Learning rate plot
plt.figure(figsize=(7,5))
plt.plot(lr_df["lr"], lr_df["accuracy"], marker="o")
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.title("MLP Accuracy vs Learning Rate")
plt.grid(True)
plt.show()


# ===========================================================
# STEP 8 — Confusion Matrix for Best MLP
# ===========================================================
print("\n===== BEST MLP CONFUSION MATRIX =====")

best_acc, _, _, _, best_cm = run_mlp(BASE_HIDDEN, BASE_ALPHA, BASE_LR)

disp = ConfusionMatrixDisplay(
    confusion_matrix=np.round(best_cm, 1),
    display_labels=LABELS
)

fig, ax = plt.subplots(figsize=(6,5))
disp.plot(cmap="Blues", ax=ax, values_format=".1f")
plt.title(f"MLP Confusion Matrix (%) | Acc={best_acc:.3f}")
plt.show()
