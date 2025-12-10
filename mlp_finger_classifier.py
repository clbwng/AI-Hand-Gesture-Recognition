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

CSV_PATH = "landmarks.csv"
LABELS = [0,1,2,3,4,5]

# ===========================================================
# Helper: Uniform evaluation across all models
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

X = df[feature_cols].values
y = df["label_fingers"].values


# ===========================================================
# STEP 2 — Train/test split
# ===========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# ===========================================================
# STEP 3 — Scale features
# ===========================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# ===========================================================
# STEP 4 — Expanded Hyperparameter Sweep (good + bad models)
# ===========================================================

HIDDEN_SIZES = [
    (8,), (16,), (16,8),        # deliberately UNDERFIT
    (32,), (64,), (64,32),      # baseline configs
    (128,64), (256,128,64)      # deliberately OVERFIT
]

ALPHAS = [
    1e-5, 5e-5,                 # weak regularization (overfit)
    1e-4, 5e-4, 1e-3,           # normal range
    1e-2, 1e-1                 # VERY strong regularization (underfit)
]

LEARNING_RATES = [
    1e-5, 5e-5,                 # too small → almost no learning
    5e-4, 1e-3, 5e-3,           # normal range
    1e-2                        # too large → instability
]


results = []
best_model = None
best_acc = -1
best_cm = None

print("\n===== SWEEPING MLP HYPERPARAMETERS =====\n")

for hs in HIDDEN_SIZES:
    for alpha in ALPHAS:
        for lr in LEARNING_RATES:

            print(f"Training MLP: hidden={hs}, alpha={alpha}, lr={lr}")

            mlp = MLPClassifier(
                hidden_layer_sizes=hs,
                activation='relu',
                solver='adam',
                alpha=alpha,
                learning_rate_init=lr,
                max_iter=400,
                batch_size=32,
                shuffle=True,
                random_state=42,
                early_stopping=True,
                n_iter_no_change=10
            )

            try:
                mlp.fit(X_train, y_train)
                y_pred = mlp.predict(X_test)

                acc, f1, prec, rec, cm_percent = evaluate_model(y_test, y_pred)

            except Exception as e:
                print("Model failed:", e)
                acc = f1 = prec = rec = 0

            # Save result
            results.append({
                "hidden": str(hs),
                "alpha": alpha,
                "lr": lr,
                "accuracy": acc,
                "f1": f1,
                "precision": prec,
                "recall": rec
            })

            # Track best
            if acc > best_acc:
                best_acc = acc
                best_model = mlp
                best_cm = cm_percent.copy()
                best_config = (hs, alpha, lr)

            print(f"  → Accuracy={acc:.4f}, F1={f1:.4f}\n")


# ===========================================================
# STEP 5 — Summary Table
# ===========================================================
results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values(by="accuracy", ascending=False)
print("\n===== MLP HYPERPARAMETER SWEEP RESULTS =====")
print(results_df_sorted.to_string(index=False))


# ===========================================================
# STEP 6 — Best Model Confusion Matrix
# ===========================================================
print("\n===== BEST MODEL CONFIGURATION =====")
print(f"Hidden: {best_config[0]}")
print(f"Alpha:  {best_config[1]}")
print(f"LR:     {best_config[2]}")
print(f"Best Accuracy: {best_acc:.4f}")

disp = ConfusionMatrixDisplay(
    confusion_matrix=np.round(best_cm, 1),
    display_labels=LABELS
)
fig, ax = plt.subplots(figsize=(6,5))
disp.plot(cmap="Blues", ax=ax, values_format=".1f")
plt.title(f"Best MLP Confusion Matrix (%) | Acc={best_acc:.3f}")
plt.show()


# ===========================================================
# STEP 7 — Line Plots for Hyperparameter Effects
# ===========================================================

# Plot 1: Accuracy vs alpha
plt.figure(figsize=(7,5))
plt.plot(results_df["alpha"], results_df["accuracy"], 'o-', linewidth=2)
plt.xscale("log")
plt.xlabel("Alpha (L2 regularization)")
plt.ylabel("Accuracy")
plt.title("MLP Accuracy vs Regularization Strength (alpha)")
plt.grid(True, linestyle="--")
plt.show()

# Plot 2: Accuracy vs learning rate
plt.figure(figsize=(7,5))
plt.plot(results_df["lr"], results_df["accuracy"], 'o-', linewidth=2)
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.title("MLP Accuracy vs Learning Rate")
plt.grid(True, linestyle="--")
plt.show()

# Plot 3: Accuracy vs network size
def network_size(hs):
    return sum(hs)   # simple measure of capacity

results_df["capacity"] = results_df["hidden"].apply(eval).apply(network_size)

plt.figure(figsize=(7,5))
plt.plot(results_df["capacity"], results_df["accuracy"], 'o-', linewidth=2)
plt.xlabel("Network Capacity (sum of hidden units)")
plt.ylabel("Accuracy")
plt.title("MLP Accuracy vs Model Capacity")
plt.grid(True, linestyle="--")
plt.show()
