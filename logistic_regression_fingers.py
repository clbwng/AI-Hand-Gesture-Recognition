import pandas as pd
import numpy as np
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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import hashlib


CSV_PATH = "landmarks_reduced.csv"
C_VALUES = [0.0001, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]      # Regularization strengths
LABELS = [0,1,2,3,4,5]           # Valid finger-count classes


# ===========================================================
# Helper: evaluate model using shared metrics
# ===========================================================
def evaluate_model(y_true, y_pred, labels=LABELS):
    """Compute consistent evaluation metrics for LR, MLP, SVM."""
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro")
    rec  = recall_score(y_true, y_pred, average="macro")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_percent = cm.astype(float) / cm.sum(axis=1)[:, None] * 100

    return acc, f1, prec, rec, cm_percent

def run_label_permutation_test(
    X_train, y_train, X_test, y_test, C
):
    """
    Train Logistic Regression on permuted labels
    and evaluate generalization.
    """
    # Shuffle training labels
    y_train_perm = np.random.permutation(y_train)

    model = LogisticRegression(
        C=C,
        max_iter=2000,
        solver="lbfgs",
        multi_class="multinomial"
    )

    model.fit(X_train, y_train_perm)

    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    train_acc = accuracy_score(y_train_perm, train_pred)
    test_acc  = accuracy_score(y_test, test_pred)

    return train_acc, test_acc


# ===========================================================
# STEP 1 — Load + Clean the dataset
# ===========================================================
df = pd.read_csv(CSV_PATH)

# Keep valid classes 0–5
df = df[df["label_fingers"].isin(LABELS)]

# Landmark feature columns
feature_cols = [c for c in df.columns if c.startswith(("x", "y", "z"))]

# Duplicate detection via hashing
def row_hash(row):
    return hashlib.md5(str(tuple(row[feature_cols])).encode()).hexdigest()

df["hash"] = df.apply(row_hash, axis=1)

print("\n===== DATASET SUMMARY =====")
print("Class counts:\n", df["label_fingers"].value_counts().sort_index())
print("Total samples:", len(df))
print("Unique landmark sets:", df["hash"].nunique(), "\n")


# Extract X and y
X = df[feature_cols].values
y = df["label_fingers"].values


# ===========================================================
# STEP 2 — Stratified Train/Test Split
# ===========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Train class distribution:", pd.Series(y_train).value_counts().sort_index().to_dict())
print("Test class distribution:", pd.Series(y_test).value_counts().sort_index().to_dict(), "\n")


# ===========================================================
# STEP 3 — Standardize the features
# ===========================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# ===========================================================
# STEP 4 — Train and evaluate for each C
# ===========================================================
results = []
train_accs = []
test_accs  = []

for C in C_VALUES:
    print(f"\n===== Training Logistic Regression (C={C}) =====")

    model = LogisticRegression(
        C=C,
        max_iter=2000,
        solver="lbfgs",
        multi_class="multinomial"
    )

    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    # Compute metrics (test set)
    acc, f1, prec, rec, cm_percent = evaluate_model(y_test, test_pred)

    results.append({
        "C": C,
        "accuracy": acc,
        "macro_f1": f1,
        "macro_precision": prec,
        "macro_recall": rec
    })

    # Store accuracies
    train_accs.append(accuracy_score(y_train, train_pred))
    test_accs.append(accuracy_score(y_test, test_pred))

    # Print model metrics
    print(f"Train Accuracy:  {train_accs[-1]:.4f}")
    print(f"Test Accuracy:   {test_accs[-1]:.4f}")
    print(f"Macro F1 Score:  {f1:.4f}")
    print(f"Macro Precision:{prec:.4f}")
    print(f"Macro Recall:   {rec:.4f}")

    # Plot confusion matrix (%) 
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.round(cm_percent, 1),
        display_labels=LABELS
    )

    fig, ax = plt.subplots(figsize=(6,5))
    disp.plot(cmap="Blues", ax=ax, values_format=".1f")
    plt.title(f"Confusion Matrix (%) — C={C}")
    plt.show()

# ===========================================================
# Plot: Training vs Test Accuracy
# ===========================================================
plt.figure(figsize=(7,5))

plt.plot(C_VALUES, train_accs, marker="o", label="Training Accuracy")
plt.plot(C_VALUES, test_accs,  marker="o", label="Test Accuracy")

plt.xscale("log")
plt.ylim(0, 1)

plt.xlabel("C (Inverse Regularization Strength)")
plt.ylabel("Accuracy")
plt.title("Logistic Regression: Training vs Test Accuracy")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.show()


# ===========================================================
# STEP 5 — Summary Table
# ===========================================================
results_df = pd.DataFrame(results)
print("\n===== SUMMARY OF LOGISTIC REGRESSION RESULTS =====")
print(results_df.to_string(index=False))


# ===========================================================
# STEP 6 — Line Plot for Accuracy vs C
# ===========================================================
plt.figure(figsize=(7,5))

plt.plot(results_df["C"], results_df["accuracy"], marker="o", linewidth=2)

plt.xscale("log")   # logistic regression regularization works best on log scale
plt.ylim(0, 1)

plt.xlabel("C Value (log scale)")
plt.ylabel("Test Accuracy")
plt.title("Logistic Regression Test Accuracy vs Regularization Strength (C)")
plt.grid(True, linestyle="--", alpha=0.6)

plt.show()

# ===========================================================
# STEP 7 — Label Permutation Test (Overfitting Check)
# ===========================================================
PERMUTATION_C = 10000.0
NUM_RUNS = 5

print("\n===== LABEL PERMUTATION TEST =====")
print(f"Using C = {PERMUTATION_C}")
print("Expected chance accuracy ≈ 16.7% (6 classes)\n")

perm_train_accs = []
perm_test_accs  = []

for i in range(NUM_RUNS):
    train_acc, test_acc = run_label_permutation_test(
        X_train, y_train, X_test, y_test, PERMUTATION_C
    )

    perm_train_accs.append(train_acc)
    perm_test_accs.append(test_acc)

    print(f"Run {i+1}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

print("\nPermutation Test Summary:")
print(f"Mean Train Accuracy: {np.mean(perm_train_accs):.4f}")
print(f"Mean Test Accuracy:  {np.mean(perm_test_accs):.4f}")

plt.figure(figsize=(6,4))
plt.bar(
    ["Permuted Train", "Permuted Test"],
    [np.mean(perm_train_accs), np.mean(perm_test_accs)],
    color=["#f44336", "#2196f3"]
)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Label Permutation Test (C = 10000)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

