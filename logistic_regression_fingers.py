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


CSV_PATH = "landmarks.csv"
C_VALUES = [0.0001, 0.001, 0.1, 1.0, 10.0]      # Regularization strengths
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

for C in C_VALUES:
    print(f"\n===== Training Logistic Regression (C={C}) =====")

    model = LogisticRegression(
        C=C,
        max_iter=2000,
        solver="lbfgs",
        multi_class="multinomial"
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute metrics
    acc, f1, prec, rec, cm_percent = evaluate_model(y_test, y_pred)

    results.append({
        "C": C,
        "accuracy": acc,
        "macro_f1": f1,
        "macro_precision": prec,
        "macro_recall": rec
    })

    # Print model metrics
    print(f"Accuracy:        {acc:.4f}")
    print(f"Macro F1 Score:  {f1:.4f}")
    print(f"Macro Precision: {prec:.4f}")
    print(f"Macro Recall:    {rec:.4f}")

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
