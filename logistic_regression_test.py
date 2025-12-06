import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import itertools
import warnings

warnings.filterwarnings("ignore")

CSV_PATH = "fingers_landmarks_clean.csv"

# -----------------------------------------------------------
# Hyperparameters to sweep
# -----------------------------------------------------------
C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
CLASS_WEIGHTS = [None, "balanced"]

REPEATS = 5     # run each config multiple times to compute variance

# -----------------------------------------------------------
# Load dataset
# -----------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Label: finger count (0 or 1 for binary fist/palm)
# MODIFY this if your labels differ (e.g., for regression vs classification)
y = df["label_fingers"].values

# Features: all x, y, z values
feature_cols = [c for c in df.columns if c.startswith("x") or c.startswith("y") or c.startswith("z")]
X = df[feature_cols].values

print(f"Loaded {len(X)} samples with {X.shape[1]} features.")

# -----------------------------------------------------------
# Train + evaluate logistic regression
# -----------------------------------------------------------
def run_one_experiment(C_value, class_weight):
    """Run a single train/val/test split + model training."""
    
    # 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=None, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=None, stratify=y_temp
    )
    
    # Feature scaling (IMPORTANT for logistic regression)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Logistic Regression
    model = LogisticRegression(
        C=C_value,
        class_weight=class_weight,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000
    )

    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)

    metrics = {
        "val_accuracy": accuracy_score(y_val, y_pred),
        "val_precision_macro": precision_score(y_val, y_pred, average="macro"),
        "val_recall_macro": recall_score(y_val, y_pred, average="macro"),
        "val_f1_macro": f1_score(y_val, y_pred, average="macro"),
    }

    return metrics


# -----------------------------------------------------------
# Grid search over hyperparameters
# -----------------------------------------------------------
results = []

print("\nStarting Logistic Regression experiments...\n")

for C_value, class_weight in itertools.product(C_VALUES, CLASS_WEIGHTS):
    trial_metrics = []

    print(f"Testing C={C_value}, class_weight={class_weight} ...")

    for i in range(REPEATS):
        m = run_one_experiment(C_value, class_weight)
        trial_metrics.append(m)

    # Compute means & std
    mean_acc = np.mean([t["val_accuracy"] for t in trial_metrics])
    std_acc = np.std([t["val_accuracy"] for t in trial_metrics])

    mean_f1 = np.mean([t["val_f1_macro"] for t in trial_metrics])
    std_f1 = np.std([t["val_f1_macro"] for t in trial_metrics])

    results.append({
        "C": C_value,
        "class_weight": class_weight,
        "mean_val_accuracy": mean_acc,
        "std_val_accuracy": std_acc,
        "mean_val_f1": mean_f1,
        "std_val_f1": std_f1
    })

# Convert results to a table
results_df = pd.DataFrame(results)
print("\n===== LOGISTIC REGRESSION RESULTS =====")
print(results_df.sort_values(by="mean_val_accuracy", ascending=False).to_string(index=False))

# Save results
results_df.to_csv("logreg_results.csv", index=False)
print("\nSaved results to logreg_results.csv\n")
