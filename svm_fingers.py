import argparse
import csv
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "landmarks_right_handed_only.csv"
TEST_SIZE = 0.2


# -------------------------------
# Load CSV
# -------------------------------
def load_landmark_csv(csv_path):
    """
    Loads the CSV produced by your landmark annotator.

    Expected columns:
      0: image_path (str)
      1: label_fingers (int)
      2..: x0..x20, y0..y20, z0..z20 (floats)
    """
    X = []
    y = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        for row in reader:
            if not row:
                continue

            try:
                label = int(row[1])
            except ValueError:
                # skip bad rows
                continue

            feats = list(map(float, row[2:]))
            X.append(feats)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


# -------------------------------
# Plot helper functions
# -------------------------------
def plot_linear_results(df):
    """Line plot: C vs CV accuracy for linear kernel."""
    df_lin = df[df["param_svm__kernel"] == "linear"].copy()
    if df_lin.empty:
        print("No linear kernel results to plot.")
        return

    # Convert C to float for proper sorting
    df_lin["C"] = df_lin["param_svm__C"].astype(float)
    df_lin = df_lin.sort_values("C")

    # Convert to numpy arrays to avoid pandas multi-d indexing issue
    C_vals = df_lin["C"].to_numpy()
    scores = df_lin["mean_test_score"].to_numpy()

    plt.figure()
    plt.plot(C_vals, scores, marker="o")
    plt.xscale("log")
    plt.xlabel("C (log scale)")
    plt.ylabel("CV Accuracy")
    plt.title("Linear Kernel: Accuracy vs C")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_rbf_results(df):
    """Heatmap: C vs gamma for RBF kernel."""
    df_rbf = df[df["param_svm__kernel"] == "rbf"].copy()
    if df_rbf.empty:
        print("No RBF kernel results to plot.")
        return

    df_rbf["C"] = df_rbf["param_svm__C"].astype(float)
    df_rbf["gamma"] = df_rbf["param_svm__gamma"].astype(float)

    pivot = df_rbf.pivot(
        index="gamma",
        columns="C",
        values="mean_test_score"
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="Blues"
    )
    plt.xlabel("C")
    plt.ylabel("gamma")
    plt.title("RBF Kernel: CV Accuracy")
    plt.tight_layout()
    plt.show()


def plot_poly_results(df):
    """Heatmaps: for each poly degree, C vs gamma."""
    df_poly = df[df["param_svm__kernel"] == "poly"].copy()
    if df_poly.empty:
        print("No poly kernel results to plot.")
        return

    df_poly["C"] = df_poly["param_svm__C"].astype(float)
    df_poly["gamma"] = df_poly["param_svm__gamma"].astype(float)
    df_poly["degree"] = df_poly["param_svm__degree"].astype(int)

    degrees = sorted(df_poly["degree"].unique())
    n_deg = len(degrees)

    plt.figure(figsize=(6 * n_deg, 5))
    for i, d in enumerate(degrees, 1):
        sub = df_poly[df_poly["degree"] == d]
        pivot = sub.pivot(
            index="gamma",
            columns="C",
            values="mean_test_score"
        )

        ax = plt.subplot(1, n_deg, i)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            ax=ax
        )
        ax.set_title(f"Poly Kernel (degree={d})")
        ax.set_xlabel("C")
        ax.set_ylabel("gamma")

    plt.tight_layout()
    plt.show()


# -------------------------------
# Train & evaluate SVM with grid search
# -------------------------------
def train_and_tune_svm(
    csv_path,
    test_size=0.2,
    random_state=42,
):
    print(f"Loading data from: {csv_path}")
    X, y = load_landmark_csv(csv_path)

    print(f"Total samples: {len(y)}")
    print("Class distribution:", Counter(y))

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    # Base pipeline: scale -> SVM
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC()),
        ]
    )

    # Hyperparameter grid:
    param_grid = [
        # Linear kernel: only C matters
        {
            "svm__kernel": ["linear"],
            "svm__C": [0.1, 1, 10, 100],
        },
        # RBF kernel: C and gamma
        {
            "svm__kernel": ["rbf"],
            "svm__C": [0.1, 1, 10, 100],
            "svm__gamma": [0.001, 0.01, 0.1, 1],
        },
        # Polynomial kernel: C, gamma, degree
        {
            "svm__kernel": ["poly"],
            "svm__C": [0.1, 1, 10],
            "svm__gamma": [0.001, 0.01, 0.1],
            "svm__degree": [2, 3, 4],
        },
    ]

    # Grid search using accuracy (as you requested)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    print("\nRunning grid search over kernels, C, gamma, degree...")
    grid.fit(X_train, y_train)

    print("\n=== Best Model (by CV accuracy) ===")
    print("Best params:", grid.best_params_)
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    # Convert cv_results_ to DataFrame for plotting
    cv_results = pd.DataFrame(grid.cv_results_)
    # Plot how accuracy changes with hyperparameters
    print("\nPlotting CV accuracy vs hyperparameters...")
    plot_linear_results(cv_results)
    plot_rbf_results(cv_results)
    plot_poly_results(cv_results)

    best_clf = grid.best_estimator_

    # Evaluate on test set
    y_test_pred = best_clf.predict(X_test)

    acc = accuracy_score(y_test, y_test_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="macro"
    )

    print("\n=== Test Set Performance (Best Model) ===")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall   : {recall:.4f}")
    print(f"Macro F1-score : {f1:.4f}\n")

    print("Classification report (per class):")
    print(classification_report(y_test, y_test_pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    classes = sorted(np.unique(y))

    print("Confusion matrix (raw counts):")
    print(cm)

    # Plot confusion matrix heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Best SVM Model")
    plt.tight_layout()
    plt.show()

    return best_clf


if __name__ == "__main__":
    train_and_tune_svm(CSV_PATH, test_size=TEST_SIZE)
