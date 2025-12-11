import argparse
import csv
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.base import clone

import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "landmarks_right_handed_only.csv"
TEST_SIZE = 0.95

# Repeated CV settings
N_SPLITS = 5       # k in k-fold
N_REPEATS = 10     # how many times to repeat k-fold


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
# New: Repeated k-fold evaluation with CIs
# -------------------------------
def repeated_cv_evaluation(X, y, best_clf, n_splits=5, n_repeats=10, random_state=42):
    """
    Run RepeatedStratifiedKFold using the best_clf and compute accuracy
    + macro-F1 for each fold. Then compute per-repeat means and 95% CIs,
    and plot them.
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    acc_scores = []
    f1_scores = []

    print(f"\nRunning RepeatedStratifiedKFold: {n_splits} folds × {n_repeats} repeats "
          f"= {n_splits * n_repeats} evaluations...")

    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y), start=1):
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y[train_idx], y[test_idx]

        clf = clone(best_clf)
        clf.fit(X_train_cv, y_train_cv)
        y_pred_cv = clf.predict(X_test_cv)

        acc = accuracy_score(y_test_cv, y_pred_cv)
        _, _, f1, _ = precision_recall_fscore_support(
            y_test_cv, y_pred_cv, average="macro"
        )

        acc_scores.append(acc)
        f1_scores.append(f1)

    acc_scores = np.array(acc_scores)
    f1_scores = np.array(f1_scores)

    # Reshape to [n_repeats, n_splits] so we can compute stats per repetition
    acc_rep = acc_scores.reshape(n_repeats, n_splits)
    f1_rep = f1_scores.reshape(n_repeats, n_splits)

    # Per-repeat mean and 95% CI (over the folds within that repeat)
    def mean_ci_per_repeat(values_rep):
        means = values_rep.mean(axis=1)
        stds = values_rep.std(axis=1, ddof=1)
        ci = 1.96 * stds / np.sqrt(values_rep.shape[1])
        return means, ci

    acc_means, acc_ci = mean_ci_per_repeat(acc_rep)
    f1_means, f1_ci = mean_ci_per_repeat(f1_rep)

    # Overall mean and CI across all splits
    def overall_mean_ci(values):
        mean = values.mean()
        std = values.std(ddof=1)
        ci = 1.96 * std / np.sqrt(len(values))
        return mean, ci

    acc_mean_all, acc_ci_all = overall_mean_ci(acc_scores)
    f1_mean_all, f1_ci_all = overall_mean_ci(f1_scores)

    print("\n=== Repeated CV Summary (across all folds & repeats) ===")
    print(f"Accuracy: mean = {acc_mean_all:.4f}, 95% CI ≈ [{acc_mean_all - acc_ci_all:.4f}, "
          f"{acc_mean_all + acc_ci_all:.4f}]")
    print(f"Macro F1: mean = {f1_mean_all:.4f}, 95% CI ≈ [{f1_mean_all - f1_ci_all:.4f}, "
          f"{f1_mean_all + f1_ci_all:.4f}]")

    # Plot per-repeat means with 95% CI error bars
    reps = np.arange(1, n_repeats + 1)

    plt.figure(figsize=(8, 5))
    plt.errorbar(reps, acc_means, yerr=acc_ci, fmt="-o", capsize=5)
    plt.xlabel("Repeat index")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy per repeat ({n_splits}-fold CV) with 95% CI")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.errorbar(reps, f1_means, yerr=f1_ci, fmt="-o", capsize=5)
    plt.xlabel("Repeat index")
    plt.ylabel("Macro F1-score")
    plt.title(f"Macro F1 per repeat ({n_splits}-fold CV) with 95% CI")
    plt.grid(True)
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

    # Train/test split (stratified) for a one-shot test evaluation
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

    # Grid search using accuracy
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
    print("\nPlotting CV accuracy vs hyperparameters...")
    plot_linear_results(cv_results)
    plot_rbf_results(cv_results)
    plot_poly_results(cv_results)

    best_clf = grid.best_estimator_

    # Evaluate on held-out test set once
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

    # Now: run repeated k-fold CV on the full dataset with the best model
    repeated_cv_evaluation(
        X,
        y,
        best_clf,
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=random_state,
    )

    return best_clf


if __name__ == "__main__":
    train_and_tune_svm(CSV_PATH, test_size=TEST_SIZE)
