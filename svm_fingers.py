import argparse
import csv
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    StratifiedKFold
)
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


# -------------------------------
# Defaults
# -------------------------------
CSV_PATH_DEFAULT = "landmarks_right_handed_only.csv"
TEST_SIZE_DEFAULT = 0.2

# Repeated CV settings for BEST model stability on train set only
N_SPLITS = 5
N_REPEATS = 10


# -------------------------------
# Load CSV
# -------------------------------
def load_landmark_csv(csv_path):
    """
    Expected columns:
      0: image_path (str)
      1: label_fingers (int)
      2..: 63 floats (x0..x20, y0..y20, z0..z20)
    """
    X, y = [], []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        _ = next(reader)  # header

        for row in reader:
            if not row:
                continue
            try:
                label = int(row[1])
                feats = list(map(float, row[2:]))
            except Exception:
                continue
            X.append(feats)
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# -------------------------------
# Plot helper functions
# -------------------------------

def kfold_cv_mean_ci(X, y, clf, n_splits=10, random_state=42):
    """
    Runs Stratified K-Fold CV on (X,y) and returns mean accuracy and 95% CI.
    CI computed as 1.96 * std/sqrt(n_splits).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        m = clone(clf)
        m.fit(X_tr, y_tr)
        y_hat = m.predict(X_val)
        scores.append(accuracy_score(y_val, y_hat))

    scores = np.array(scores, dtype=float)
    mean = scores.mean()
    std = scores.std(ddof=1) if len(scores) > 1 else 0.0
    ci = 1.96 * std / np.sqrt(len(scores)) if len(scores) > 1 else 0.0
    return mean, ci, scores


def plot_single_bar_mean_ci(mean, ci, model_name="Model", title="10-Fold CV Accuracy (Mean ± 95% CI)"):
    """
    Makes a plot like your screenshot: one bar + errorbar (95% CI).
    """
    plt.figure(figsize=(6, 5))
    plt.bar([model_name], [mean])
    plt.errorbar([model_name], [mean], yerr=[ci], fmt="none", ecolor="black", capsize=8)
    plt.ylim(0.0, 1.02)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_linear_results(df):
    df_lin = df[df["param_svm__kernel"] == "linear"].copy()
    if df_lin.empty:
        print("No linear kernel results to plot.")
        return

    df_lin["C"] = df_lin["param_svm__C"].astype(float)
    df_lin = df_lin.sort_values("C")

    C_vals = df_lin["C"].to_numpy()
    scores = df_lin["mean_test_score"].to_numpy()

    plt.figure()
    plt.plot(C_vals, scores, marker="o")
    plt.xscale("log")
    plt.xlabel("C (log scale)")
    plt.ylabel("CV Accuracy")
    plt.title("Linear Kernel: CV Accuracy vs C (train-set CV)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_rbf_results(df):
    df_rbf = df[df["param_svm__kernel"] == "rbf"].copy()
    if df_rbf.empty:
        print("No RBF kernel results to plot.")
        return

    df_rbf["C"] = df_rbf["param_svm__C"].astype(float)
    df_rbf["gamma"] = df_rbf["param_svm__gamma"].astype(float)

    pivot = df_rbf.pivot(index="gamma", columns="C", values="mean_test_score")

    plt.figure(figsize=(6, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues")
    plt.xlabel("C")
    plt.ylabel("gamma")
    plt.title("RBF Kernel: CV Accuracy (train-set CV)")
    plt.tight_layout()
    plt.show()


def plot_poly_results(df):
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
        pivot = sub.pivot(index="gamma", columns="C", values="mean_test_score")

        ax = plt.subplot(1, n_deg, i)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues", ax=ax)
        ax.set_title(f"Poly Kernel (degree={d})")
        ax.set_xlabel("C")
        ax.set_ylabel("gamma")

    plt.tight_layout()
    plt.show()


# -------------------------------
# Repeated CV evaluation + CI for BEST model (train only)
# -------------------------------
def repeated_cv_best_model_ci(
    X_train,
    y_train,
    best_clf,
    n_splits=5,
    n_repeats=10,
    random_state=42,
):
    """
    Runs RepeatedStratifiedKFold on TRAIN ONLY for the best model and reports:
      - mean accuracy ± 95% CI
      - mean macro-F1 ± 95% CI
    Also plots per-repeat mean with CI over folds.
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    acc_scores = []
    f1_scores = []

    total_evals = n_splits * n_repeats
    print(f"\nRunning RepeatedStratifiedKFold on TRAIN ONLY: "
          f"{n_splits} folds × {n_repeats} repeats = {total_evals} evaluations...")

    for (train_idx, val_idx) in rskf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        clf = clone(best_clf)
        clf.fit(X_tr, y_tr)
        y_val_pred = clf.predict(X_val)

        acc_scores.append(accuracy_score(y_val, y_val_pred))
        _, _, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average="macro")
        f1_scores.append(f1)

    acc_scores = np.array(acc_scores)
    f1_scores = np.array(f1_scores)

    # Overall mean & 95% CI across all 50 evals
    def mean_ci(values):
        mean = values.mean()
        std = values.std(ddof=1)
        ci = 1.96 * std / np.sqrt(len(values))
        return mean, ci

    acc_mean, acc_ci = mean_ci(acc_scores)
    f1_mean, f1_ci = mean_ci(f1_scores)

    print("\n=== Best Model Stability on TRAIN (Repeated CV) ===")
    print(f"Accuracy: mean = {acc_mean:.4f}, 95% CI ≈ [{acc_mean - acc_ci:.4f}, {acc_mean + acc_ci:.4f}]")
    print(f"Macro F1: mean = {f1_mean:.4f}, 95% CI ≈ [{f1_mean - f1_ci:.4f}, {f1_mean + f1_ci:.4f}]")

    # Optional: plot per-repeat means with CI over folds (your original style)
    acc_rep = acc_scores.reshape(n_repeats, n_splits)
    f1_rep = f1_scores.reshape(n_repeats, n_splits)

    def mean_ci_per_repeat(values_rep):
        means = values_rep.mean(axis=1)
        stds = values_rep.std(axis=1, ddof=1)
        ci = 1.96 * stds / np.sqrt(values_rep.shape[1])
        return means, ci

    acc_means, acc_cis = mean_ci_per_repeat(acc_rep)
    f1_means, f1_cis = mean_ci_per_repeat(f1_rep)

    reps = np.arange(1, n_repeats + 1)

    plt.figure(figsize=(8, 5))
    plt.errorbar(reps, acc_means, yerr=acc_cis, fmt="-o", capsize=5)
    plt.xlabel("Repeat index")
    plt.ylabel("Accuracy")
    plt.title(f"Best Model: Accuracy per repeat ({n_splits}-fold) with 95% CI (TRAIN)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.errorbar(reps, f1_means, yerr=f1_cis, fmt="-o", capsize=5)
    plt.xlabel("Repeat index")
    plt.ylabel("Macro F1-score")
    plt.title(f"Best Model: Macro F1 per repeat ({n_splits}-fold) with 95% CI (TRAIN)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return (acc_mean, acc_ci, f1_mean, f1_ci)


# -------------------------------
# Utility: evaluate on test set + plot CM
# -------------------------------
def evaluate_on_test(X_test, y_test, clf, title_suffix=""):
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")

    print(f"\n=== Test Set Performance {title_suffix} ===")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Macro Precision: {prec:.4f}")
    print(f"Macro Recall   : {rec:.4f}")
    print(f"Macro F1-score : {f1:.4f}\n")

    print("Classification report (per class):")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    classes = sorted(np.unique(y_test))

    # Normalize by true class (row-wise) → percentages
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Percentage (%)"},
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (%) {title_suffix}")
    plt.tight_layout()
    plt.show()


    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
    }


def learning_curve_on_fixed_test(
    X_train,
    y_train,
    X_test,
    y_test,
    best_clf,
    n_points=23,
    min_frac=0.05,   # start at 5% to avoid tiny subsets
    max_frac=1.0,    # end at 100%
    repeats=10,
    random_state=42,
    csv_out_path="learning_curve.csv",
):
    """
    Train best_clf on evenly spaced fractions of the TRAIN pool, evaluate on FIXED test set.

    - For frac < 1.0: stratified subsampling with 'repeats' runs -> mean accuracy
    - For frac == 1.0: train once on full train pool -> single accuracy
    Outputs CSV with columns: dataset_percent,accuracy
    """
    # 23 evenly spaced fractions between min_frac and max_frac (inclusive)
    fractions = np.linspace(min_frac, max_frac, n_points, dtype=float)

    results_rows = []

    print("\n==============================")
    print("Learning curve on fixed test set")
    print(f"Points: {n_points} evenly spaced from {min_frac:.3f} to {max_frac:.3f}")
    print(f"Repeats per fraction (<100%): {repeats}")
    print(f"CSV output: {csv_out_path}")
    print("==============================\n")

    n_total = len(y_train)

    for frac in fractions:
        frac = float(frac)
        n_sub = int(round(frac * n_total))
        n_sub = max(1, min(n_sub, n_total))  # clamp

        # 100% special case
        if n_sub == n_total:
            clf = clone(best_clf)
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test))

            dataset_percent = 100.0
            results_rows.append({"dataset_percent": dataset_percent, "accuracy": acc})

            print(f"Train frac={frac:.6f} ({dataset_percent:.3f}%) n={n_sub:4d}  Test acc={acc:.4f}")
            continue

        # Stratified subsampling for frac < 1.0
        sss = StratifiedShuffleSplit(
            n_splits=repeats,
            train_size=n_sub,
            random_state=random_state + int(round(frac * 10000)),
        )

        accs = []
        for sub_idx, _ in sss.split(X_train, y_train):
            X_sub = X_train[sub_idx]
            y_sub = y_train[sub_idx]

            clf = clone(best_clf)
            clf.fit(X_sub, y_sub)
            accs.append(accuracy_score(y_test, clf.predict(X_test)))

        accs = np.array(accs, dtype=float)
        acc_mean = float(accs.mean())

        dataset_percent = frac * 100.0
        results_rows.append({"dataset_percent": dataset_percent, "accuracy": acc_mean})

        print(f"Train frac={frac:.6f} ({dataset_percent:.3f}%) n={n_sub:4d}  Test acc mean={acc_mean:.4f}")

    # Save CSV with ONLY dataset_percent and accuracy
    df_out = pd.DataFrame(results_rows, columns=["dataset_percent", "accuracy"])
    df_out.to_csv(csv_out_path, index=False)

    # Plot
    # Plot (convert pandas Series -> numpy to avoid pandas multidim indexing error)
    x = df_out["dataset_percent"].to_numpy(dtype=float)
    y = df_out["accuracy"].to_numpy(dtype=float)

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("Training data used (%) of training pool")
    plt.ylabel("Test accuracy")
    plt.title("Learning Curve: Best Model vs Training Set Size (fixed test set)")
    plt.grid(True)
    plt.ylim(0.0, 1.02)
    plt.tight_layout()
    plt.show()


    print(f"\nSaved learning-curve CSV to: {csv_out_path}")
    return df_out

# -------------------------------
# Main training pipeline
# -------------------------------
def train_full_pipeline(csv_path, test_size=0.2, random_state=42):
    print(f"Loading data from: {csv_path}")
    X, y = load_landmark_csv(csv_path)

    print(f"Total samples: {len(y)}")
    print("Class distribution:", Counter(y))

    # Hold-out test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    # Pipeline
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("svm", SVC()),
    ])

    # Hyperparameter grid
    param_grid = [
        {"svm__kernel": ["linear"], "svm__C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100]},
        {"svm__kernel": ["rbf"], "svm__C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100], "svm__gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1, 10]},
    ]

    # param_grid = [
    #     {"svm__kernel": ["rbf"], "svm__C": [0.01, 0.05, 0.1, 0.5, 1, 10, 100], "svm__gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1, 10]},
    # ]

    # GridSearchCV on TRAIN ONLY, 5-fold
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    print("\nRunning grid search over kernels, C, gamma, degree (5-fold CV on TRAIN)...")
    grid.fit(X_train, y_train)

    print("\n=== Best Model (by CV accuracy on TRAIN) ===")
    print("Best params:", grid.best_params_)
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    # Plots for hyperparameter sweeps (from cv_results_)
    cv_results = pd.DataFrame(grid.cv_results_)
    print("\nPlotting CV accuracy vs hyperparameters...")
    plot_linear_results(cv_results)
    plot_rbf_results(cv_results)
    plot_poly_results(cv_results)

    # Best estimator refit on full train
    best_clf = grid.best_estimator_
    best_clf.fit(X_train, y_train)

    cv10_mean, cv10_ci, _ = kfold_cv_mean_ci(
        X_train, y_train, best_clf, n_splits=10, random_state=random_state
    )
    plot_single_bar_mean_ci(
        cv10_mean, cv10_ci,
        model_name="SVM",
        title="SVM 10-Fold CV Accuracy (Mean ± 95% CI)"
    )
    print(f"\nSVM 10-fold CV on TRAIN: mean={cv10_mean:.4f}, 95% CI=[{cv10_mean-cv10_ci:.4f}, {cv10_mean+cv10_ci:.4f}]")


    # Evaluate best model on test set
    test_metrics = evaluate_on_test(X_test, y_test, best_clf, title_suffix="Test Set")

    # Best-model stability + CI on TRAIN ONLY (Repeated CV)
    acc_mean, acc_ci, f1_mean, f1_ci = repeated_cv_best_model_ci(
        X_train,
        y_train,
        best_clf,
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=random_state,
    )

    # -------------------------
    # Permutation test
    # -------------------------
    print("\n==============================")
    print("PERMUTATION TEST: shuffle TRAIN labels, keep test labels unchanged")
    print("==============================")

    rng = np.random.default_rng(random_state)
    y_train_perm = rng.permutation(y_train)

    perm_clf = clone(best_clf)
    perm_clf.fit(X_train, y_train_perm)

    perm_test_metrics = evaluate_on_test(X_test, y_test, perm_clf, title_suffix="(TRAIN LABELS PERMUTED, Held-out Test)")

    # Comparison bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(["True labels", "Permuted train labels"],
            [test_metrics["accuracy"], perm_test_metrics["accuracy"]])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy: True vs Permuted-Label Training")
    plt.tight_layout()
    plt.show()


    # -------------------------
    # Learning curve (train on 10%, 20%, ..., 100% of training pool; test on fixed test set)
    # -------------------------
    learning_curve_on_fixed_test(
        X_train,
        y_train,
        X_test,
        y_test,
        best_clf,
        n_points=100,
        min_frac=0.01,
        max_frac=1.0,
        repeats=10,
        random_state=random_state,
        csv_out_path="svm_learning_curve.csv",
    )




    print("\n=== Quick Summary ===")
    print(f"Best hyperparameters (true labels): {grid.best_params_}")
    print(f"Test accuracy (true labels): {test_metrics['accuracy']:.4f}")
    print(f"Test precision (macro): {test_metrics['precision_macro']:.4f}")
    print(f"Test recall (macro): {test_metrics['recall_macro']:.4f}")
    print(f"Test f1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"Train stability (repeated CV) accuracy mean ± 95% CI: {acc_mean:.4f} ± {acc_ci:.4f}")
    print(f"Permutation test accuracy (test): {perm_test_metrics['accuracy']:.4f}")


    print("\n=== FINAL HELD-OUT TEST METRICS ===")
    print(f"Test Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision_macro']:.4f}  (macro)")
    print(f"Test Recall   : {test_metrics['recall_macro']:.4f}  (macro)")
    print(f"Test F1       : {test_metrics['f1_macro']:.4f}  (macro)")



    return best_clf


# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_PATH_DEFAULT, help="Path to landmark CSV")
    parser.add_argument("--test_size", type=float, default=TEST_SIZE_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_full_pipeline(args.csv, test_size=args.test_size, random_state=args.seed)
