import pandas as pd
import matplotlib.pyplot as plt

# ===========================================================
# CONFIG
# ===========================================================
LR_CSV  = "logistic_regression_learning_curve.csv"
MLP_CSV = "mlp_learning_curve.csv"
SVM_CSV = "svm_learning_curve.csv"

# ===========================================================
# Load results
# ===========================================================
df_lr  = pd.read_csv(LR_CSV).sort_values("dataset_percent")
df_mlp = pd.read_csv(MLP_CSV).sort_values("dataset_percent")
df_svm = pd.read_csv(SVM_CSV).sort_values("dataset_percent")

print("\n===== LEARNING CURVE DATA (LR) =====")
print(df_lr.head())

print("\n===== LEARNING CURVE DATA (MLP) =====")
print(df_mlp.head())

print("\n===== LEARNING CURVE DATA (SVM) =====")
print(df_svm.head())

# ===========================================================
# Plot: Accuracy vs Dataset Size (Combined)
# ===========================================================
fig, ax = plt.subplots(figsize=(7,5))

# Logistic Regression
ax.plot(
    df_lr["dataset_percent"].to_numpy(dtype=float),
    df_lr["accuracy"].to_numpy(dtype=float),
    marker="o",
    linestyle="-",
    linewidth=2,
    color="#1f77b4",
    label="Logistic Regression"
)

# MLP
ax.plot(
    df_mlp["dataset_percent"].to_numpy(dtype=float),
    df_mlp["accuracy"].to_numpy(dtype=float),
    marker="s",
    linestyle="-",
    linewidth=2,
    color="orange",
    label="MLP"
)

# SVM
ax.plot(
    df_svm["dataset_percent"].to_numpy(dtype=float),
    df_svm["accuracy"].to_numpy(dtype=float),
    marker="s",
    linestyle="-",
    linewidth=2,
    color="green",
    label="SVM"
)


ax.set_xlabel("Dataset Size (%)")
ax.set_ylabel("Accuracy")
ax.set_title("Learning Curve: Logistic Regression vs SVM vs MLP")

ax.set_ylim(0, 1)
ax.grid(True, linestyle="--", alpha=0.5)

# Remove plot borders/spines (clean look)
# for spine in ax.spines.values():
#     spine.set_visible(False)

ax.legend()
plt.tight_layout()
plt.show()
