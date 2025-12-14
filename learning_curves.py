import pandas as pd
import matplotlib.pyplot as plt

# ===========================================================
# CONFIG
# ===========================================================
LR_CSV  = "logistic_regression_learning_curve.csv"
MLP_CSV = "mlp_learning_curve.csv"

# ===========================================================
# Load results
# ===========================================================
df_lr  = pd.read_csv(LR_CSV).sort_values("dataset_percent")
df_mlp = pd.read_csv(MLP_CSV).sort_values("dataset_percent")

print("\n===== LEARNING CURVE DATA (LR) =====")
print(df_lr.head())

print("\n===== LEARNING CURVE DATA (MLP) =====")
print(df_mlp.head())

# ===========================================================
# Plot: Accuracy vs Dataset Size (Combined)
# ===========================================================
fig, ax = plt.subplots(figsize=(7,5))

# Logistic Regression
ax.plot(
    df_lr["dataset_percent"],
    df_lr["accuracy"],
    marker="o",
    linestyle="-",
    linewidth=2,
    color="#1f77b4",
    label="Logistic Regression"
)

# MLP
ax.plot(
    df_mlp["dataset_percent"],
    df_mlp["accuracy"],
    marker="s",
    linestyle="--",
    linewidth=2,
    color="orange",
    label="MLP"
)

ax.set_xlabel("Dataset Size (%)")
ax.set_ylabel("Accuracy")
ax.set_title("Learning Curve: Logistic Regression vs MLP")

ax.set_ylim(0, 1)
ax.grid(True, linestyle="--", alpha=0.5)

# Remove plot borders/spines (clean look)
# for spine in ax.spines.values():
#     spine.set_visible(False)

ax.legend()
plt.tight_layout()
plt.show()
