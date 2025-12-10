import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

CSV_PATH = "landmarks.csv"      # <-- your unified CSV
DATASET_DIR = "dataset"         # folder containing all images (not required for training)
C_VALUES = [0.1, 1.0, 10.0]      # test 3 C values

# -----------------------------------------------------------
# Load dataset
# -----------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Extract labels (must be 1–5 only)
y = df["label_fingers"].astype(int).values

# Feature columns = all x*, y*, z*
feature_cols = [c for c in df.columns if c.startswith(("x", "y", "z"))]
X = df[feature_cols].values

print(f"Loaded {len(X)} samples with {X.shape[1]} features.")
print("Class distribution:", np.bincount(y)[1:], "\n")

# -----------------------------------------------------------
# Stratified Train/Test Split
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Train samples = {len(X_train)}")
print(f"Test samples  = {len(X_test)}\n")

# -----------------------------------------------------------
# Standardize features (fit only on training)
# -----------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -----------------------------------------------------------
# Train and evaluate models for 3 C values
# -----------------------------------------------------------
test_accuracies = []

for C in C_VALUES:
    print(f"\n===== Training Logistic Regression (C={C}) =====")

    model = LogisticRegression(
        C=C,
        max_iter=2000,
        solver="lbfgs",
        multi_class="multinomial"
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    test_accuracies.append(acc)

    print(f"Test Accuracy (C={C}): {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,2,3,4,5])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix – Finger Count (C={C})")
    plt.show()

# -----------------------------------------------------------
# Accuracy comparison bar plot
# -----------------------------------------------------------
plt.figure(figsize=(6,5))
plt.bar([str(c) for c in C_VALUES], test_accuracies, color=["#4caf50", "#2196f3", "#ff9800"])
plt.ylim(0, 1)
plt.ylabel("Test Accuracy")
plt.xlabel("C Value")
plt.title("Logistic Regression Performance Across C Values")
plt.show()
