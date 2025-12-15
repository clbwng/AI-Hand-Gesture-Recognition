import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import hashlib

# ===========================================================
# Configuration
# ===========================================================
CSV_PATH = "landmarks_reduced.csv"
LABELS = [0,1,2,3,4,5]

N_RUNS = 5
RANDOM_STATE = 42

# Fixed MLP config (intentionally high-capacity)
MLP_CONFIG = dict(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-5,
    learning_rate_init=1e-3,
    max_iter=500,
    batch_size=32,
    shuffle=True,
    early_stopping=False,
    random_state=RANDOM_STATE
)

# ===========================================================
# Load + clean dataset
# ===========================================================
df = pd.read_csv(CSV_PATH)
df = df[df["label_fingers"].isin(LABELS)]

feature_cols = [c for c in df.columns if c.startswith(("x","y","z"))]

def row_hash(row):
    return hashlib.md5(str(tuple(row[feature_cols])).encode()).hexdigest()

df["hash"] = df.apply(row_hash, axis=1)

print("\n===== DATASET SUMMARY =====")
print(df["label_fingers"].value_counts().sort_index())
print("Total samples:", len(df))
print("Unique samples:", df["hash"].nunique(), "\n")

X = df[feature_cols].values
y = df["label_fingers"].values


# ===========================================================
# Permutation Test
# ===========================================================
train_accs = []
test_accs  = []

print("===== MLP PERMUTATION TEST =====\n")

for run in range(N_RUNS):
    print(f"Run {run+1}")

    # Shuffle labels
    y_perm = np.random.permutation(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_perm,
        test_size=0.20,
        stratify=y_perm,
        random_state=run
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Train MLP
    mlp = MLPClassifier(**MLP_CONFIG)
    mlp.fit(X_train, y_train)

    # Evaluate
    train_pred = mlp.predict(X_train)
    test_pred  = mlp.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc  = accuracy_score(y_test, test_pred)

    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"  Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}\n")


# ===========================================================
# Summary
# ===========================================================
print("===== PERMUTATION TEST SUMMARY =====")
print(f"Mean Train Accuracy: {np.mean(train_accs):.4f}")
print(f"Mean Test Accuracy:  {np.mean(test_accs):.4f}")
