# AI Hand Gesture Recognition (concise)

Lightweight pipeline to extract MediaPipe hand landmarks and train classical ML models to predict finger count (0–5).

## What’s in this repo
- `annotate_landmarks.py` — extract 21 MediaPipe landmarks (x,y,z) per image and save CSV rows.
- `verify_landmarks.py` — visual check: overlay landmarks on images (random samples).
- `logistic_regression_fingers.py` — logistic regression baseline: train/val/test split, hyperparameter sweep, confusion matrix, CV, learning curve, permuted-label sanity check.
- `svm_fingers.py` — SVM utilities and evaluation helpers (grid search, plots, CI calculation).
- `mlp_finger_classifier.py` — MLP baseline with similar evaluation flow to logistic regression.
- `landmarks*.csv` — example CSV files produced by the annotation pipeline (feature columns: `x0..x20, y0..y20, z0..z20`, label column: `label_fingers`).

## Quick start
1. Install deps:

```bash
pip install -r requirements.txt
```

2. Extract landmarks from an images folder:

```bash
python annotate_landmarks.py --input path/to/images --output landmarks_reduced.csv
```

3. Verify a few annotations:

```bash
python verify_landmarks.py
```

4. Run the logistic regression baseline (example):

```bash
python logistic_regression_fingers.py
```

Notes: the model scripts expect CSVs with `image_path, label_fingers, x0..x20, y0..y20, z0..z20`. Adjust `CSV_PATH` in the scripts if needed.

## Outputs and evaluation
- Confusion matrices, learning-curve CSVs, and CV summaries are produced by the model scripts.
- Files written by the scripts (e.g., `logistic_regression_learning_curve.csv`, `mlp_learning_curve.csv`) are saved in the repository root by default.

## Tips
- Use `label_fingers` values 0–5. Many scripts filter to supported labels via their `LABELS` config.
- If you change dataset filenames or column layout, update the `feature_cols` selection in the model scripts.

## Contact / Notes
This repo is focused on landmark-based classical ML baselines (not end-to-end image CNNs). If you want a script added (e.g., a single `train_models.py` orchestrator), I can add one.
