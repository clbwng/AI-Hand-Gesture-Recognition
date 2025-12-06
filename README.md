# AI Hand Gesture Recognition

Using logistic regression, SVM, and MLP to classify hand gestures based on MediaPipe hand landmark detection.

---

## Overview

This project builds a complete hand-gesture classification pipeline using **MediaPipe Hands** for landmark extraction and classical machine-learning models (Logistic Regression, SVM, MLP). The goal is to classify:

- **Finger Count** (0–5)
- **Handedness** (Left / Right)

based purely on 21 landmark points (x, y, z) per hand.

---

## Dataset

The model is trained on the publicly available **Kaggle Fingers dataset**, where each image is labeled according to the number of fingers extended and whether the hand is left or right.

### Dataset Format

```
<uuid>_<finger><hand>.png
16779a42..._5R.png   → Right hand, 5 fingers
c1fe6c0b..._2L.png   → Left hand, 2 fingers
```

All images are stored in:

```
images/train/
```

---

## Landmark Extraction & Annotation Pipeline

A custom annotation script was built to convert raw images into a structured CSV suitable for machine learning. The pipeline performs:

### ✔ Filename-based label extraction
- Second-last character = finger count (0–5)
- Last character = handedness (L/R)

### ✔ Image preprocessing for more accurate detection
- Contrast boosting
- Uniform resizing to improve MediaPipe stability

### ✔ MediaPipe landmark extraction
- 21 keypoints per image
- Normalized (x, y, z) values written to CSV

### ✔ Confidence-based filtering

Images are only accepted if MediaPipe returns a valid detection with:

```
detection score ≥ 0.8
```

This ensures high-quality annotations and reduces noise during training.

### ✔ Final dataset output

A clean `fingers_landmarks_clean.csv` file with the format:

```
image_path, label_hand, label_fingers,
x0, x1, ..., x20,
y0, ..., y20,
z0, ..., z20
```

This serves as the input to the machine-learning models.

---

## Annotation Verification Tool

A separate verification script randomly loads annotated samples and overlays the landmarks on the original image. This helps validate the correctness of MediaPipe detections before training.

### Features:
- Random sample selection
- Landmark visualization with pixel-accurate mapping
- Automatic window handling (next image on keypress or close)

This step is essential to ensure data quality before model training.

---

## Machine Learning Models (Upcoming / In Progress)

The project will train and compare the following models:

### 🔹 Logistic Regression
Baseline classifier for quick finger-count and hand-side prediction.

### 🔹 Support Vector Machine (SVM)
More powerful nonlinear classifier using RBF kernels.

### 🔹 Multi-Layer Perceptron (MLP)
A simple neural network trained on normalized landmark coordinates.

### Evaluation Metrics

Models will be evaluated on:
- Accuracy
- Confusion matrices
- Cross-validation
- Generalization to unseen hand poses

---

## Goals

- Build a fast, lightweight classifier for gesture recognition
- Evaluate how classical ML compares to deep learning on landmark-based inputs
- Provide a reproducible pipeline from raw images → landmarks → ML model

---

## Future Extensions

- Gesture classification beyond finger counting
- Integration with robotic control (e.g., mapping gestures to robot hand poses)
- Data augmentation & improved detection heuristics

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AI-Hand-Gesture-Recognition.git
cd AI-Hand-Gesture-Recognition

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Step 1: Extract Landmarks
```bash
python annotate_landmarks.py

# annotation.py can also be used for a lower confidence detection
```

### Step 2: Verify Annotations
```bash
python verify_annotations.py
```

### Step 3: Train Models - TO BE COMPLETED
```bash
python train_models.py
```

---

## Project Structure

```
AI-Hand-Gesture-Recognition/
├── images/
│   └── train/              # Raw hand gesture images
├── data/
│   └── fingers_landmarks_clean.csv  # Extracted landmarks
├── models/                 # Saved model checkpoints
├── annotate_landmarks.py   # Landmark extraction script
├── verify_annotations.py   # Visualization tool
├── train_models.py         # Model training pipeline
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- **MediaPipe** by Google for hand landmark detection
- **Kaggle Fingers Dataset** for training data
- Classical ML libraries: scikit-learn, pandas, numpy

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## Contact

For questions or feedback, reach out at your.email@example.com