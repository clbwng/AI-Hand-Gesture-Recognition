import os
import csv
import cv2
import mediapipe as mp

DATA_DIR = "images/train"
OUTPUT_CSV = "fingers_landmarks_clean.csv"

mp_hands = mp.solutions.hands

# ---------------------------------------------------------
# Extract labels from filename
# Format: <uuid>_<finger><hand>.png  →  "5R", "3L", "0R"
# ---------------------------------------------------------
def extract_labels_from_filename(filename):
    name, _ = os.path.splitext(filename)
    finger = name[-2]        # '0'–'5'
    hand = name[-1].upper()  # 'L' or 'R'
    return hand, int(finger)

# ---------------------------------------------------------
# Optional preprocessing to improve detection accuracy
# ---------------------------------------------------------
def preprocess_image(img):
    # Slight contrast boost
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)

    # Resize to a consistent larger width (MediaPipe performs better)
    TARGET_WIDTH = 512
    h, w = img.shape[:2]
    scale = TARGET_WIDTH / w
    img = cv2.resize(img, (TARGET_WIDTH, int(h * scale)))

    return img

# ---------------------------------------------------------
# Main script
# ---------------------------------------------------------
def main():
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.1,   # stricter for better accuracy
        min_tracking_confidence=0.1
    ) as hands, open(OUTPUT_CSV, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile)

        # CSV header
        header = ["image_path", "label_hand", "label_fingers"]
        header += [f"x{i}" for i in range(21)]
        header += [f"y{i}" for i in range(21)]
        header += [f"z{i}" for i in range(21)]
        writer.writerow(header)

        print("Starting annotation...\n")

        for filename in os.listdir(DATA_DIR):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(DATA_DIR, filename)

            # Parse labels
            try:
                label_hand, label_fingers = extract_labels_from_filename(filename)
            except Exception:
                print("Skipping invalid filename:", filename)
                continue

            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print("Could not load:", img_path)
                continue

            # Preprocess for better detection
            img_proc = preprocess_image(img)

            img_rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # No detection → skip
            if not results.multi_hand_landmarks or not results.multi_handedness:
                print("No hand detected:", filename)
                continue

            # Confidence filtering
            handed = results.multi_handedness[0].classification[0]
            score = handed.score

            # Skip low-confidence detections
            if score < 0.80:
                print(f"Low confidence ({score:.2f}) → skipping:", filename)
                continue

            # Extract landmarks
            lm = results.multi_hand_landmarks[0].landmark
            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            zs = [p.z for p in lm]

            # Save to CSV
            writer.writerow([img_path, label_hand, label_fingers] + xs + ys + zs)

            print(f"OK: {filename} (score={score:.2f})")

    print("\n---------------------------------------")
    print(" Annotation complete! Saved to:")
    print(" →", OUTPUT_CSV)
    print("---------------------------------------\n")


if __name__ == "__main__":
    main()
