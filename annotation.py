import os
import csv
import cv2
import mediapipe as mp

# -----------------------------
# PATH CONFIG
# -----------------------------
DATA_DIR = "train"    # <- your training folder from Roboflow
OUTPUT_CSV = "finger_landmarks.csv"

mp_hands = mp.solutions.hands


# -------------------------------------------------
# PARSE LABEL FROM FILENAME
# FILENAME FORMAT: 5_103_jpg.rf.xxxxxxx.jpg
# Label = first number before the first underscore
# -------------------------------------------------
def extract_finger_label(filename):
    """
    Extracts the finger count label from filenames like:
    '5_103_jpg.rf.d650acc4dde9.jpg' → 5
    """
    return int(filename.split("_")[0])


# -------------------------------------------------
# MAIN LANDMARK EXTRACTION LOOP
# -------------------------------------------------
def main():
    # Initialize MediaPipe Hands
    # normal detection confidence works fine for Roboflow images
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,   # dataset images are good quality
        min_tracking_confidence=0.5
    ) as hands, open(OUTPUT_CSV, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile)

        # -----------------------------------------
        # WRITE CSV HEADER
        # -----------------------------------------
        header = ["image_path", "label_fingers"]
        header += [f"x{i}" for i in range(21)]
        header += [f"y{i}" for i in range(21)]
        header += [f"z{i}" for i in range(21)]
        writer.writerow(header)

        # -----------------------------------------
        # LOOP THROUGH DATASET IMAGES
        # -----------------------------------------
        for filename in os.listdir(DATA_DIR):
            if not filename.lower().endswith(("png", "jpg", "jpeg")):
                continue

            img_path = os.path.join(DATA_DIR, filename)

            # 1) extract label (finger count 0–5)
            try:
                finger_label = extract_finger_label(filename)
            except Exception:
                print("Skipping bad filename:", filename)
                continue

            # 2) load image from disk
            img = cv2.imread(img_path)
            if img is None:
                print("Could not load:", filename)
                continue

            # 3) convert BGR → RGB for MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 4) run MediaPipe hand detection
            results = hands.process(img_rgb)

            if not results.multi_hand_landmarks:
                print("❌ No hand detected:", filename)
                continue

            landmarks = results.multi_hand_landmarks[0].landmark

            # 5) extract all 21 (x,y,z) landmark coordinates
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            zs = [lm.z for lm in landmarks]

            # 6) write to CSV
            writer.writerow([img_path, finger_label] + xs + ys + zs)
            print("✔ Processed:", filename)

    print("\n🎉 Done! Saved CSV to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
