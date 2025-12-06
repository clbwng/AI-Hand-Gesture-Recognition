import os
import csv
import cv2
import mediapipe as mp

DATA_DIR = "images/train"
OUTPUT_CSV = "fingers_landmarks.csv"

mp_hands = mp.solutions.hands

def extract_labels_from_filename(filename):
    """
    Filename: <uuid>_<finger><hand>.png
              example: 16779a42..._5R.png
    - Second last char = finger count (0–5)
    - Last char = hand (L/R)
    """
    name, _ = os.path.splitext(filename)
    finger = name[-2]        # e.g. '5'
    hand = name[-1].upper()  # 'L' or 'R'
    return hand, int(finger)

def main():
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.01,   # lower for weak detections
        min_tracking_confidence=0.01     # must also be lowered
    ) as hands, open(OUTPUT_CSV, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile)

        # CSV header
        header = ["image_path", "label_hand", "label_fingers"]
        header += [f"x{i}" for i in range(21)]
        header += [f"y{i}" for i in range(21)]
        header += [f"z{i}" for i in range(21)]
        writer.writerow(header)

        # Iterate through training images
        for filename in os.listdir(DATA_DIR):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(DATA_DIR, filename)

            # -------- Parse labels from filename --------
            try:
                label_hand, label_fingers = extract_labels_from_filename(filename)
            except Exception:
                print("Skipping file with invalid label format:", filename)
                continue

            # -------- Load image --------
            img = cv2.imread(img_path)
            if img is None:
                print("Could not load image:", img_path)
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # Hand detection check
            if not results.multi_hand_landmarks:
                print("No hand detected:", img_path)
                continue

            # We're only using the first detected hand
            landmarks = results.multi_hand_landmarks[0].landmark

            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            zs = [lm.z for lm in landmarks]

            # Write row: labels + landmarks
            writer.writerow([img_path, label_hand, label_fingers] + xs + ys + zs)

            print("Processed:", filename)

    print("\nDone! Saved CSV to:", OUTPUT_CSV)

if __name__ == "__main__":
    main()
