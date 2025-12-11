import os
import csv
import cv2
import mediapipe as mp
import argparse

mp_hands = mp.solutions.hands

def extract_finger_label(filename):
    # Format: 5_103_jpg.rf.xxxxx.jpg → label = 5
    return int(filename.split("_")[0])

def process_folder(input_dir, output_csv):
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as hands, open(output_csv, "w", newline="") as csvfile:

        writer = csv.writer(csvfile)
        header = ["image_path", "label_fingers"]
        header += [f"x{i}" for i in range(21)]
        header += [f"y{i}" for i in range(21)]
        header += [f"z{i}" for i in range(21)]
        writer.writerow(header)

        files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith(("png", "jpg", "jpeg"))]

        for filename in files:
            img_path = os.path.join(input_dir, filename)

            # label
            try:
                label = extract_finger_label(filename)
            except Exception:
                print("Skipping invalid filename:", filename)
                continue

            img = cv2.imread(img_path)
            if img is None:
                print("Could not load:", img_path)
                continue

            # First pass: detect hand and handedness on original image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if not results.multi_hand_landmarks:
                print("No hand detected:", filename)
                continue

            handedness = results.multi_handedness[0].classification[0].label  # "Left" or "Right"

            # If it's a left hand, flip image and recompute landmarks so
            # that saved landmarks correspond to a "right-hand" orientation.
            if handedness == "Left":
                print(f"Detected LEFT hand in {filename}, flipping image for normalization.")
                img_flipped = cv2.flip(img, 1)  # horizontal flip (mirror)
                img_flipped_rgb = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB)
                results_flipped = hands.process(img_flipped_rgb)

                if not results_flipped.multi_hand_landmarks:
                    print("No hand detected after flip (skipping):", filename)
                    continue

                lm = results_flipped.multi_hand_landmarks[0].landmark
            else:
                # Right hand: use original landmarks
                lm = results.multi_hand_landmarks[0].landmark

            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            zs = [p.z for p in lm]

            writer.writerow([img_path, label] + xs + ys + zs)
            print("Processed:", filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to image folder")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    args = parser.parse_args()

    process_folder(args.input, args.output)
