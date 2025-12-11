# @misc{
#         fingers-numbers_dataset,
#         title = { Fingers Numbers Dataset },
#         type = { Open Source Dataset },
#         author = { Hands },
#         howpublished = { \url{ https://universe.roboflow.com/hands-rirpj/fingers-numbers } },
#         url = { https://universe.roboflow.com/hands-rirpj/fingers-numbers },
#         journal = { Roboflow Universe },
#         publisher = { Roboflow },
#         year = { 2023 },
#         month = { jun },
#         note = { visited on 2025-12-10 },
#     }

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
            except:
                print("Skipping invalid filename:", filename)
                continue

            img = cv2.imread(img_path)
            if img is None:
                print("Could not load:", img_path)
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if not results.multi_hand_landmarks:
                print("No hand detected:", filename)
                continue

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
