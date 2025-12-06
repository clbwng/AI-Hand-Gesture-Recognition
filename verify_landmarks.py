import csv
import random
import cv2
import numpy as np

CSV_PATH = "fingers_landmarks_clean.csv"
NUM_SAMPLES = 5

def load_csv_rows(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            rows.append(row)
    return rows

def draw_landmarks(img, xs, ys):
    h, w = img.shape[:2]
    for x, y in zip(xs, ys):
        px = int(float(x) * w)
        py = int(float(y) * h)

        # small filled dot
        cv2.circle(img, (px, py), 2, (0, 255, 0), -1)

        # thin outline to improve visibility on bright backgrounds
        cv2.circle(img, (px, py), 3, (0, 0, 0), 1)

    return img

def show_image_with_exit(window_name, img):
    cv2.imshow(window_name, img)

    while True:
        key = cv2.waitKey(50) & 0xFF

        # Press any key (or q/ESC) to continue
        if key != 255:
            break

        # If window is closed manually → break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyWindow(window_name)

def main():
    print("Loading CSV...")
    rows = load_csv_rows(CSV_PATH)

    print(f"Selecting {NUM_SAMPLES} random samples...")
    samples = random.sample(rows, NUM_SAMPLES)

    for idx, row in enumerate(samples):
        img_path = row[0]
        label_hand = row[1]
        label_fingers = row[2]

        img = cv2.imread(img_path)
        if img is None:
            print("Could not load image:", img_path)
            continue

        xs = row[3:3+21]
        ys = row[3+21:3+21*2]

        img_drawn = draw_landmarks(img.copy(), xs, ys)

        # Resize for visibility
        scale = 700 / img_drawn.shape[1]
        new_size = (700, int(img_drawn.shape[0] * scale))
        img_resized = cv2.resize(img_drawn, new_size)

        window_name = f"Sample {idx+1}/{NUM_SAMPLES} | Hand={label_hand}, Fingers={label_fingers}"
        print("Showing:", window_name)

        show_image_with_exit(window_name, img_resized)

    print("Done!")

if __name__ == "__main__":
    main()
