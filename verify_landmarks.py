import csv
import random
import cv2
import numpy as np

CSV_PATH = "finger_landmarks.csv"   # <-- your generated CSV
NUM_SAMPLES = 5                     # how many random samples to show


# ----------------------------------------------------------
# Load CSV rows into memory
# ----------------------------------------------------------
def load_csv_rows(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header row
        for row in reader:
            rows.append(row)
    return rows


# ----------------------------------------------------------
# Draw the 21 MediaPipe landmarks on the image
# ----------------------------------------------------------
def draw_landmarks(img, xs, ys):
    h, w = img.shape[:2]

    for x, y in zip(xs, ys):
        px = int(float(x) * w)
        py = int(float(y) * h)

        # filled dot
        cv2.circle(img, (px, py), 2, (0, 255, 0), -1)

        # outer edge for visibility
        cv2.circle(img, (px, py), 3, (0, 0, 0), 1)

    return img


# ----------------------------------------------------------
# Show image until a key is pressed or window is closed
# ----------------------------------------------------------
def show_image_with_exit(window_name, img):
    cv2.imshow(window_name, img)

    while True:
        key = cv2.waitKey(50) & 0xFF

        # any key -> exit this image
        if key != 255:
            break

        # user manually closes window
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyWindow(window_name)


# ----------------------------------------------------------
# Main verification procedure
# ----------------------------------------------------------
def main():
    print("Loading CSV...")
    rows = load_csv_rows(CSV_PATH)

    print(f"Selecting {NUM_SAMPLES} random samples...")
    samples = random.sample(rows, NUM_SAMPLES)

    for idx, row in enumerate(samples):
        img_path = row[0]
        label_fingers = row[1]          # <-- only one label now

        img = cv2.imread(img_path)
        if img is None:
            print("Could not load:", img_path)
            continue

        # Extract landmark vectors
        xs = row[2:2+21]
        ys = row[2+21:2+21*2]

        # Draw landmarks
        img_drawn = draw_landmarks(img.copy(), xs, ys)

        # Resize for display
        scale = 800 / img_drawn.shape[1]
        new_size = (800, int(img_drawn.shape[0] * scale))
        img_resized = cv2.resize(img_drawn, new_size)

        window_name = f"Sample {idx+1}/{NUM_SAMPLES} | Fingers={label_fingers}"
        print("Showing:", window_name)

        show_image_with_exit(window_name, img_resized)

    print("Done!")


if __name__ == "__main__":
    main()
