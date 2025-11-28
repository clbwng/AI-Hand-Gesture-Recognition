import os
import cv2
import mediapipe as mp
import kagglehub

# 1. Download dataset
path = kagglehub.dataset_download("koryakinp/fingers")
print("Dataset path:", path)
print("Contents:", os.listdir(path))

# 👉 ADJUST THIS LINE once you see what prints above
dataset_root = path  # or os.path.join(path, "FingerData") etc.

zero_dir = os.path.join(dataset_root, "0")
five_dir = os.path.join(dataset_root, "5")

print("Zero dir:", zero_dir)
print("Five dir:", five_dir)

def pick_samples(folder, n=3):
    if not os.path.isdir(folder):
        print(f"[WARN] Folder does not exist: {folder}")
        return []
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    return files[:n]

zero_samples = pick_samples(zero_dir)
five_samples = pick_samples(five_dir)

print("Zero-finger samples:", zero_samples)
print("Five-finger samples:", five_samples)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def test_image(img_path, visualize=True):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load: {img_path}")
        return

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    print(f"\nTesting {img_path}")
    if result.multi_hand_landmarks:
        print(" → Landmarks detected! 👍")
        if visualize:
            annotated = img.copy()
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
            cv2.imshow("Landmarks", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print(" → No hand detected ❌")

print("\n=== Testing zero-finger images ===")
for img_path in zero_samples:
    test_image(img_path)

print("\n=== Testing five-finger images ===")
for img_path in five_samples:
    test_image(img_path)
