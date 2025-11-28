import sys
import cv2
import mediapipe as mp

def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python mp_hand_show_large.py /path/to/image.png")
    #     sys.exit(1)

    image_path = "hand.png"
    # image_path = "fist.png"
    print(f"Loading: {image_path}")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot load image: {image_path}")
        sys.exit(1)

    # Mediapipe setup
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.01
    ) as hands:

        # Convert BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect hands
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            print("No hands detected.")
        else:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                print(f"\n=== Hand {hand_idx} Landmarks ===")
                for i, lm in enumerate(hand_landmarks.landmark):
                    print(f"  {i}: x={lm.x:.5f}, y={lm.y:.5f}, z={lm.z:.5f}")

                # Draw all landmarks on image
                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # -----------------------
        # Resize image for display
        # -----------------------
        display_width = 700
        h, w = img.shape[:2]
        scale = display_width / w
        new_size = (display_width, int(h * scale))

        img_large = cv2.resize(img, new_size)

        # Show window
        cv2.imshow("MediaPipe Hand Landmarks (Large)", img_large)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
