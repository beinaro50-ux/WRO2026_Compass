#!/usr/bin/env python3

import cv2
import numpy as np

# ----------------------------------------------------------------------
CAMERA_INDEX = 0          # Microdia Integrated_Webcam_HD -> /dev/video2
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SAMPLE_BOX_SIZE = 100       # size (px) of the square sample region at center
# ----------------------------------------------------------------------

# A practical set of common color names -> RGB values.
# (Standard/basic color names, not any copyrighted material.)
NAMED_COLORS = {
    "Black": (0, 0, 0), "White": (255, 255, 255), "Gray": (128, 128, 128),
    "Light Gray": (211, 211, 211), "Dark Gray": (64, 64, 64), "Silver": (192, 192, 192),
    "Red": (255, 0, 0), "Dark Red": (139, 0, 0), "Crimson": (220, 20, 60),
    "Maroon": (128, 0, 0), "Firebrick": (178, 34, 34), "Tomato": (255, 99, 71),
    "Orange Red": (255, 69, 0), "Orange": (255, 165, 0), "Dark Orange": (255, 140, 0),
    "Gold": (255, 215, 0), "Yellow": (255, 255, 0), "Khaki": (240, 230, 140),
    "Olive": (128, 128, 0), "Yellow Green": (154, 205, 50), "Green": (0, 128, 0),
    "Lime": (0, 255, 0), "Dark Green": (0, 100, 0), "Forest Green": (34, 139, 34),
    "Sea Green": (46, 139, 87), "Spring Green": (0, 255, 127), "Teal": (0, 128, 128),
    "Turquoise": (64, 224, 208), "Cyan": (0, 255, 255), "Light Blue": (173, 216, 230),
    "Sky Blue": (135, 206, 235), "Steel Blue": (70, 130, 180), "Dodger Blue": (30, 144, 255),
    "Blue": (0, 0, 255), "Navy": (0, 0, 128), "Royal Blue": (65, 105, 225),
    "Indigo": (75, 0, 130), "Purple": (128, 0, 128), "Violet": (238, 130, 238),
    "Magenta": (255, 0, 255), "Orchid": (218, 112, 214), "Pink": (255, 192, 203),
    "Hot Pink": (255, 105, 180), "Deep Pink": (255, 20, 147), "Salmon": (250, 128, 114),
    "Coral": (255, 127, 80), "Brown": (165, 42, 42), "Chocolate": (210, 105, 30),
    "Saddle Brown": (139, 69, 19), "Sienna": (160, 82, 45), "Tan": (210, 180, 140),
    "Beige": (245, 245, 220), "Ivory": (255, 255, 240), "Wheat": (245, 222, 179),
}


def closest_color_name(rgb):
    """Return the nearest named color to the given (R,G,B) tuple."""
    r, g, b = rgb
    best_name, best_dist = None, float("inf")
    for name, (nr, ng, nb) in NAMED_COLORS.items():
        dist = (r - nr) ** 2 + (g - ng) ** 2 + (b - nb) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera at index {index}. "
            "Check `v4l2-ctl --list-devices` and try a different index."
        )
    return cap


def main():
    cap = open_camera(CAMERA_INDEX)
    box_size = SAMPLE_BOX_SIZE

    print("Point the camera at an object. Press 'q' to quit, 's' to snapshot, '+/-' to resize sample box.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed, retrying...")
                continue

            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            half = box_size // 2

            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w, cx + half), min(h, cy + half)

            roi = frame[y1:y2, x1:x2]
            avg_bgr = roi.reshape(-1, 3).mean(axis=0)
            b, g, r = avg_bgr
            rgb = (int(r), int(g), int(b))

            name = closest_color_name(rgb)

            display = frame.copy()
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Info panel
            cv2.rectangle(display, (0, 0), (300, 70), (30, 30, 30), -1)
            cv2.putText(display, f"Color: {name}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"RGB: {rgb}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Swatch of the detected average color
            cv2.rectangle(display, (250, 5), (295, 65), (int(b), int(g), int(r)), -1)
            cv2.rectangle(display, (250, 5), (295, 65), (255, 255, 255), 1)

            cv2.imshow("Color Name Reader", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                import time
                fname = f"color_snapshot_{int(time.time())}.png"
                cv2.imwrite(fname, display)
                print(f"Saved {fname}  -> detected: {name} {rgb}")
            elif key in (ord('+'), ord('=')):
                box_size = min(min(w, h) - 2, box_size + 10)
            elif key == ord('-'):
                box_size = max(10, box_size - 10)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
