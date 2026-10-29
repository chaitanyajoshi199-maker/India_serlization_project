"""
Real-time Camera Stream Gateway Testing Script.

Runs an interactive live video feed with an on-screen HUD (Heads-Up Display)
to visually monitor frame validation status and optical metrics.

Usage:
    python scripts/test_camera.py          # Opens default webcam (index 0)
    python scripts/test_camera.py 1        # Opens external USB camera (index 1)
    python scripts/test_camera.py video.mp4 # Tests with a prerecorded video
"""
import sys
import time
from pathlib import Path

import cv2

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capture import CameraStreamGateway, GatewayConfig


def run_live_test(source=0):
    print("\n" + "=" * 65)
    print(" REAL-TIME CAMERA STREAM GATEWAY - LIVE MONITOR")
    print(" Press 'q' or 'ESC' to exit")
    print("=" * 65 + "\n")

    # Set target_resolution=None to automatically adapt to whatever resolution
    # your webcam or industrial camera natively outputs (e.g. 720p, 1080p, 480p)
    config = GatewayConfig(
        target_resolution=None,       # None = Auto-accept camera's native resolution
        blur_threshold=100.0,         # Adjust if your lens has lower natural sharpness
        min_brightness=40.0,
        max_brightness=220.0,
        max_glare_pixel_ratio=0.02,
        min_contrast_spread=80.0,
    )

    gateway = CameraStreamGateway(source=source, config=config, auto_start=True)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            # Measure loop FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            # 1. Fetch the latest validated frame
            res = gateway.get_valid_frame(timeout=1.0)

            # Get the display frame (either valid frame or raw buffer for visualization)
            display_frame = res.frame if res.frame is not None else gateway._last_raw_frame

            if display_frame is None:
                print("Waiting for camera feed...")
                time.sleep(0.05)
                continue

            # Work on a copy for visualization HUD
            vis = display_frame.copy()
            h, w = vis.shape[:2]

            # 2. Draw HUD based on validation status
            if res.is_valid:
                color = (0, 220, 0)  # Bright Green
                status_text = "STATUS: OK (ACCEPTED)"
                err_text = "All Optical Checks Passed"
            else:
                color = (0, 0, 255)  # Bright Red
                status_text = f"STATUS: DROPPED ({res.error_code.value if res.error_code else 'UNKNOWN'})"
                err_text = f"Frame dropped before QR/OCR"

            # Top Banner Box
            cv2.rectangle(vis, (10, 10), (w - 10, 110), (20, 20, 20), -1)
            cv2.rectangle(vis, (10, 10), (w - 10, 110), color, 2)

            # Status Title
            cv2.putText(vis, status_text, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(vis, err_text, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Metrics readout
            blur_val = res.metrics.get("blur_laplacian_var", "N/A")
            bright_val = res.metrics.get("mean_brightness", "N/A")
            contrast_val = res.metrics.get("contrast_gap", "N/A")

            metrics_str = f"FPS: {fps:.1f} | Res: {w}x{h} | Blur: {blur_val} (min:100) | Bright: {bright_val} (40-220) | Contrast: {contrast_val} (min:80)"
            cv2.putText(vis, metrics_str, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # 3. Show Window
            cv2.imshow("Camera Stream Gateway - Quality Inspector", vis)

            # Exit on 'q' or ESC
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        gateway.stop()
        cv2.destroyAllWindows()
        print("\nGateway stopped cleanly.")


if __name__ == "__main__":
    # If user passes an argument (e.g. 0, 1, or path to video)
    cam_source = 0
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        cam_source = int(arg) if arg.isdigit() else arg

    run_live_test(cam_source)
