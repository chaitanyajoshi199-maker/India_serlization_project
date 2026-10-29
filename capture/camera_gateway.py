"""
High-Performance Global Frame Extraction & Optical Validation Gateway.

Acts as the primary hardware/optical firewall for live camera feeds (DirectShow, V4L2, RTSP, GigE)
prior to downstream QR decoding and 9-parameter OCR modules.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

# Configure module-level logger
logger = logging.getLogger("CameraStreamGateway")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [CameraGateway] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class FrameErrorCode(str, Enum):
    """Standardized error codes for rejected camera frames."""
    ERR_STREAM_DISCONNECTED = "ERR_STREAM_DISCONNECTED"
    ERR_INVALID_FRAME_DIMENSIONS = "ERR_INVALID_FRAME_DIMENSIONS"
    ERR_FRAME_BLURRY = "ERR_FRAME_BLURRY"
    ERR_UNDER_EXPOSED = "ERR_UNDER_EXPOSED"
    ERR_OVER_EXPOSED_GLARE = "ERR_OVER_EXPOSED_GLARE"
    ERR_LOW_CONTRAST = "ERR_LOW_CONTRAST"
    ERR_QUEUE_TIMEOUT = "ERR_QUEUE_TIMEOUT"


@dataclass(frozen=True)
class GatewayConfig:
    """Configurable thresholds for optical validation."""
    target_resolution: Optional[Tuple[int, int]] = (1920, 1080)  # (width, height)
    blur_threshold: float = 100.0                               # Min Laplacian variance
    min_brightness: float = 40.0                                 # Min mean grayscale intensity
    max_brightness: float = 220.0                                # Max mean grayscale intensity
    max_glare_pixel_ratio: float = 0.02                          # Max fraction of saturated pixels (>=254)
    min_contrast_spread: float = 80.0                            # Min (95th - 5th percentile) intensity gap
    stride_for_percentiles: int = 4                              # Spatial subsampling stride for 60+ FPS stat checks


@dataclass
class FrameValidationResult:
    """Result container returned for every frame evaluated."""
    is_valid: bool
    frame: Optional[np.ndarray] = None
    error_code: Optional[FrameErrorCode] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    frame_index: int = 0
    timestamp: float = field(default_factory=time.time)


class CameraStreamGateway:
    """
    Thread-safe camera frame ingestion and optical quality gatekeeper.

    Features:
    - Runs an isolated frame-grabbing loop in a background daemon thread to eliminate
      driver-level buffer queue lag (guarantees zero latency on live conveyor belts).
    - Executes Check A, B, C, D in strict sequence with microsecond-level efficiency.
    - Drops degraded frames instantly, shielding downstream OCR / QR engines from garbage.
    """

    def __init__(
        self,
        source: Union[int, str, cv2.VideoCapture],
        config: Optional[GatewayConfig] = None,
        auto_start: bool = True,
    ) -> None:
        """
        Initialize the stream gateway.

        :param source: Device index (e.g., 0, 1), video file path, or RTSP/GStreamer URL.
        :param config: Operational thresholds. Defaults to GatewayConfig() if None.
        :param auto_start: Automatically start the background capture thread.
        """
        self.source = source
        self.config = config or GatewayConfig()

        self._cap: Optional[cv2.VideoCapture] = None
        self._external_cap: bool = False

        if isinstance(source, cv2.VideoCapture) or (hasattr(source, "read") and hasattr(source, "isOpened")):
            self._cap = source
            self._external_cap = True
        else:
            self._init_capture()

        # Thread synchronization primitives
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

        # State storage
        self._last_raw_frame: Optional[np.ndarray] = None
        self._last_capture_success: bool = False
        self._frame_counter: int = 0
        self._dropped_counter: int = 0

        if auto_start:
            self.start()

    def _init_capture(self) -> None:
        """Initialize the OpenCV VideoCapture device with optimal low-latency flags."""
        logger.info("Initializing camera source: %s", self.source)
        if isinstance(self.source, int):
            # Hardware webcam/directshow on Windows
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            logger.error("Failed to open video source: %s", self.source)
            return

        # Attempt to request target resolution at hardware driver level
        if self.config.target_resolution:
            w, h = self.config.target_resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        # Minimize driver buffer queue size to 1 where supported
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self) -> CameraStreamGateway:
        """Start the background grabbing worker thread."""
        if self._is_running:
            return self

        if self._cap is None or not self._cap.isOpened():
            self._init_capture()

        self._is_running = True
        self._thread = threading.Thread(target=self._capture_worker, name="CameraGrabberThread", daemon=True)
        self._thread.start()
        logger.info("CameraStreamGateway background capture thread started.")
        return self

    def _capture_worker(self) -> None:
        """Dedicated thread continuously reading the camera to prevent buffer lag."""
        while self._is_running:
            if self._cap is None or not self._cap.isOpened():
                with self._lock:
                    self._last_capture_success = False
                    self._last_raw_frame = None
                self._new_frame_event.set()
                time.sleep(0.05)
                continue

            success, frame = self._cap.read()

            with self._lock:
                self._last_capture_success = success
                self._last_raw_frame = frame
                self._frame_counter += 1

            # Notify consumer thread that fresh hardware frame has arrived
            self._new_frame_event.set()

    def get_valid_frame(self, timeout: float = 1.0) -> FrameValidationResult:
        if not self._new_frame_event.wait(timeout=timeout):
            return FrameValidationResult(
                is_valid=False,
                error_code=FrameErrorCode.ERR_QUEUE_TIMEOUT,
                metrics={"reason": "Timeout waiting for camera frame."},
                frame_index=self._frame_counter,
            )

        self._new_frame_event.clear()

        with self._lock:
            success = self._last_capture_success
            raw_frame = self._last_raw_frame
            idx = self._frame_counter

        metrics: Dict[str, Any] = {}

        # =========================================================================
        # Check A: Stream Connection & Dimensions
        # =========================================================================
        if not success or raw_frame is None or raw_frame.size == 0:
            self._dropped_counter += 1
            return FrameValidationResult(
                is_valid=False,
                error_code=FrameErrorCode.ERR_STREAM_DISCONNECTED,
                metrics=metrics,
                frame_index=idx,
            )

        h, w = raw_frame.shape[:2]
        metrics["width"], metrics["height"] = w, h

        if self.config.target_resolution is not None:
            expected_w, expected_h = self.config.target_resolution
            if (w, h) != (expected_w, expected_h):
                self._dropped_counter += 1
                return FrameValidationResult(
                    is_valid=False,
                    error_code=FrameErrorCode.ERR_INVALID_FRAME_DIMENSIONS,
                    metrics=metrics,
                    frame_index=idx,
                )

        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY) if raw_frame.ndim == 3 else raw_frame

        # =========================================================================
        # Check B (RUN FIRST): Illumination, Exposure & Presence Gate
        # =========================================================================
        mean_brightness = float(cv2.mean(gray)[0])
        metrics["mean_brightness"] = round(mean_brightness, 2)

        # 1. Dark screen / Empty conveyor / Lens covered
        if mean_brightness < self.config.min_brightness:
            self._dropped_counter += 1
            return FrameValidationResult(
                is_valid=False,
                error_code=FrameErrorCode.ERR_UNDER_EXPOSED,
                metrics=metrics,
                frame_index=idx,
            )

        # 2. Overexposed scene
        if mean_brightness > self.config.max_brightness:
            self._dropped_counter += 1
            return FrameValidationResult(
                is_valid=False,
                error_code=FrameErrorCode.ERR_OVER_EXPOSED_GLARE,
                metrics=metrics,
                frame_index=idx,
            )

        # 3. Localized Glare hotspot (reflective metallic/blister foil)
        stride = self.config.stride_for_percentiles
        sampled_gray = gray[::stride, ::stride]
        saturated_ratio = float(np.mean(sampled_gray >= 254))
        metrics["glare_saturated_ratio"] = round(saturated_ratio, 4)

        if saturated_ratio > self.config.max_glare_pixel_ratio:
            self._dropped_counter += 1
            return FrameValidationResult(
                is_valid=False,
                error_code=FrameErrorCode.ERR_OVER_EXPOSED_GLARE,
                metrics=metrics,
                frame_index=idx,
            )

        # =========================================================================
        # Check C: Sharpness / Focus (ONLY runs if frame is adequately illuminated)
        # =========================================================================
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_var = float(laplacian.var())
        metrics["blur_laplacian_var"] = round(blur_var, 2)

        if blur_var < self.config.blur_threshold:
            self._dropped_counter += 1
            return FrameValidationResult(
                is_valid=False,
                error_code=FrameErrorCode.ERR_FRAME_BLURRY,
                metrics=metrics,
                frame_index=idx,
            )

        # =========================================================================
        # Check D: Dynamic Range / Contrast Level
        # =========================================================================
        p95, p5 = np.percentile(sampled_gray, (95, 5))
        contrast_gap = float(p95 - p5)
        metrics["contrast_gap"] = round(contrast_gap, 2)

        if contrast_gap < self.config.min_contrast_spread:
            self._dropped_counter += 1
            return FrameValidationResult(
                is_valid=False,
                error_code=FrameErrorCode.ERR_LOW_CONTRAST,
                metrics=metrics,
                frame_index=idx,
            )

        # All checks passed: Deliver to downstream
        return FrameValidationResult(
            is_valid=True,
            frame=raw_frame,
            error_code=None,
            metrics=metrics,
            frame_index=idx,
        )


    def stop(self) -> None:
        """Gracefully release camera resources and terminate background threads."""
        self._is_running = False
        self._new_frame_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if not self._external_cap and self._cap is not None:
            self._cap.release()
            self._cap = None

        logger.info("CameraStreamGateway stopped. Total processed: %d, Dropped: %d",
                    self._frame_counter, self._dropped_counter)

    def __enter__(self) -> CameraStreamGateway:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


# =============================================================================
# Demonstration / Mock Entry Point
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" CAMERA STREAM GATEWAY - INTEGRATION TEST & VALIDATION BENCH")
    print("=" * 70 + "\n")

    # Create a mock video source simulating a real camera feed with 5 frames:
    # 1: Clean high-quality frame (SHOULD PASS)
    # 2: Defocused blurry frame (FAIL Check B)
    # 3: Underexposed black frame (FAIL Check C)
    # 4: Glare-washed white frame (FAIL Check C)
    # 5: Washed-out low-contrast gray frame (FAIL Check D)
    
    test_w, test_h = 1920, 1080
    config = GatewayConfig(
        target_resolution=(test_w, test_h),
        blur_threshold=100.0,
        min_brightness=40.0,
        max_brightness=220.0,
        max_glare_pixel_ratio=0.02,
        min_contrast_spread=80.0,
    )

    class MockCameraSource:
        """Simulates camera read() yielding varying optical scenarios."""
        def __init__(self):
            self.step = 0

        def isOpened(self):
            return True

        def read(self):
            time.sleep(0.04)  # Simulate 25 FPS frame interval
            self.step += 1

            if self.step == 1:
                # 1. High-contrast sharp synthetic pattern (PASS)
                img = np.zeros((test_h, test_w, 3), dtype=np.uint8)
                img[:] = 50
                cv2.circle(img, (960, 540), 300, (230, 230, 230), -1)
                cv2.putText(img, "VALID QR & OCR REGION", (600, 540),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 4)
                return True, img

            elif self.step == 2:
                # 2. Defocused / Blurry image (FAIL Check B)
                img = np.zeros((test_h, test_w, 3), dtype=np.uint8)
                img[:] = 100
                cv2.putText(img, "BLURRY TEXT", (600, 540), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (200, 200, 200), 2)
                blurred = cv2.GaussianBlur(img, (61, 61), 0)
                return True, blurred

            elif self.step == 3:
                # 3. Underexposed image with sharp noise/edges (FAIL Check C: mean < 40)
                # High frequency noise ensures blur_var > 100, isolating the underexposure check
                noise = np.random.randint(0, 35, (test_h, test_w), dtype=np.uint8)
                img = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
                return True, img

            elif self.step == 4:
                # 4. Overexposed glare hotspot (FAIL Check C: glare_ratio > 0.02)
                # Sharp image with large saturated white blob (>=254)
                noise = np.random.randint(50, 160, (test_h, test_w), dtype=np.uint8)
                img = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
                cv2.circle(img, (960, 540), 220, (255, 255, 255), -1)
                return True, img

            elif self.step == 5:
                # 5. Low contrast / flat image (FAIL Check D)
                # Intensities only range between 110 and 130 (spread = 20 < 80)
                img = np.random.randint(110, 130, (test_h, test_w, 3), dtype=np.uint8)
                return True, img

            return False, None

    # Instantiate gateway with the mock source (auto_start=False for manual stepping)
    mock = MockCameraSource()
    gateway = CameraStreamGateway(source=mock, config=config, auto_start=False)

    print("Simulating downstream inspection loop receiving 5 consecutive frames:\n")

    for i in range(1, 6):
        ret, frame = mock.read()
        with gateway._lock:
            gateway._last_capture_success = ret
            gateway._last_raw_frame = frame
            gateway._frame_counter += 1
        gateway._new_frame_event.set()

        res = gateway.get_valid_frame(timeout=0.5)

        if res.is_valid:
            print(f">>> DOWNSTREAM ACTION: Frame #{res.frame_index} ACCEPTED! Dispatching to QR / OCR engine.")
            print(f"    Metrics: {res.metrics}\n")
        else:
            print(f"--- DOWNSTREAM ACTION: Frame #{res.frame_index} REJECTED [Error: {res.error_code}]. Frame dropped.")
            print(f"    Metrics: {res.metrics}\n")

    gateway.stop()
    print("=" * 70)
    print(" TEST COMPLETED.")
    print("=" * 70)
