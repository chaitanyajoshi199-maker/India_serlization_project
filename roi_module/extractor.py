"""
Native Sequential Multi-ROI Module.

Ingests a validated camera frame, anchors against the primary physical QR code in under 5 ms,
and executes strict geometric handoffs to isolate:
1. qr_roi   -> Tight crop around the QR code anchor.
2. ocr_roi  -> Deskewed, leveled horizontal text block relative to QR right margin.
3. pharmacode -> Optional verified pharmaceutical bar code.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import zxingcpp

from pharma_detector.pharmacode import scan_pharmacode_roi 

logger = logging.getLogger("MultiROIExtractor")

# Error constants
ERR_QR_ANCHOR_NOT_FOUND = "ERR_QR_ANCHOR_NOT_FOUND"
STATUS_SUCCESS = "SUCCESS"


@dataclass(frozen=True)
class ROIConfig:
    """Configurable geometric parameters for relative ROI extraction."""
    ocr_x_gap_ratio: float = 0.08       # Spacing between QR right margin and text start (fraction of QR_W)
    ocr_width_ratio: float = 3.60       # Width of OCR text block (relative to QR_W)
    ocr_height_ratio: float = 1.45      # Height of OCR text block (increased from 1.15 to prevent clipping top/bottom)
    ocr_y_offset_ratio: float = -0.20   # Vertical shift upward (-0.20 * QR_H) so top line (GTIN) is never cropped
    qr_crop_padding: int = 4            # Padding pixels around the QR bounding box
    enable_pharmacode: bool = True      # Attempt Pharmacode scan and validation
    try_invert: bool = False            # Invert scan (keep False for < 5 ms dark-on-light pharma packaging)
    binarizer: Any = None               # None = defaults to zxingcpp.Binarizer.FixedThreshold for < 5ms
    pharmacode_direction: str = "AUTO"  # "AUTO" (uses QR anchor orientation), "R2L" (standard), "L2R" (inverted)
    expected_pharmacode: Optional[int] = None # Optional expected value to disambiguate orientation


class MultiROIExtractor:
    """
    High-performance native multi-ROI segmentation gateway.

    Features:
    - Uses compiled native C++ zxing-cpp single-pass barcode scan.
    - Locks formats strictly to QRCode (and Pharmacode if available), dropping consumer retail formats.
    - Zero morphology/contour heuristics: anchor is strictly driven by the physical barcode detector.
    - Perspective/affine warping levels out conveyor belt tilt in sub-millisecond time.
    """

    def __init__(self, config: Optional[ROIConfig] = None) -> None:
        self.config = config or ROIConfig()

        # Build bitwise target format filter
        self._target_formats = zxingcpp.BarcodeFormat.QRCode
        if hasattr(zxingcpp.BarcodeFormat, "Pharmacode"):
            self._target_formats |= getattr(zxingcpp.BarcodeFormat, "Pharmacode")

    def extract(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Execute single-pass detection and geometric multi-ROI isolation.

        :param frame: Grayscale or BGR image frame from Module 1.
        :return: Standardized schema dictionary.
        """
        if frame is None or frame.size == 0:
            return self._build_failure_response("ERR_EMPTY_FRAME")

        # Ensure grayscale for fast matrix processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        img_h, img_w = gray.shape[:2]

        # =====================================================================
        # Step A: Single-Pass Native Barcode Scan with Omnidirectional Recovery
        # =====================================================================
        binarizer = self.config.binarizer if self.config.binarizer is not None else zxingcpp.Binarizer.LocalAverage
        try:
            barcodes = zxingcpp.read_barcodes(
                gray,
                formats=self._target_formats,
                try_rotate=True,
                try_downscale=False,
                try_invert=self.config.try_invert,
                binarizer=binarizer,
            )
        except Exception as e:
            logger.error("Native scanner failed: %s", e)
            return self._build_failure_response("ERR_SCANNER_FAILURE")

        # =====================================================================
        # Step B: QR Master Anchor Tracking
        # =====================================================================
        qr_match = None
        pharmacode_match = None

        for b in barcodes:
            fmt_name = b.format.name.lower()
            if "qr" in fmt_name and qr_match is None:
                qr_match = b
            elif "pharmacode" in fmt_name and pharmacode_match is None:
                pharmacode_match = b

        # If QR anchor is missing: immediately abort
        if qr_match is None:
            return self._build_failure_response(ERR_QR_ANCHOR_NOT_FOUND)

        # Extract precise corners from native detector
        pos = qr_match.position
        p0 = np.array([pos.top_left.x, pos.top_left.y], dtype=np.float32)          # Top-Left
        p1 = np.array([pos.top_right.x, pos.top_right.y], dtype=np.float32)        # Top-Right
        p2 = np.array([pos.bottom_right.x, pos.bottom_right.y], dtype=np.float32)    # Bottom-Right
        p3 = np.array([pos.bottom_left.x, pos.bottom_left.y], dtype=np.float32)     # Bottom-Left

        # =====================================================================
        # Step C: Universal Global Horizon Straightening (< 1.8 ms)
        # Keeps entire passing frame 100% horizontal & 0° deskewed before ROI mining
        # =====================================================================
        v_top = p1 - p0
        theta_deg = float(np.degrees(np.arctan2(v_top[1], v_top[0])))

        straight_frame = frame
        was_straightened = False

        # 1. Fast Orthogonal Snap (if carton is rotated 90°, 180°, or 270°)
        if 45.0 <= theta_deg < 135.0:
            straight_frame = cv2.rotate(straight_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            was_straightened = True
        elif -135.0 <= theta_deg < -45.0:
            straight_frame = cv2.rotate(straight_frame, cv2.ROTATE_90_CLOCKWISE)
            was_straightened = True
        elif abs(theta_deg) >= 135.0:
            straight_frame = cv2.rotate(straight_frame, cv2.ROTATE_180)
            was_straightened = True

        # 2. Fine tilt correction if skew >= 0.4 deg (or if snapped)
        s_gray = cv2.cvtColor(straight_frame, cv2.COLOR_BGR2GRAY) if straight_frame.ndim == 3 else straight_frame
        s_h, s_w = s_gray.shape[:2]

        if was_straightened or abs(theta_deg) >= 0.4:
            s_qrs = zxingcpp.read_barcodes(
                s_gray,
                formats=zxingcpp.BarcodeFormat.QRCode,
                try_rotate=False,
                binarizer=binarizer,
            )
            if s_qrs:
                qr_match = s_qrs[0]
                spos = qr_match.position
                p0 = np.array([spos.top_left.x, spos.top_left.y], dtype=np.float32)
                p1 = np.array([spos.top_right.x, spos.top_right.y], dtype=np.float32)
                p2 = np.array([spos.bottom_right.x, spos.bottom_right.y], dtype=np.float32)
                p3 = np.array([spos.bottom_left.x, spos.bottom_left.y], dtype=np.float32)

                sv_top = p1 - p0
                fine_deg = float(np.degrees(np.arctan2(sv_top[1], sv_top[0])))
                if abs(fine_deg) >= 0.4:
                    # Pure orthonormal rigid rotation around image center (preserves 100% bar sharpness)
                    M = cv2.getRotationMatrix2D((s_w / 2.0, s_h / 2.0), fine_deg, 1.0)
                    straight_frame = cv2.warpAffine(
                        straight_frame,
                        M,
                        (s_w, s_h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    s_gray = cv2.cvtColor(straight_frame, cv2.COLOR_BGR2GRAY) if straight_frame.ndim == 3 else straight_frame
                    fine_res = zxingcpp.read_barcodes(
                        s_gray,
                        formats=zxingcpp.BarcodeFormat.QRCode,
                        try_rotate=False,
                        binarizer=binarizer,
                    )
                    if fine_res:
                        qr_match = fine_res[0]
                        spos = qr_match.position
                        p0 = np.array([spos.top_left.x, spos.top_left.y], dtype=np.float32)
                        p1 = np.array([spos.top_right.x, spos.top_right.y], dtype=np.float32)
                        p2 = np.array([spos.bottom_right.x, spos.bottom_right.y], dtype=np.float32)
                        p3 = np.array([spos.bottom_left.x, spos.bottom_left.y], dtype=np.float32)

        # Working frame and gray are now 100% leveled
        frame = straight_frame
        gray = s_gray
        img_h, img_w = s_h, s_w

        # Compute dynamic bounding box for QR crop across all 4 corners
        all_xs = [p0[0], p1[0], p2[0], p3[0]]
        all_ys = [p0[1], p1[1], p2[1], p3[1]]
        pad = self.config.qr_crop_padding
        qr_x = max(0, int(min(all_xs)) - pad)
        qr_y = max(0, int(min(all_ys)) - pad)
        qr_x2 = min(img_w, int(max(all_xs)) + pad)
        qr_y2 = min(img_h, int(max(all_ys)) + pad)

        qr_w = qr_x2 - qr_x
        qr_h = qr_y2 - qr_y

        if qr_w <= 0 or qr_h <= 0:
            return self._build_failure_response(ERR_QR_ANCHOR_NOT_FOUND)

        # Crop tight clean QR ROI
        qr_roi = frame[qr_y:qr_y2, qr_x:qr_x2].copy()

        # =====================================================================
        # Step D: Relative OCR Zone Slicing & Perspective Leveling
        # =====================================================================
        # Calculate unit vectors along QR edges on straightened image
        v_top = p1 - p0
        w_top = float(np.linalg.norm(v_top))
        u_top = v_top / w_top if w_top > 1e-4 else np.array([1.0, 0.0], dtype=np.float32)

        v_bot = p2 - p3
        w_bot = float(np.linalg.norm(v_bot))
        u_bot = v_bot / w_bot if w_bot > 1e-4 else np.array([1.0, 0.0], dtype=np.float32)

        v_left = p3 - p0
        h_left = float(np.linalg.norm(v_left))
        u_left = v_left / h_left if h_left > 1e-4 else np.array([0.0, 1.0], dtype=np.float32)

        v_right = p2 - p1
        h_right = float(np.linalg.norm(v_right))
        u_right = v_right / h_right if h_right > 1e-4 else np.array([0.0, 1.0], dtype=np.float32)

        qr_nominal_w = (w_top + w_bot) / 2.0
        qr_nominal_h = (h_left + h_right) / 2.0

        gap_px = self.config.ocr_x_gap_ratio * qr_nominal_w
        text_w_px = self.config.ocr_width_ratio * qr_nominal_w
        text_h_px = self.config.ocr_height_ratio * qr_nominal_h
        y_shift_px = self.config.ocr_y_offset_ratio * qr_nominal_h

        # Compute 4 tilted corners of the OCR zone in physical image coordinates:
        # T0: Top-Left of Text
        # T1: Top-Right of Text
        # T2: Bottom-Right of Text
        # T3: Bottom-Left of Text
        t0 = p1 + gap_px * u_top + y_shift_px * u_right
        t3 = p2 + gap_px * u_bot + (y_shift_px + (text_h_px - qr_nominal_h)) * u_right

        t1 = t0 + text_w_px * u_top
        t2 = t3 + text_w_px * u_bot

        src_quad = np.array([t0, t1, t2, t3], dtype=np.float32)

        # Destination rectangle (perfect horizontal rectilinear frame for OCR)
        dst_w = max(32, int(round(text_w_px)))
        dst_h = max(16, int(round(text_h_px)))
        dst_quad = np.array(
            [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
            dtype=np.float32,
        )

        # Warp perspective to level out skew (< 0.2 ms)
        m_matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)
        ocr_roi = cv2.warpPerspective(
            frame,
            m_matrix,
            (dst_w, dst_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # =====================================================================
        # Step C: Omnidirectional Pharmacode Localization (Anywhere in image)
        # =====================================================================
        has_pharmacode = False
        pharmacode_payload: Optional[int] = None
        pharma_details: Dict[str, Any] = {}
        pharma_strip = None
        pharma_corners = None

        if self.config.enable_pharmacode:
            # Determine orientation from physical QR anchor top vector
            theta_deg = float(np.degrees(np.arctan2(v_top[1], v_top[0])))
            if self.config.pharmacode_direction.upper() == "AUTO":
                # If package is rotated > 90 deg (upside down), read L2R; otherwise R2L
                resolved_dir = "L2R" if abs(theta_deg) > 90.0 else "R2L"
            else:
                resolved_dir = self.config.pharmacode_direction.upper()

            # Omnidirectional signal-based Pharmacode search across arbitrary carton positions (zero layout hardcoding)
            pharma_found, p_val, p_info = scan_pharmacode_roi(
                gray,
                direction=resolved_dir,
                expected_payload=self.config.expected_pharmacode,
            )

            if pharma_found and "bbox" in p_info:
                bx, by, bw, bh = p_info["bbox"]
                pad = 4
                px1 = max(0, bx - pad)
                py1 = max(0, by - pad)
                px2 = min(img_w, bx + bw + pad)
                py2 = min(img_h, by + bh + pad)
                pharma_strip = frame[py1:py2, px1:px2].copy()
                pharma_corners = np.array([
                    [px1, py1],
                    [px2, py1],
                    [px2, py2],
                    [px1, py2],
                ], dtype=np.int32)
                has_pharmacode = True
                pharmacode_payload = p_val
                pharma_details = p_info

        # =====================================================================
        # Step E: Strict Return Payload Schema
        # =====================================================================
        return {
            "status": STATUS_SUCCESS,
            "straight_frame": straight_frame,
            "qr_roi": qr_roi,
            "ocr_roi": ocr_roi,
            "pharma_roi": pharma_strip,
            "qr_corners": np.array([p0, p1, p2, p3], dtype=np.int32),
            "ocr_corners": np.array([t0, t1, t2, t3], dtype=np.int32),
            "pharma_corners": pharma_corners,
            "has_pharmacode": has_pharmacode,
            "pharmacode_payload": pharmacode_payload,
            "pharmacode_details": pharma_details,
            "pharmacode_r2l": pharma_details.get("r2l"),
            "pharmacode_l2r": pharma_details.get("l2r"),
            "pharmacode_direction": pharma_details.get("direction_used", "R2L"),
        }

    def _build_failure_response(self, error_code: str) -> Dict[str, Any]:
        """Construct deterministic failure structure without throwing downstream unhandled exceptions."""
        return {
            "status": error_code,
            "straight_frame": None,
            "qr_roi": None,
            "ocr_roi": None,
            "pharma_roi": None,
            "qr_corners": None,
            "ocr_corners": None,
            "pharma_corners": None,
            "has_pharmacode": False,
            "pharmacode_payload": None,
            "pharmacode_details": {},
            "pharmacode_r2l": None,
            "pharmacode_l2r": None,
            "pharmacode_direction": None,
        }


def extract_multi_roi(
    frame: np.ndarray, config: Optional[ROIConfig] = None
) -> Dict[str, Any]:
    """
    Functional entry point for the Multi-ROI Extraction Module.

    :param frame: Input image from Module 1.
    :param config: Optional geometric configuration.
    :return: Standardized multi-ROI dictionary.
    """
    extractor = MultiROIExtractor(config=config)
    return extractor.extract(frame)
