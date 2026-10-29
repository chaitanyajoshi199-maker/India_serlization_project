"""
Real-Time Live Camera Feed Multi-Module Inspection Pipeline for Backend Integration.

Ingests frames sequentially from an active camera feed (Webcam, USB Camera, RTSP, or Video File),
passes each frame through the unified multi-engine pipeline:
1. Capture Module     : Optical Quality Validation (Exposure, Glare, Blur, Contrast)
2. Multi-ROI Module   : QR-Anchored Perspective Normalization & Multi-ROI Isolation
3. Specialized Sub-Engines:
   - Module 1: QR Verifier (ISO/IEC 15415 5-Parameter Print Quality Grading & Decoded Payload)
   - Module 2: OCR Engine (RapidOCR Multi-Line Text Extraction + Per-Line Confidence Scores, NO OCV)
   - Module 3: Pharmacode Module (Shun74 1D Scanline Profiler & Decoded Code)

Returns Structured Payload per Frame specifically formatted for Backend APIs:
{
    "qr": {
        "overall_grade": "A",
        "overall_grade_numeric": 4,
        "extracted_text": "LINK:0h2QAtdSGRoD",
        "iso_parameters": {
            "decode": {"grade": "A", "value": 1.0, ...},
            "symbol_contrast": {"grade": "A", "value": 0.88, ...},
            "modulation": {"grade": "B", "value": 0.62, ...},
            "axial_non_uniformity": {"grade": "A", "value": 0.02, ...},
            "unused_error_correction": {"grade": "A", "value": 1.0, ...}
        }
    },
    "pharmacode": {
        "extracted_text": "12345"
    },
    "ocr": {
        "ocr_extracted_text": "EXP: 07-2032 | LIC : PL-19635 | BATCH: B-015767 | SR:614344303780",
        "lines": [
            {"text": "EXP: 07-2032", "confidence": 0.985},
            {"text": "LIC : PL-19635", "confidence": 0.962},
            {"text": "BATCH: B-015767", "confidence": 0.991},
            {"text": "SR:614344303780", "confidence": 0.943}
        ]
    },
    "raw_frame": np.ndarray,
    "annotated_frame": np.ndarray
}

Usage:
    # 1. Run as standalone live camera application:
    python live_camera_feed_pipeline.py                     # Default webcam (index 0)
    python live_camera_feed_pipeline.py --cam 1             # External USB camera
    python live_camera_feed_pipeline.py --flip              # Mirror mode for front-facing webcam
    python live_camera_feed_pipeline.py --video path.mp4    # Process video file

    # 2. Import into your backend service:
    from live_camera_feed_pipeline import LiveCameraPipeline
    pipeline = LiveCameraPipeline(camera_source=0)
    for result in pipeline.stream():
        print(result["qr"]["overall_grade"], result["ocr"]["ocr_extracted_text"])
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import cv2
import numpy as np

# Ensure project root in sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from capture.datalogic_image_source import DatalogicImageSource

def load_datalogic_config() -> Dict[str, Any]:
    """Load portable Datalogic image-source configuration."""
    config_path = ROOT / "config" / "datalogic.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Datalogic configuration not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config

from integration_pipeline import (
    EnzeneIntegratedPipeline,
    IntegratedInspectionResult,
)
from capture.camera_gateway import GatewayConfig
from qr_verifier import ISOGrade


def draw_label_pill(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    text_color: Tuple[int, int, int],
    bg_color: Tuple[int, int, int] = (16, 16, 16),
    font_scale: float = 0.44,
    font_face: int = cv2.FONT_HERSHEY_DUPLEX,
    thickness: int = 1,
) -> None:
    """Draws crisp text with a solid dark rectangular backdrop pill for maximum visibility."""
    (tw, th), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    x, y = org
    img_h, img_w = img.shape[:2]
    x1 = max(0, x - 3)
    y1 = max(0, y - th - 4)
    x2 = min(img_w - 1, x + tw + 5)
    y2 = min(img_h - 1, y + baseline + 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), text_color, 1)
    cv2.putText(img, text, (x, y), font_face, font_scale, text_color, thickness, cv2.LINE_AA)


def annotate_inspection_frame(
    raw_frame: np.ndarray,
    result: IntegratedInspectionResult,
    backend_dict: Dict[str, Any],
    fps: float = 0.0,
) -> np.ndarray:
    """
    Renders high-visibility industrial visual overlays:
    - Bounding boxes around QR, OCR zone, and Pharmacode
    - Pure extracted OCR text lines with per-line confidence values drawn directly on detection
    - Top telemetry HUD banner with latency breakdown & overall status
    """
    vis_base = result.straight_frame if (result.straight_frame is not None and result.straight_frame.size > 0) else raw_frame
    vis = vis_base.copy()
    img_h, img_w = vis.shape[:2]

    # 1. Top Industrial Banner
    is_narrow = (img_w < 820)
    banner_h = 145 if is_narrow else 125
    banner = np.full((banner_h, img_w, 3), 20, dtype=np.uint8)

    # 2. Draw Bounding Polygons & Badges
    # A. QR Code Bounding Polygon (Green = Pass, Amber = Reject)
    if result.qr_corners is not None and len(result.qr_corners) == 4:
        pts_qr = result.qr_corners.reshape((-1, 1, 2))
        qr_color = (0, 255, 120) if result.qr.ok else (0, 180, 255)
        cv2.polylines(vis, [pts_qr], isClosed=True, color=qr_color, thickness=2, lineType=cv2.LINE_AA)
        top_pt = (int(result.qr_corners[0][0]), max(18, int(result.qr_corners[0][1]) - 6))
        qr_badge = f"QR: Grade {result.qr.overall_letter} ({result.qr.overall_grade:.1f}/4.0)"
        draw_label_pill(vis, qr_badge, top_pt, qr_color, font_scale=0.46)

    # B. OCR Text Zone Polygon & Extracted Text with Confidence
    ocr_lines = backend_dict.get("ocr", {}).get("lines", [])
    if result.ocr_corners is not None and len(result.ocr_corners) == 4:
        pts_ocr = result.ocr_corners.reshape((-1, 1, 2))
        ocr_color = (40, 215, 255) if ocr_lines else (80, 120, 180)
        cv2.polylines(vis, [pts_ocr], isClosed=True, color=ocr_color, thickness=2, lineType=cv2.LINE_AA)

        ocr_pt = (int(result.ocr_corners[0][0]), max(18, int(result.ocr_corners[0][1]) - 6))
        ocr_badge = f"OCR: {len(ocr_lines)} Lines Extracted"
        draw_label_pill(vis, ocr_badge, ocr_pt, ocr_color, font_scale=0.46)

        # Draw each line of extracted text directly below the bounding box
        if ocr_lines:
            ox_min = max(4, int(np.min(result.ocr_corners[:, 0])))
            oy_max = int(np.max(result.ocr_corners[:, 1]))
            oy_min = int(np.min(result.ocr_corners[:, 1]))

            y_cursor = oy_max + 18
            if y_cursor + len(ocr_lines) * 19 > img_h - 6:
                y_cursor = max(22, oy_min + 18)

            for item in ocr_lines:
                t_str = item.get("text", "")
                c_val = item.get("confidence", 0.0)
                display_line = f"{t_str} (conf: {c_val:.2f})"
                if y_cursor < img_h - 4:
                    draw_label_pill(
                        vis,
                        display_line,
                        (ox_min + 2, y_cursor),
                        text_color=(255, 255, 255),
                        bg_color=(15, 15, 15),
                        font_scale=0.43,
                        font_face=cv2.FONT_HERSHEY_SIMPLEX,
                    )
                    y_cursor += 19

    # C. Pharmacode Bounding Polygon (Sky Blue)
    pharma_text = backend_dict.get("pharmacode", {}).get("extracted_text")
    if result.pharma_corners is not None and len(result.pharma_corners) == 4:
        pts_pharma = result.pharma_corners.reshape((-1, 1, 2))
        pharma_color = (255, 190, 0) if pharma_text else (180, 140, 50)  # Sky Blue in BGR
        cv2.polylines(vis, [pts_pharma], isClosed=True, color=pharma_color, thickness=2, lineType=cv2.LINE_AA)

        p_y = int(result.pharma_corners[0][1]) - 6
        if p_y < 18:
            p_y = int(np.max(result.pharma_corners[:, 1])) + 16
        p_pt = (int(result.pharma_corners[0][0]), max(18, p_y))
        p_badge = f"PHARMA: Code {pharma_text} ({result.pharma.bar_count}b)" if pharma_text else "PHARMA: None"
        draw_label_pill(vis, p_badge, p_pt, pharma_color, font_scale=0.46)

    # D. 1D Retail Barcode Bounding Polygon (Purple / Pink - strictly distinct from Pharma)
    bc_dict = backend_dict.get("barcode", {})
    bc_text = bc_dict.get("extracted_text")
    bc_fmt = bc_dict.get("format", "1D")
    bc_origin = bc_dict.get("gs1_origin", "")
    origin_str = f" [{bc_origin}]" if bc_origin and bc_origin != "Unknown" else ""
    
    pts_bc = None
    if result.barcode_corners is not None and len(result.barcode_corners) >= 4:
        pts_bc = result.barcode_corners.reshape((-1, 1, 2))
    elif result.barcode.details.get("bbox"):
        bx, by, bw, bh = result.barcode.details["bbox"]
        pts_bc = np.array([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]], dtype=np.int32).reshape((-1, 1, 2))

    if pts_bc is not None:
        bc_color = (255, 80, 200) if bc_text else (160, 60, 140)
        cv2.polylines(vis, [pts_bc], isClosed=True, color=bc_color, thickness=2, lineType=cv2.LINE_AA)

        top_corner = pts_bc[0][0]
        bc_y = int(top_corner[1]) - 6
        if bc_y < 18:
            bc_y = int(np.max(pts_bc[:, 0, 1])) + 16
        bc_pt = (int(top_corner[0]), max(18, bc_y))
        bc_badge = f"{bc_fmt}: {bc_text}{origin_str}" if bc_text else "BARCODE: None"
        draw_label_pill(vis, bc_badge, bc_pt, bc_color, font_scale=0.46)

    # 3. Top Banner Information Lines
    lats = result.latencies
    overall_status = "PASS" if result.is_ok else "SEARCHING / REJECT"
    overall_color = (0, 230, 120) if result.is_ok else (40, 160, 240)

    # Line 1: Status & Timing
    title_left = f"STATUS: [{overall_status}]  |  TOTAL: {lats.total_ms:5.1f} ms  |  FPS: {fps:4.1f}"
    timing_str = f"[ROI:{lats.roi_extraction_ms:2.0f}ms | QR:{lats.qr_ms:2.0f}ms | OCR:{lats.ocr_ms:2.0f}ms | Pharma:{lats.pharma_ms:2.0f}ms | Barcode:{lats.barcode_ms:2.0f}ms]"
    cv2.putText(banner, title_left, (12, 22), cv2.FONT_HERSHEY_DUPLEX, 0.48, overall_color, 1, cv2.LINE_AA)
    if not is_narrow:
        cv2.putText(banner, timing_str, (max(12, img_w - 530), 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1, cv2.LINE_AA)

    # Line 2: QR Details
    qr_payload = backend_dict.get("qr", {}).get("extracted_text")
    qr_txt = f"QR: Grade {result.qr.overall_letter} ({result.qr.overall_grade:.1f}/4.0) | Payload: {str(qr_payload)[:40]}"
    cv2.putText(banner, qr_txt, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    # Line 3: OCR Extracted Text (NO OCV)
    ocr_raw_text = backend_dict.get("ocr", {}).get("ocr_extracted_text", "None")
    ocr_txt = f"OCR: ({len(ocr_lines)} Lines) Text: {str(ocr_raw_text)[:65]}"
    cv2.putText(banner, ocr_txt, (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 215, 255), 1, cv2.LINE_AA)

    # Line 4: Pharmacode Code (Sky Blue)
    pharma_code = backend_dict.get("pharmacode", {}).get("extracted_text")
    pharma_txt = f"PHARMACODE: Code {pharma_code or 'None'} ({result.pharma.bar_count} bars, {result.pharma.direction_used})"
    cv2.putText(banner, pharma_txt, (12, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 190, 0), 1, cv2.LINE_AA)

    # Line 5: 1D Retail Barcode
    bc_txt = f"1D BARCODE: {bc_fmt} {bc_text or 'None'}{origin_str} | Checksum: {'PASS' if bc_dict.get('checksum_valid') else 'N/A'}"
    cv2.putText(banner, bc_txt, (12, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 120, 220), 1, cv2.LINE_AA)

    # Accent bottom line
    cv2.line(banner, (0, banner_h - 2), (img_w, banner_h - 2), overall_color, 2)
    return np.vstack([banner, vis])


class OutputRecorder:
    """
    Manages continuous persistence of inspection outcomes with a rolling FIFO window:
    - results.json   : retains only the last `max_history` (default: 50) frame telemetry entries
    - results_images/ :
        - raw_nok_images/   : retains only the last `max_history` (default: 50) raw uncompressed NOK images
        - annotated_images/ : retains only the last `max_history` (default: 50) compressed JPEG images
    Automatically purges older images and JSON records from disk when new ones arrive, preventing storage overflow.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = ROOT / "output",
        max_history: int = 50,
        save_images: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.max_history = max(1, int(max_history))
        self.save_images = save_images
        self.images_dir = self.output_dir / "results_images"
        self.raw_nok_dir = self.images_dir / "raw_nok_images"
        self.annotated_dir = self.images_dir / "annotated_images"
        self.json_path = self.output_dir / "results.json"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_images:
            self.images_dir.mkdir(parents=True, exist_ok=True)
            self.raw_nok_dir.mkdir(parents=True, exist_ok=True)
            self.annotated_dir.mkdir(parents=True, exist_ok=True)
            self._prune_existing_files(self.annotated_dir)
            self._prune_existing_files(self.raw_nok_dir)
            self.saved_annotated_files = self._get_existing_files(self.annotated_dir)
            self.saved_raw_nok_files = self._get_existing_files(self.raw_nok_dir)
        else:
            self.saved_annotated_files = []
            self.saved_raw_nok_files = []

        # Load existing JSON records and prune to max_history
        self.records: List[Dict[str, Any]] = []
        if self.json_path.exists():
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.records = data[-self.max_history:]
            except Exception:
                self.records = []

        self._write_json()
        self._last_flush = time.perf_counter()

    def _get_existing_files(self, folder: Path) -> List[Path]:
        try:
            return sorted([p for p in folder.glob("*") if p.is_file()], key=lambda p: p.stat().st_mtime)
        except Exception:
            return []

    def _prune_existing_files(self, folder: Path) -> None:
        """Purges oldest files in directory if total count exceeds max_history."""
        try:
            files = sorted([p for p in folder.glob("*") if p.is_file()], key=lambda p: p.stat().st_mtime)
            if len(files) > self.max_history:
                excess = len(files) - self.max_history
                for f in files[:excess]:
                    try:
                        f.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception:
            pass

    def record_frame(
        self,
        frame_idx: int,
        backend_dict: Dict[str, Any],
        raw_frame: Optional[np.ndarray] = None,
        annotated_vis: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Logs frame telemetry to results.json and optionally saves images if save_images is enabled.
        """
        timestamp_str = datetime.now().isoformat()
        annotated_filename = f"frame_{frame_idx:06d}_annotated.jpg"
        annotated_file_path = self.annotated_dir / annotated_filename

        ann_rel_path: Optional[str] = None
        raw_rel_path: Optional[str] = None

        # Only save image files to disk if save_images is explicitly enabled
        if self.save_images:
            compressed_bytes = backend_dict.get("annotated_frame")
            saved_ann = False
            try:
                if isinstance(compressed_bytes, (bytes, bytearray)):
                    with open(annotated_file_path, "wb") as f:
                        f.write(compressed_bytes)
                    saved_ann = True
                elif annotated_vis is not None and annotated_vis.size > 0:
                    cv2.imwrite(
                        str(annotated_file_path),
                        annotated_vis,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    )
                    saved_ann = True
            except OSError:
                pass

            if saved_ann:
                ann_rel_path = f"results_images/annotated_images/{annotated_filename}"
                self.saved_annotated_files.append(annotated_file_path)
                while len(self.saved_annotated_files) > self.max_history:
                    old_f = self.saved_annotated_files.pop(0)
                    try:
                        old_f.unlink(missing_ok=True)
                    except Exception:
                        pass

            # Save raw uncompressed image into raw_nok_images/ ONLY if frame is NOK
            is_nok = (backend_dict.get("status") == "NOK") or (backend_dict.get("raw_frame") is not None)
            if is_nok:
                raw_to_save = backend_dict.get("raw_frame")
                if raw_to_save is None:
                    raw_to_save = raw_frame
                if raw_to_save is not None and isinstance(raw_to_save, np.ndarray) and raw_to_save.size > 0:
                    raw_filename = f"frame_{frame_idx:06d}_raw_nok.png"
                    raw_file_path = self.raw_nok_dir / raw_filename
                    try:
                        cv2.imwrite(str(raw_file_path), raw_to_save)
                        raw_rel_path = f"results_images/raw_nok_images/{raw_filename}"
                        self.saved_raw_nok_files.append(raw_file_path)
                        while len(self.saved_raw_nok_files) > self.max_history:
                            old_f = self.saved_raw_nok_files.pop(0)
                            try:
                                old_f.unlink(missing_ok=True)
                            except Exception:
                                pass
                    except OSError:
                        pass

        # Create JSON entry for results.json
        entry = {
            "frame_index": frame_idx,
            "timestamp": timestamp_str,
            "status": backend_dict.get("status", "NOK"),
            "overall_status": backend_dict.get("overall_status", "NOK"),
            "qr": backend_dict.get("qr", {}),
            "pharmacode": backend_dict.get("pharmacode", {}),
            "barcode": backend_dict.get("barcode", {}),
            "ocr": backend_dict.get("ocr", {}),
            "annotated_image": ann_rel_path,
            "raw_nok_image": raw_rel_path,
        }

        self.records.append(entry)
        if len(self.records) > self.max_history:
            self.records = self.records[-self.max_history:]

        # Periodic flush (every 0.5s or every 5 frames) to balance real-time sync with high FPS
        now = time.perf_counter()
        if (now - self._last_flush >= 0.5) or (len(self.records) % 5 == 0):
            self.flush()

        return entry

    def flush(self) -> None:
        """Persists rolling inspection records to results.json."""
        self._last_flush = time.perf_counter()
        self._write_json()

    def _write_json(self) -> None:
        try:
            tmp_path = self.json_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.records[-self.max_history:], f, indent=2)
            tmp_path.replace(self.json_path)
        except Exception as e:
            try:
                with open(self.json_path, "w", encoding="utf-8") as f:
                    json.dump(self.records[-self.max_history:], f, indent=2)
            except Exception as err:
                print(f"[WARN] Failed to write results.json: {err}")


class LiveCameraPipeline:
    """
    Live Camera Feed Pipeline Manager.
    
    Feeds frames continuously from the camera into the multi-engine inspection pipeline,
    yields clean structured dictionaries for backend consumption, and automatically
    archives results to results.json and result_images/.
    """

    def __init__(
        self,
        camera_source: Union[int, str] = 0,
        width: Optional[int] = 1280,
        height: Optional[int] = 720,
        mirror: bool = False,
        qr_pass_threshold: ISOGrade = ISOGrade.C,
        optical_checks: bool = False,
        ocr_interval_sec: float = 0.25,
        save_results: bool = True,
        save_images: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        max_history: int = 50,
    ) -> None:
        self.camera_source = camera_source
        self.width = width
        self.height = height
        self.mirror = mirror
        self.optical_checks = optical_checks
        self.ocr_interval_sec = ocr_interval_sec
        self.save_results = save_results
        self.save_images = save_images
        self.output_dir = Path(output_dir) if output_dir else (ROOT / "output")
        self.max_history = max_history

        # Persistent Output Recorder (results.json & optional result_images/ rolling buffer)
        self.recorder = (
            OutputRecorder(
                output_dir=self.output_dir,
                max_history=self.max_history,
                save_images=self.save_images,
            )
            if self.save_results
            else None
        )

        # Initialize Camera
        self.cap = self._init_camera(camera_source, width, height)

        # Initialize Enzene Integration Pipeline (pure OCR, NO OCV target matching)
        gw_cfg = GatewayConfig(
            target_resolution=None,
            blur_threshold=15.0,
            max_glare_pixel_ratio=0.10,
            min_brightness=15.0,
            max_brightness=250.0,
        )
        self.pipeline = EnzeneIntegratedPipeline(
            gateway_config=gw_cfg,
            qr_pass_threshold=qr_pass_threshold,
            enable_qr=True,
            enable_ocr=True,
            enable_pharma=True,
            target_ocr_data=None,           # STRICTLY NO OCV MASTER TARGET (Pure OCR only)
            fallback_pharma_full_frame=True,
        )

        self._last_ocr_time = 0.0
        self._last_ocr_res = None
        self._frame_count = 0

    def _init_camera(
        self,
        source: Union[int, str],
        width: Optional[int],
        height: Optional[int],
    ) -> Any:
        """Initialize either a live Datalogic image source or an OpenCV source."""

        # ---------------------------------------------------------
        # DATALOGIC MATRIX 220 LIVE IMAGE MODE
        # ---------------------------------------------------------
        if isinstance(source, str) and source.lower() == "datalogic":
            try:
                cfg = load_datalogic_config()
            except Exception as exc:
                print(f"[DATALOGIC] Configuration error: {exc}")
                sys.exit(1)

            if not cfg.get("enabled", False):
                print("[DATALOGIC] Datalogic mode is disabled in config/datalogic.json")
                sys.exit(1)

            matrix_ip = cfg["matrix_ip"]
            image_cfg = cfg["image"]
            connection_cfg = cfg.get("connection", {})

            datalogic_source = DatalogicImageSource(
                matrix_ip=matrix_ip,
                image_port=image_cfg["port"],
                image_format=image_cfg.get("format", "bmp"),
                header_hex=image_cfg.get("header_hex", "02"),
                terminator_hex=image_cfg.get("terminator_hex", "0D0A"),
                reconnect_delay_seconds=connection_cfg.get(
                    "reconnect_delay_seconds",
                    1.0,
                ),
                socket_timeout_seconds=connection_cfg.get(
                    "socket_timeout_seconds",
                    2.0,
                ),
                save_to_disk=self.save_images,
            )

            if not datalogic_source.open():
                print(
                    "[ERROR] Could not connect to "
                    "Datalogic Matrix 220 image channel."
                )
                sys.exit(1)

            print("[DATALOGIC] Live Matrix image source enabled.")
            print(
                f"[DATALOGIC] Matrix : "
                f"{matrix_ip}:{image_cfg['port']}"
            )

            return datalogic_source

        # ---------------------------------------------------------
        # NORMAL OPENCV MODE
        # ---------------------------------------------------------
        if isinstance(source, int) or (
            isinstance(source, str) and source.isdigit()
        ):
            idx = int(source)
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"[ERROR] Failed to open camera source: {source}")
            sys.exit(1)

        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return cap

    def inspect_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Executes full inspection on a single frame and returns the exact backend payload dictionary.
        """
        self._frame_count += 1
        now = time.perf_counter()

        # Check if OCR should run on this frame
        should_run_ocr = (now - self._last_ocr_time) >= self.ocr_interval_sec

        # Execute Multi-Module Inspection
        res: IntegratedInspectionResult = self.pipeline.process_frame(
            frame=frame,
            run_optical_checks=self.optical_checks,
            force_ocr=should_run_ocr,
        )

        if should_run_ocr and res.ocr.raw_lines:
            self._last_ocr_time = now
            self._last_ocr_res = res.ocr
        elif not should_run_ocr and self._last_ocr_res is not None:
            # Preserve cached OCR text between intervals to maintain high camera FPS
            res.ocr = self._last_ocr_res

        # Build Clean Backend Payload
        backend_dict = res.to_backend_result(raw_frame=frame)

        # Generate Annotated Image
        annotated_vis = annotate_inspection_frame(frame, res, backend_dict)

        # Finalize backend payload:
        # - annotated_frame: compressed JPEG bytes for each OK/NOK frame (for frontend)
        # - raw_frame: uncompressed raw ndarray ONLY for NOK frames (for db storage)
        backend_dict = res.to_backend_result(
            raw_frame=frame,
            annotated_frame=annotated_vis,
        )
        # Keep uncompressed array for local OpenCV display window
        backend_dict["_annotated_display"] = annotated_vis

        # Auto-save results to output/results.json and output/result_images/
        if self.recorder is not None:
            self.recorder.record_frame(
                frame_idx=self._frame_count,
                backend_dict=backend_dict,
                raw_frame=frame,
                annotated_vis=annotated_vis,
            )

        return backend_dict

    def stream(self) -> Generator[Dict[str, Any], None, None]:
        """
        Generator yielding inspection results frame-by-frame for streaming services / backend workers.
        """
        while self.cap.isOpened():
            success, raw_frame = self.cap.read()
            if not success or raw_frame is None or raw_frame.size == 0:
                time.sleep(0.01)
                continue

            if self.mirror:
                raw_frame = cv2.flip(raw_frame, 1)

            result_dict = self.inspect_frame(raw_frame)
            yield result_dict

    def release(self) -> None:
        """Closes camera device cleanly and flushes saved results."""
        if self.recorder is not None:
            self.recorder.flush()
        if self.cap and self.cap.isOpened():
            self.cap.release()


def run_interactive_camera(
    source: Union[int, str] = 0,
    width: int = 1280,
    height: int = 720,
    mirror: bool = False,
    optical_checks: bool = False,
    ocr_interval: float = 0.25,
    save_results: bool = True,
    save_images: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    max_history: int = 50,
) -> None:
    """
    Runs the live camera pipeline, displays real-time OpenCV window, prints backend telemetry,
    and automatically logs results to results.json and optionally result_images/.
    """
    out_dir_path = Path(output_dir) if output_dir else (ROOT / "output")

    print("\n" + "=" * 85)
    print(" ENZENE MULTI-MODULE LIVE CAMERA FEED INSPECTION PIPELINE")
    print("=" * 85)
    print(f" Camera Source     : {source}")
    print(f" Target Resolution : {width} x {height}")
    print(f" Mirror Mode       : {'ENABLED' if mirror else 'DISABLED'}")
    print(f" Mode              : PURE OCR (NO OCV TARGET MATCHING)")
    print(f" OCR Run Interval  : Every {ocr_interval:.2f}s")
    if save_results:
        print(f" Results Output    : {out_dir_path.resolve()}")
        print(f"   * Buffer History: Last {max_history} frames (Rolling ring buffer)")
        print(f"   * JSON Log      : {out_dir_path.resolve() / 'results.json'}")
        if save_images:
            print(f"   * Results Images: {out_dir_path.resolve() / 'results_images'} (ENABLED)")
            print(f"     ├── raw_nok_images/   (max {max_history} uncompressed images, NOK only)")
            print(f"     └── annotated_images/ (max {max_history} compressed JPEGs)")
        else:
            print(f"   * Results Images: DISABLED (Zero Disk Space Usage - Memory only)")
    else:
        print(f" Results Output    : DISABLED")
    print(" Hotkeys:")
    print("   'q' / ESC : Quit")
    print("   's'       : Save manual snapshot (camera_snapshots/)")
    print("   'f'       : Toggle Mirror (Flip)")
    print("   'p'       : Print current backend JSON in terminal")
    print("=" * 85 + "\n")

    pipeline = LiveCameraPipeline(
        camera_source=source,
        width=width,
        height=height,
        mirror=mirror,
        optical_checks=optical_checks,
        ocr_interval_sec=ocr_interval,
        save_results=save_results,
        save_images=save_images,
        output_dir=out_dir_path,
        max_history=max_history,
    )

    win_name = "Enzene Live Camera Multi-Module Inspector"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1100, 700)

    fps_history: List[float] = []
    t_prev = time.perf_counter()
    save_counter = 0

    try:
        while pipeline.cap.isOpened():
            t_start = time.perf_counter()
            success, raw_frame = pipeline.cap.read()

            if not success or raw_frame is None or raw_frame.size == 0:
                print("[WARNING] Frame drop from camera. Re-polling...")
                time.sleep(0.01)
                continue

            if pipeline.mirror:
                raw_frame = cv2.flip(raw_frame, 1)

            # Process frame (automatically records to results.json & results_images/)
            result = pipeline.inspect_frame(raw_frame)

            # Compute smoothed FPS
            dt = time.perf_counter() - t_start
            fps_curr = 1.0 / max(dt, 1e-4)
            fps_history.append(fps_curr)
            if len(fps_history) > 15:
                fps_history.pop(0)
            avg_fps = float(np.mean(fps_history))

            # Display on screen using uncompressed visual frame
            annotated_display = result.get("_annotated_display")
            if annotated_display is None:
                if isinstance(result["annotated_frame"], bytes):
                    annotated_display = cv2.imdecode(
                        np.frombuffer(result["annotated_frame"], np.uint8),
                        cv2.IMREAD_COLOR
                    )
                else:
                    annotated_display = result["annotated_frame"]

            cv2.imshow(win_name, annotated_display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                print("[INFO] Quitting live camera inspection.")
                break
            elif key in (ord('f'), ord('F')):
                pipeline.mirror = not pipeline.mirror
                print(f"[INFO] Mirror Mode: {'ENABLED' if pipeline.mirror else 'DISABLED'}")
            elif key in (ord('p'), ord('P')):
                # Print backend JSON to terminal
                json_payload = {
                    "status": result.get("status"),
                    "qr": result["qr"],
                    "pharmacode": result["pharmacode"],
                    "ocr": result["ocr"],
                    "raw_frame_stored": result["raw_frame"] is not None,
                    "annotated_frame_bytes": len(result["annotated_frame"]) if isinstance(result["annotated_frame"], bytes) else 0,
                }
                print("\n" + "=" * 75)
                print("CURRENT FRAME BACKEND PAYLOAD:")
                print(json.dumps(json_payload, indent=2))
                print("=" * 75 + "\n")
            elif key in (ord('s'), ord('S')):
                save_counter += 1
                out_dir = ROOT / "camera_snapshots"
                raw_nok_dir = out_dir / "raw_nok_images"
                ann_dir = out_dir / "annotated_images"
                raw_nok_dir.mkdir(parents=True, exist_ok=True)
                ann_dir.mkdir(parents=True, exist_ok=True)
                
                # Save Raw Image (Only present if frame is NOK)
                raw_saved_msg = "None (Omitted for OK frame)"
                if result["raw_frame"] is not None:
                    raw_file = raw_nok_dir / f"frame_{save_counter:04d}_raw_nok.png"
                    cv2.imwrite(str(raw_file), result["raw_frame"])
                    raw_saved_msg = str(raw_file.resolve())
                
                # Save Annotated Image (Compressed JPEG)
                annotated_file = ann_dir / f"frame_{save_counter:04d}_annotated.jpg"
                if isinstance(result["annotated_frame"], bytes):
                    with open(annotated_file, "wb") as f:
                        f.write(result["annotated_frame"])
                elif result["annotated_frame"] is not None:
                    cv2.imwrite(str(annotated_file), result["annotated_frame"])
                
                # Save JSON Telemetry
                json_file = out_dir / f"frame_{save_counter:04d}_backend.json"
                json_data = {
                    "status": result.get("status"),
                    "qr": result["qr"],
                    "pharmacode": result["pharmacode"],
                    "ocr": result["ocr"],
                    "raw_frame_stored": result["raw_frame"] is not None,
                    "annotated_frame_bytes": len(result["annotated_frame"]) if isinstance(result["annotated_frame"], bytes) else 0,
                }
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2)

                print(f"\n[MANUAL SNAPSHOT SAVED #{save_counter:04d}]")
                print(f"  * Status          : {result.get('status')}")
                print(f"  * Raw Frame       : {raw_saved_msg}")
                print(f"  * Annotated Frame : {annotated_file.resolve()}")
                print(f"  * Backend JSON    : {json_file.resolve()}\n")

    finally:
        pipeline.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Live camera feed multi-module pharmaceutical carton inspector.")
    parser.add_argument(
    "--cam",
    default=None,
    help="Camera source. Leave empty to use Datalogic configuration."
)
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height (default: 720)")
    parser.add_argument("--flip", action="store_true", help="Mirror horizontally (recommended for front-facing webcams)")
    parser.add_argument("--optical", action="store_true", help="Enable optical quality checks (blur, exposure, glare)")
    parser.add_argument("--ocr-interval", type=float, default=0.25, help="OCR throttle interval in seconds (default: 0.25s)")
    parser.add_argument("--output-dir", default=str(ROOT / "output"), help="Output directory to save results.json and result_images/ (default: output/)")
    parser.add_argument("--max-history", type=int, default=50, help="Maximum number of rolling images and JSON entries to retain (default: 50)")
    parser.add_argument("--save-images", action="store_true", default=False, help="Enable saving annotated and raw NOK images to disk (default: disabled to preserve disk space)")
    parser.add_argument("--no-save", action="store_true", help="Disable automatic saving of results to output directory")
    args = parser.parse_args()

    datalogic_cfg = load_datalogic_config()

    if args.cam is None:
        src = "datalogic" if datalogic_cfg.get("enabled", False) else 0
    else:
        if str(args.cam).lower() == "datalogic":
            src = "datalogic"
        elif str(args.cam).isdigit():
            src = int(args.cam)
        else:
            src = args.cam

    run_interactive_camera(
        source=src,
        width=args.width,
        height=args.height,
        mirror=args.flip,
        optical_checks=args.optical,
        ocr_interval=args.ocr_interval,
        save_results=not args.no_save,
        save_images=args.save_images,
        output_dir=args.output_dir,
        max_history=args.max_history,
    )


if __name__ == "__main__":
    main()
