"""
End-to-End Industrial Inspection Integration Module.

Integrates 3 sequential processing stages into a single unified pipeline:
1. Capture Module     : Camera stream ingestion & optical quality gatekeeper (exposure, blur, glare).
2. Multi-ROI Module   : Physical QR-anchored perspective leveling & ROI isolation (QR, OCR, Pharmacode).
3. Specialized Sub-Engines:
   - Module 1: QR Verifier (ISO/IEC 15415 grading, payload decoding, motion deblurring).
   - Module 2: OCR/OCV Engine (RapidOCR text parsing, invariant length checks for GTIN, Lot, Exp, Serial).
   - Module 3: Pharmacode Module (shun74 1D Scanline Profiler, zero false-positive cluster decoding).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Capture Module
from capture.camera_gateway import (
    CameraStreamGateway,
    GatewayConfig,
    FrameErrorCode,
    FrameValidationResult,
)

# ROI Module
from roi_module.extractor import (
    MultiROIExtractor,
    ROIConfig,
    STATUS_SUCCESS as ROI_SUCCESS,
    ERR_QR_ANCHOR_NOT_FOUND,
)

# Module 1: QR Verifier
from qr_verifier import QRVerifier, ISOGrade
from qr_verifier.core import VerificationResult as QRVerificationResult

# Module 2: Pure OCR Engine
from OCR_OCV_engine.core_engine import IndustrialOCVEngine, extract_parameter_candidates

# Module 3: Pharmacode Detector
from pharma_detector.pharmacode import scan_pharmacode_roi

# Module 4: 1D Retail Barcode Engine (EAN-13, EAN-8, UPC-A, Code 128)
from barcode_detector import Industrial1DBarcodeDetector, BarcodeDetectionResult

logger = logging.getLogger("IntegrationPipeline")


@dataclass
class QRInspectionSummary:
    """Consolidated findings from the QR Verifier engine."""
    detected: bool = False
    status_str: str = "NOK"                # "OK" or "NOK"
    payload: Optional[str] = None
    reason: str = "NO_QR_ANCHOR"
    overall_letter: str = "F"             # "A", "B", "C", "D", "F"
    overall_grade: float = 0.0
    iso_parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    failing_metrics: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def status(self) -> str:
        return "PASS" if self.status_str == "OK" else "REJECT"

    @property
    def status(self) -> str:
        return "PASS" if self.status_str == "OK" else "FAIL"

    @property
    def status(self) -> str:
        return "PASS" if self.status_str == "OK" else "FAIL"

    @property
    def ok(self) -> bool:
        return self.status_str == "OK"


@dataclass
class OCRInspectionSummary:
    """Consolidated findings from the OCR/OCV engine."""
    status_str: str = "NOK"               # "OK" or "NOK"
    text_ext: str = ""                    # Extracted text block
    raw_lines: List[Dict[str, Any]] = field(default_factory=list)
    parsed_fields: Dict[str, str] = field(default_factory=dict)
    ocv_report: Optional[Dict[str, Any]] = None
    ocv_reason: str = "TARGET_NOT_CONFIGURED"
    reason: str = "NO_OCR_ROI"
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status_str == "OK"


@dataclass
class PharmacodeSummary:
    """Consolidated findings from the Shun74 Pharmacode engine."""
    found: bool = False
    status_str: str = "NOK"               # "OK" or "NOK"
    payload: Optional[int] = None
    reason: str = "NO_PHARMACODE_DETECTED"
    bar_count: int = 0
    bars: List[int] = field(default_factory=list)     # 1=Narrow, 2=Wide
    bar_pattern: str = ""                             # "NWNNW..."
    r2l: Optional[int] = None
    l2r: Optional[int] = None
    direction_used: str = "R2L"
    streak: int = 0
    confidence: int = 0
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status_str == "OK"


@dataclass
class BarcodeSummary:
    """Consolidated findings from the 1D Retail Barcode engine (EAN-13, EAN-8, UPC, Code 128)."""
    found: bool = False
    status_str: str = "NOK"                # "OK" or "NOK"
    payload: Optional[str] = None          # e.g. "8905722000777"
    format: Optional[str] = None           # e.g. "EAN13"
    checksum_valid: bool = False
    gs1_origin: Optional[str] = None       # e.g. "GS1 India"
    orientation: int = 0
    reason: str = "NO_BARCODE_DETECTED"
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status_str == "OK"


@dataclass
class PipelineLatencies:
    """Execution timing breakdown per subsystem (milliseconds)."""
    optical_check_ms: float = 0.0
    roi_extraction_ms: float = 0.0
    qr_ms: float = 0.0
    ocr_ms: float = 0.0
    pharma_ms: float = 0.0
    barcode_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class IntegratedInspectionResult:
    """Standardized top-level result container for every processed frame."""
    frame_index: int = 0
    timestamp: float = field(default_factory=time.time)
    
    # 1. Optical Check
    optical_valid: bool = False
    optical_error: Optional[str] = None
    optical_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # 2. ROI Segmentation
    roi_status: str = "NOT_EXTRACTED"
    qr_corners: Optional[np.ndarray] = None       # 4 points (x, y)
    ocr_corners: Optional[np.ndarray] = None      # 4 points (x, y)
    pharma_corners: Optional[np.ndarray] = None   # 4 points (x, y)
    barcode_corners: Optional[np.ndarray] = None  # 4 points (x, y)
    
    # Cropped ROIs & Straightened Frame for downstream / UI display
    straight_frame: Optional[np.ndarray] = None
    qr_roi: Optional[np.ndarray] = None
    ocr_roi: Optional[np.ndarray] = None
    pharma_roi: Optional[np.ndarray] = None
    barcode_roi: Optional[np.ndarray] = None
    
    # 3. Sub-Engine Outputs
    qr: QRInspectionSummary = field(default_factory=QRInspectionSummary)
    ocr: OCRInspectionSummary = field(default_factory=OCRInspectionSummary)
    pharma: PharmacodeSummary = field(default_factory=PharmacodeSummary)
    barcode: BarcodeSummary = field(default_factory=BarcodeSummary)
    
    # 4. Latency Telemetry
    latencies: PipelineLatencies = field(default_factory=PipelineLatencies)

    @property
    def has_any_detection(self) -> bool:
        """True if at least one sub-engine found a code or text."""
        return self.qr.detected or bool(self.ocr.raw_lines) or self.pharma.found or self.barcode.found

    @property
    def is_ok(self) -> bool:
        """
        True if all mandatory checks (QR, Pharmacode, Optical, OCV) passed.
        Returns False (NOK) if any required module fails.
        """
        if not (self.qr.ok and self.pharma.ok and self.optical_valid):
            return False
        if self.ocr.ocv_report is not None and self.ocr.ocv_report.get("overall_status") == "FAIL":
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire inspection telemetry into a database-ready JSON serializable dict."""
        iso_time = datetime.fromtimestamp(self.timestamp).isoformat()
        return {
            "frame_index": self.frame_index,
            "timestamp": iso_time,
            "overall_status": "PASS" if self.is_ok else "FAIL",
            "status": "OK" if self.is_ok else "NOK",
            "optical_quality": {
                "valid": self.optical_valid,
                "error": self.optical_error,
                "mean_brightness": self.optical_metrics.get("mean_brightness"),
                "glare_ratio": self.optical_metrics.get("glare_saturated_ratio"),
                "blur_laplacian": self.optical_metrics.get("blur_laplacian_var"),
            },
            "roi": {
                "status": self.roi_status,
                "has_qr_crop": self.qr_roi is not None,
                "has_ocr_crop": self.ocr_roi is not None,
                "has_pharma_crop": self.pharma_roi is not None,
                "has_barcode_crop": self.barcode_roi is not None,
            },
            "qr": {
                "status": self.qr.status_str,
                "payload": self.qr.payload,
                "overall_grade": f"Grade {self.qr.overall_letter} ({self.qr.overall_grade:.1f}/4.0)",
                "overall_letter": self.qr.overall_letter,
                "overall_numeric": self.qr.overall_grade,
                "iso_parameters": self.qr.iso_parameters,
                "failing_metrics": self.qr.failing_metrics,
                "reason": self.qr.reason,
                "latency_ms": round(self.qr.latency_ms, 2),
            },
            "ocr_ocv": {
                "status": self.ocr.status_str,
                "text_ext": self.ocr.text_ext,
                "raw_lines": self.ocr.raw_lines,
                "parsed_fields": self.ocr.parsed_fields,
                "ocv_reason": self.ocr.ocv_reason,
                "reason": self.ocr.reason,
                "latency_ms": round(self.ocr.latency_ms, 2),
            },
            "pharma": {
                "status": self.pharma.status_str,
                "payload": self.pharma.payload,
                "bar_count": self.pharma.bar_count,
                "bars": self.pharma.bars,
                "bar_pattern": self.pharma.bar_pattern,
                "direction_used": self.pharma.direction_used,
                "r2l": self.pharma.r2l,
                "l2r": self.pharma.l2r,
                "streak": self.pharma.streak,
                "reason": self.pharma.reason,
                "latency_ms": round(self.pharma.latency_ms, 2),
            },
            "barcode": {
                "status": self.barcode.status_str,
                "payload": self.barcode.payload,
                "format": self.barcode.format,
                "checksum_valid": self.barcode.checksum_valid,
                "gs1_origin": self.barcode.gs1_origin,
                "orientation": self.barcode.orientation,
                "reason": self.barcode.reason,
                "latency_ms": round(self.barcode.latency_ms, 2),
            },
            "latencies_ms": {
                "optical_check": round(self.latencies.optical_check_ms, 2),
                "roi_extraction": round(self.latencies.roi_extraction_ms, 2),
                "qr_verify": round(self.latencies.qr_ms, 2),
                "ocr_extract": round(self.latencies.ocr_ms, 2),
                "pharma_scan": round(self.latencies.pharma_ms, 2),
                "barcode_scan": round(self.latencies.barcode_ms, 2),
                "total": round(self.latencies.total_ms, 2),
            },
        }

    def to_backend_result(
        self,
        raw_frame: Optional[np.ndarray] = None,
        annotated_frame: Optional[Union[np.ndarray, bytes]] = None,
        compress_quality: int = 80,
    ) -> Dict[str, Any]:
        """
        Formats specifically requested backend payload:
        - status: "OK" | "NOK"
        - qr: overall_grade, extracted_text, each iso parameter grade
        - pharmacode: extracted_text
        - ocr: ocr_extracted_text, lines with confidence
        - raw_frame: raw uncompressed image (ONLY for NOK frames, None for OK frames)
        - annotated_frame: compressed JPEG bytes for each OK/NOK frame (for frontend display)
        """
        ocr_lines_with_conf = []
        for line in self.ocr.raw_lines:
            if isinstance(line, dict):
                ocr_lines_with_conf.append({
                    "text": line.get("text", ""),
                    "confidence": round(float(line.get("confidence", 0.0)), 4),
                })
            elif isinstance(line, (list, tuple)) and len(line) >= 3:
                ocr_lines_with_conf.append({
                    "text": str(line[1]),
                    "confidence": round(float(line[2]), 4),
                })
            else:
                ocr_lines_with_conf.append({
                    "text": str(line),
                    "confidence": 1.0,
                })

        # Evaluate OK vs NOK
        is_ok = self.is_ok
        status_str = "OK" if is_ok else "NOK"

        # 1. Annotated frame: Compressed JPEG bytes for both OK and NOK frames (for frontend streaming)
        compressed_annotated: Optional[bytes] = None
        if annotated_frame is not None:
            if isinstance(annotated_frame, (bytes, bytearray)):
                compressed_annotated = bytes(annotated_frame)
            elif isinstance(annotated_frame, np.ndarray) and annotated_frame.size > 0:
                success, enc = cv2.imencode(
                    ".jpg",
                    annotated_frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), compress_quality]
                )
                if success:
                    compressed_annotated = enc.tobytes()

        # 2. Raw frame: Uncompressed raw image ONLY for NOK frames to store in DB
        # For OK frames, raw_frame is None to save storage and network transfer
        raw_output = raw_frame if (not is_ok) else None

        return {
            "status": status_str,
            "overall_status": status_str,
            "qr": {
                "status": self.qr.status_str,
                "overall_grade": self.qr.overall_letter,
                "overall_grade_numeric": self.qr.overall_grade,
                "extracted_text": self.qr.payload,
                "iso_parameters": self.qr.iso_parameters,
            },
            "pharmacode": {
                "status": self.pharma.status_str,
                "extracted_text": str(self.pharma.payload) if self.pharma.payload is not None else None,
            },
            "barcode": {
                "status": self.barcode.status_str,
                "format": self.barcode.format,
                "extracted_text": self.barcode.payload,
                "checksum_valid": self.barcode.checksum_valid,
                "gs1_origin": self.barcode.gs1_origin,
            },
            "ocr": {
                "status": self.ocr.status_str,
                "ocr_extracted_text": self.ocr.text_ext,
                "lines": ocr_lines_with_conf,
            },
            "raw_frame": raw_output,
            "annotated_frame": compressed_annotated,
        }

    def to_terminal_report(self) -> str:
        """
        Formats a clean, comprehensive multi-line industrial report for terminal display.
        Covers:
        a) QR: ok/nok, payload if ok, reason, overall_grade, each iso param grade
        b) OCR_OCV: ok/nok, text_ext, ocv_reason, reason, parsed fields
        c) Pharma: ok/nok, payload, reason, bars, direction, streak
        d) Optical & Speeds
        """
        lines = []
        divider = "=" * 85
        subdiv = "-" * 85
        t_str = datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S.%f")[:-3]

        lines.append(divider)
        lines.append(f" [INSPECTION TELEMETRY] Frame #{self.frame_index:05d} | Time: {t_str} | Total: {self.latencies.total_ms:5.1f} ms")
        lines.append(divider)

        # -------------------------------------------------------------
        # A) QR MODULE
        # -------------------------------------------------------------
        qr_badge = f"[{self.qr.status_str}]"
        lines.append(f" a) QR CODE MODULE: {qr_badge}")
        if self.qr.ok:
            lines.append(f"    * Payload       : {self.qr.payload}")
        else:
            lines.append(f"    * Payload       : None")
        lines.append(f"    * Overall Grade : Grade {self.qr.overall_letter} ({self.qr.overall_grade:.1f}/4.0)")
        lines.append(f"    * Reason        : {self.qr.reason}")
        
        # Each ISO parameter grade
        if self.qr.iso_parameters:
            lines.append(f"    * ISO Parameter Grades:")
            param_labels = {
                "cell_contrast": "Cell Contrast",
                "cell_modulation": "Cell Modulation",
                "pattern_damage": "Pattern Damage",
                "decode": "Decode",
                "grid_non_uniformity": "Grid Non-Uniformity",
                "axial_non_uniformity": "Axial Non-Uniformity",
                "unused_ecc": "Unused ECC",
                # Backward-compat aliases
                "symbol_contrast": "Cell Contrast",
                "modulation": "Cell Modulation",
                "unused_error_correction": "Unused ECC",
            }
            order = [
                "cell_contrast",
                "cell_modulation",
                "pattern_damage",
                "decode",
                "grid_non_uniformity",
                "axial_non_uniformity",
                "unused_ecc",
            ]
            for p_key in order:
                if p_key in self.qr.iso_parameters:
                    p_val = self.qr.iso_parameters[p_key]
                    lbl = param_labels.get(p_key, p_key)
                    p_grade = p_val.get("letter", "N/A")
                    p_num = p_val.get("value", 0.0)
                    p_pass = "PASS" if p_val.get("passed", False) else "FAIL"
                    p_thresh = p_val.get("threshold", "")
                    thresh_str = f" [Limit: {p_thresh}]" if p_thresh else ""
                    lines.append(f"      - {lbl:24s}: Grade {p_grade} ({p_num:.3f}) [{p_pass}]{thresh_str}")
        lines.append(f"    * Speed         : {self.qr.latency_ms:.2f} ms")
        lines.append(subdiv)

        # -------------------------------------------------------------
        # B) OCR / OCV MODULE
        # -------------------------------------------------------------
        ocr_badge = f"[{self.ocr.status_str}]"
        lines.append(f" b) OCR / OCV MODULE: {ocr_badge}")
        lines.append(f"    * Extracted Text: {self.ocr.text_ext if self.ocr.text_ext else 'None'}")
        if self.ocr.parsed_fields:
            lines.append(f"    * Parsed Fields :")
            for f_name, f_val in self.ocr.parsed_fields.items():
                lines.append(f"      - {f_name:12s}: {f_val}")
        lines.append(f"    * OCV Reason    : {self.ocr.ocv_reason}")
        lines.append(f"    * Reason        : {self.ocr.reason}")
        lines.append(f"    * Speed         : {self.ocr.latency_ms:.2f} ms")
        lines.append(subdiv)

        # -------------------------------------------------------------
        # C) PHARMACODE MODULE
        # -------------------------------------------------------------
        pharma_badge = f"[{self.pharma.status_str}]"
        lines.append(f" c) PHARMACODE MODULE: {pharma_badge}")
        if self.pharma.found and self.pharma.payload is not None:
            lines.append(f"    * Decoded Code  : {self.pharma.payload}")
            lines.append(f"    * Bar Details   : {self.pharma.bar_count} bars [{self.pharma.bar_pattern}] (1=Narrow, 2=Wide)")
            lines.append(f"    * Direction Used: {self.pharma.direction_used} (R2L: {self.pharma.r2l} | L2R: {self.pharma.l2r})")
            lines.append(f"    * Streak Height : {self.pharma.streak} consecutive scanlines")
        else:
            lines.append(f"    * Decoded Code  : None")
        lines.append(f"    * Reason        : {self.pharma.reason}")
        lines.append(f"    * Speed         : {self.pharma.latency_ms:.2f} ms")
        lines.append(subdiv)

        # -------------------------------------------------------------
        # D) 1D RETAIL BARCODE MODULE (EAN-13, EAN-8, UPC-A, Code 128)
        # -------------------------------------------------------------
        bc_badge = f"[{self.barcode.status_str}]"
        lines.append(f" d) 1D RETAIL BARCODE MODULE: {bc_badge}")
        if self.barcode.found and self.barcode.payload:
            lines.append(f"    * Decoded Code  : {self.barcode.payload} (Format: {self.barcode.format})")
            lines.append(f"    * GS1 Origin    : {self.barcode.gs1_origin}")
            lines.append(f"    * Checksum      : {'VALID (Modulo-10 PASS)' if self.barcode.checksum_valid else 'FAIL'}")
            lines.append(f"    * Orientation   : {self.barcode.orientation} degrees")
        else:
            lines.append(f"    * Decoded Code  : None")
        lines.append(f"    * Reason        : {self.barcode.reason}")
        lines.append(f"    * Speed         : {self.barcode.latency_ms:.2f} ms")
        lines.append(subdiv)

        # -------------------------------------------------------------
        # E) OPTICAL QUALITY & TIMINGS
        # -------------------------------------------------------------
        lines.append(f" e) OPTICAL QUALITY & PIPELINE TIMINGS:")
        opt_s = "OK" if self.optical_valid else f"NOK ({self.optical_error})"
        lines.append(f"    * Optical Gate  : [{opt_s}] Brightness: {self.optical_metrics.get('mean_brightness', 0):.1f} | Glare: {self.optical_metrics.get('glare_saturated_ratio', 0)*100:.1f}% | Blur: {self.optical_metrics.get('blur_laplacian_var', 0):.1f}")
        lats = self.latencies
        lines.append(f"    * Latency Split : Capture: {lats.optical_check_ms:.1f}ms | ROI: {lats.roi_extraction_ms:.1f}ms | QR: {lats.qr_ms:.1f}ms | OCR: {lats.ocr_ms:.1f}ms | Pharma: {lats.pharma_ms:.1f}ms | Barcode: {lats.barcode_ms:.1f}ms")
        lines.append(divider)

        return "\n".join(lines)


class EnzeneIntegratedPipeline:
    """
    Unified multi-engine pharmaceutical carton inspection pipeline.

    Takes a camera frame from Capture Module, extracts multi-ROIs, and concurrently/sequentially
    dispatches each region to:
    1. QR Verifier (ISO 15415)
    2. RapidOCR / OCV Engine
    3. Shun74 Pharmacode Detector
    """

    def __init__(
        self,
        gateway_config: Optional[GatewayConfig] = None,
        roi_config: Optional[ROIConfig] = None,
        qr_pass_threshold: ISOGrade = ISOGrade.C,
        ocr_blur_threshold: float = 10.0,
        enable_qr: bool = True,
        enable_ocr: bool = True,
        enable_pharma: bool = True,
        enable_barcode: bool = True,
        target_ocr_data: Optional[Dict[str, str]] = None,
        expected_pharmacode: Optional[int] = None,
        expected_barcode: Optional[str] = None,
        pharmacode_direction: str = "AUTO",
        fallback_pharma_full_frame: bool = True,
    ) -> None:
        """
        Initialize all engines and configuration parameters.
        """
        # Configurations
        self.gateway_config = gateway_config or GatewayConfig(
            target_resolution=None,       # Allow dynamic webcam / camera resolution
            blur_threshold=30.0,          # Standard industrial line blur gate
            min_brightness=25.0,
            max_brightness=240.0,
        )
        self.roi_config = roi_config or ROIConfig(
            ocr_x_gap_ratio=0.08,
            ocr_width_ratio=3.60,
            ocr_height_ratio=1.45,
            ocr_y_offset_ratio=-0.20,
            enable_pharmacode=True,
            pharmacode_direction=pharmacode_direction,
            expected_pharmacode=expected_pharmacode,
        )

        self.enable_qr = enable_qr
        self.enable_ocr = enable_ocr
        self.enable_pharma = enable_pharma
        self.enable_barcode = enable_barcode
        self.target_ocr_data = target_ocr_data
        self.expected_pharmacode = expected_pharmacode
        self.expected_barcode = expected_barcode
        self.pharmacode_direction = pharmacode_direction
        self.fallback_pharma_full_frame = fallback_pharma_full_frame

        # Initialize Sub-Engines
        logger.info("Initializing Enzene Integrated Inspection Engines...")
        self.roi_extractor = MultiROIExtractor(config=self.roi_config)
        self.qr_verifier = QRVerifier(pass_threshold=qr_pass_threshold)
        self.ocr_engine = IndustrialOCVEngine(min_blur_score=ocr_blur_threshold)
        self.barcode_detector = Industrial1DBarcodeDetector(expected_code=expected_barcode)
        logger.info("All engines initialized successfully.")

        self._frame_count = 0

    def set_pharmacode_direction(self, direction: str) -> None:
        """Dynamically update Pharmacode decoding direction across pipeline and ROI extractor."""
        self.pharmacode_direction = direction
        self.roi_config = ROIConfig(
            ocr_x_gap_ratio=self.roi_config.ocr_x_gap_ratio,
            ocr_width_ratio=self.roi_config.ocr_width_ratio,
            ocr_height_ratio=self.roi_config.ocr_height_ratio,
            ocr_y_offset_ratio=self.roi_config.ocr_y_offset_ratio,
            enable_pharmacode=self.roi_config.enable_pharmacode,
            pharmacode_direction=direction,
            expected_pharmacode=self.expected_pharmacode,
        )
        self.roi_extractor = MultiROIExtractor(config=self.roi_config)

    def evaluate_optical_quality(self, frame: np.ndarray) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Applies optical validation rules from Capture Module.
        Returns: (is_valid, error_code_str, metrics_dict)
        """
        if frame is None or frame.size == 0:
            return False, FrameErrorCode.ERR_STREAM_DISCONNECTED.value, {}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = gray.shape[:2]
        metrics: Dict[str, Any] = {"width": w, "height": h}

        # 1. Illumination
        mean_brightness = float(cv2.mean(gray)[0])
        metrics["mean_brightness"] = round(mean_brightness, 2)
        if mean_brightness < self.gateway_config.min_brightness:
            return False, FrameErrorCode.ERR_UNDER_EXPOSED.value, metrics
        if mean_brightness > self.gateway_config.max_brightness:
            return False, FrameErrorCode.ERR_OVER_EXPOSED_GLARE.value, metrics

        # 2. Glare hotspot
        stride = self.gateway_config.stride_for_percentiles
        sampled = gray[::stride, ::stride]
        sat_ratio = float(np.mean(sampled >= 254))
        metrics["glare_saturated_ratio"] = round(sat_ratio, 4)
        if sat_ratio > self.gateway_config.max_glare_pixel_ratio:
            return False, FrameErrorCode.ERR_OVER_EXPOSED_GLARE.value, metrics

        # 3. Blur / Sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_var = float(laplacian.var())
        metrics["blur_laplacian_var"] = round(blur_var, 2)
        if blur_var < self.gateway_config.blur_threshold:
            return False, FrameErrorCode.ERR_FRAME_BLURRY.value, metrics

        return True, None, metrics

    def process_frame(
        self,
        frame: np.ndarray,
        run_optical_checks: bool = True,
        force_ocr: bool = True,
    ) -> IntegratedInspectionResult:
        """
        Execute full end-to-end integration cycle on a single camera frame.
        """
        self._frame_count += 1
        start_pipeline = time.perf_counter()
        
        result = IntegratedInspectionResult(frame_index=self._frame_count)
        lat = result.latencies

        # =====================================================================
        # Step 1: Capture Module Quality Gate
        # =====================================================================
        if run_optical_checks:
            t0 = time.perf_counter()
            opt_valid, opt_err, opt_met = self.evaluate_optical_quality(frame)
            lat.optical_check_ms = (time.perf_counter() - t0) * 1000.0
            result.optical_valid = opt_valid
            result.optical_error = opt_err
            result.optical_metrics = opt_met

            if not opt_valid:
                lat.total_ms = (time.perf_counter() - start_pipeline) * 1000.0
                return result
        else:
            result.optical_valid = True

        # =====================================================================
        # Step 2: Multi-ROI Segmentation & Universal Horizon Straightening
        # =====================================================================
        t0 = time.perf_counter()
        roi_output = self.roi_extractor.extract(frame)
        lat.roi_extraction_ms = (time.perf_counter() - t0) * 1000.0

        straight_frame = roi_output.get("straight_frame")
        if straight_frame is None or straight_frame.size == 0:
            straight_frame = frame
        result.straight_frame = straight_frame

        result.roi_status = roi_output.get("status", "UNKNOWN")
        result.qr_roi = roi_output.get("qr_roi")
        result.ocr_roi = roi_output.get("ocr_roi")
        result.pharma_roi = roi_output.get("pharma_roi")
        result.qr_corners = roi_output.get("qr_corners")
        result.ocr_corners = roi_output.get("ocr_corners")
        result.pharma_corners = roi_output.get("pharma_corners")

        # =====================================================================
        # Step 3: Module 1 - QR Verification
        # =====================================================================
        if self.enable_qr and result.qr_roi is not None:
            t0 = time.perf_counter()
            qr_res: QRVerificationResult = self.qr_verifier.verify(result.qr_roi)
            qr_ms = (time.perf_counter() - t0) * 1000.0

            is_pass = (qr_res.status == "PASS" and qr_res.payload is not None)
            qr_ok = "OK" if is_pass else "NOK"
            
            # Formulate clear reason
            if is_pass:
                qr_reason = "ISO 15415 verification passed"
            elif qr_res.failing_metrics:
                qr_reason = f"Failing ISO metrics: {', '.join(qr_res.failing_metrics)}"
            elif qr_res.payload is None:
                qr_reason = "QR decode failure / unreadable matrix"
            else:
                qr_reason = f"Grade {qr_res.overall_letter} below pass threshold"

            result.qr = QRInspectionSummary(
                detected=(qr_res.payload is not None),
                status_str=qr_ok,
                payload=qr_res.payload,
                reason=qr_reason,
                overall_letter=qr_res.overall_letter,
                overall_grade=qr_res.overall_grade,
                iso_parameters=qr_res.metrics,
                failing_metrics=qr_res.failing_metrics,
                latency_ms=qr_ms,
            )
            lat.qr_ms = qr_ms
        elif result.qr_roi is None:
            result.qr.status_str = "NOK"
            result.qr.reason = "ERR_QR_ANCHOR_NOT_FOUND (QR not detected in frame)"

        # =====================================================================
        # Step 4: Module 2 - OCR & OCV Engine
        # =====================================================================
        if self.enable_ocr and result.ocr_roi is not None and force_ocr:
            t0 = time.perf_counter()
            ocr_res = self.ocr_engine.process_frame(
                result.ocr_roi,
                target_data=self.target_ocr_data,
                annotate=False,
            )
            ocr_ms = (time.perf_counter() - t0) * 1000.0

            raw_lines = ocr_res.get("raw_ocr_lines", [])
            parsed_text = "\n".join([line.get("text", "") for line in raw_lines])

            # Extract structured parameters (GTIN, UID, LOT, EXP)
            parsed_dict = {}
            if extract_parameter_candidates and raw_lines:
                cand_input = [(None, l.get("text", ""), l.get("confidence", 0.0)) for l in raw_lines]
                extracted = extract_parameter_candidates(cand_input)
                for k, v in extracted.items():
                    if isinstance(v, dict) and "value" in v:
                        parsed_dict[k] = v["value"]

            # Determine OCV reason and OK/NOK status
            ocv_rep = ocr_res.get("ocv_report")
            res_status = ocr_res.get("status", "FAIL")
            ocr_ok = "OK" if res_status in ("PASS", "SUCCESS") and raw_lines else "NOK"

            if ocv_rep and "parameters" in ocv_rep:
                param_reasons = []
                for p_name, p_data in ocv_rep["parameters"].items():
                    param_reasons.append(f"{p_name}:{p_data.get('status', 'FAIL')}")
                ocv_reason = " | ".join(param_reasons)
            elif self.target_ocr_data is not None:
                ocv_reason = "OCV verification failed"
            else:
                ocv_reason = "OCV master target not configured (Extraction only)"

            result.ocr = OCRInspectionSummary(
                status_str=ocr_ok,
                text_ext=parsed_text,
                raw_lines=raw_lines,
                parsed_fields=parsed_dict,
                ocv_report=ocv_rep,
                ocv_reason=ocv_reason,
                reason=ocr_res.get("reason", "None" if ocr_ok == "OK" else "OCR extraction failure"),
                latency_ms=ocr_ms,
            )
            lat.ocr_ms = ocr_ms
        elif result.ocr_roi is None:
            result.ocr.status_str = "NOK"
            result.ocr.reason = "NO_OCR_ROI (Awaiting valid QR anchor handoff)"

        # =====================================================================
        # Step 5: Module 3 - 1D Retail Barcode Engine (EAN-13, EAN-8, UPC, Code 128)
        # Always scans the universal leveled straight_frame
        # =====================================================================
        barcode_bbox_for_pharma = None
        if self.enable_barcode:
            t0 = time.perf_counter()
            b_res = self.barcode_detector.scan(straight_frame)
            b_ms = (time.perf_counter() - t0) * 1000.0
            lat.barcode_ms = b_ms

            if b_res.found and b_res.corners is not None:
                result.barcode_corners = b_res.corners
                bx, by, bw, bh = b_res.bbox or (0, 0, 0, 0)
                if bw > 0 and bh > 0:
                    barcode_bbox_for_pharma = (bx, by, bw, bh)
                    pad = 4
                    px1 = max(0, bx - pad)
                    py1 = max(0, by - pad)
                    px2 = min(straight_frame.shape[1], bx + bw + pad)
                    py2 = min(straight_frame.shape[0], by + bh + pad)
                    result.barcode_roi = straight_frame[py1:py2, px1:px2].copy()

            result.barcode = BarcodeSummary(
                found=b_res.found,
                status_str=b_res.status_str,
                payload=b_res.payload,
                format=b_res.format,
                checksum_valid=b_res.checksum_valid,
                gs1_origin=b_res.gs1_origin,
                orientation=b_res.orientation,
                reason=b_res.reason,
                latency_ms=b_res.latency_ms,
                details=b_res.details,
            )

        # =====================================================================
        # Step 6: Module 4 - Pharmacode 1D Profiler (with Dynamic Mutual Exclusion)
        # =====================================================================
        if self.enable_pharma:
            t0 = time.perf_counter()
            pharma_found = False
            pharma_payload = None
            pharma_details: Dict[str, Any] = {}

            # Case A: Localized by ROI Extractor
            if roi_output.get("has_pharmacode") and roi_output.get("pharmacode_payload") is not None:
                p_cand_bbox = roi_output.get("pharmacode_details", {}).get("bbox")
                is_overlap_barcode = False
                if barcode_bbox_for_pharma and p_cand_bbox:
                    bx, by, bw, bh = barcode_bbox_for_pharma
                    cx, cy = p_cand_bbox[0] + p_cand_bbox[2] / 2.0, p_cand_bbox[1] + p_cand_bbox[3] / 2.0
                    if (bx <= cx <= bx + bw) and (by <= cy <= by + bh):
                        is_overlap_barcode = True

                if not is_overlap_barcode:
                    pharma_found = True
                    pharma_payload = roi_output.get("pharmacode_payload")
                    pharma_details = roi_output.get("pharmacode_details", {})
                    result.pharma_roi = roi_output.get("pharma_roi")
                    result.pharma_corners = roi_output.get("pharma_corners")

            # Case B: Scan universal leveled straight_frame
            if not pharma_found and self.fallback_pharma_full_frame:
                frame_gray = straight_frame if straight_frame.ndim == 2 else cv2.cvtColor(straight_frame, cv2.COLOR_BGR2GRAY)
                fh, fw = frame_gray.shape[:2]

                f_found, f_val, f_info = scan_pharmacode_roi(
                    frame_gray,
                    direction=self.pharmacode_direction,
                    expected_payload=self.expected_pharmacode,
                    barcode_bbox=barcode_bbox_for_pharma,
                )
                if f_found and "bbox" in f_info:
                    pharma_found = True
                    pharma_payload = f_val
                    pharma_details = f_info
                    bx, by, bw, bh = f_info["bbox"]
                    pad = 4
                    px1 = max(0, bx - pad)
                    py1 = max(0, by - pad)
                    px2 = min(fw, bx + bw + pad)
                    py2 = min(fh, by + bh + pad)
                    result.pharma_roi = straight_frame[py1:py2, px1:px2].copy()
                    result.pharma_corners = np.array([
                        [px1, py1],
                        [px2, py1],
                        [px2, py2],
                        [px1, py2],
                    ], dtype=np.int32)

            pharma_ms = (time.perf_counter() - t0) * 1000.0

            # Determine Pharmacode OK / NOK
            bars = pharma_details.get("bars", [])
            bar_pattern_str = "".join(["W" if b == 2 else "N" for b in bars]) if bars else ""
            if pharma_found and pharma_payload is not None:
                if self.expected_pharmacode is not None:
                    pharma_ok = "OK" if (pharma_payload == self.expected_pharmacode) else "NOK"
                    pharma_reason = "Match with master batch code" if pharma_ok == "OK" else f"Mismatch (Expected {self.expected_pharmacode}, got {pharma_payload})"
                else:
                    pharma_ok = "OK"
                    pharma_reason = f"Valid physical Pharmacode ({len(bars)} bars, pitch std < 28%)"
            else:
                pharma_ok = "NOK"
                pharma_reason = "NO_PHARMACODE_DETECTED"

            result.pharma = PharmacodeSummary(
                found=pharma_found,
                status_str=pharma_ok,
                payload=pharma_payload,
                reason=pharma_reason,
                bar_count=len(bars),
                bars=bars,
                bar_pattern=bar_pattern_str,
                r2l=pharma_details.get("r2l", pharma_payload),
                l2r=pharma_details.get("l2r", pharma_payload),
                direction_used=pharma_details.get("direction_used", "R2L"),
                streak=pharma_details.get("streak", 0),
                confidence=pharma_details.get("confidence", 0),
                latency_ms=pharma_ms,
                details=pharma_details,
            )
            lat.pharma_ms = pharma_ms

        # Total pipeline time
        lat.total_ms = (time.perf_counter() - start_pipeline) * 1000.0
        return result
