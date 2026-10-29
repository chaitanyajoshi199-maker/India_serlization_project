"""
Production-Grade Industrial OCR Engine for Pharmaceutical Packaging.

Self-contained, high-performance text extraction:
1. Fast ONNX CPU Text Extraction (RapidOCR with zero-loss optimizations)
2. Spatial Horizontal Line Clustering (merges fragmented words on same row)
3. Structured Field Extraction (Regex matching for GTIN, Batch/Lot, Expiry, Serial/UID)

Zero OCV validation / Zero duplicate blur gates / Zero external helper scripts.
"""
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# Serialization Field Extraction Patterns
PARAM_PATTERNS = {
    "GTIN": [
        r"(?:GTIN|01)[\s:]*([0-9]{8,14})",
        r"\b([0-9]{14})\b",
        r"\b([0-9]{12,14})\b",
    ],
    "UNIQUE_CODE": [
        r"(?:SR|SN|UID|SERIAL|21)[\s:]*([A-Z0-9]{6,24})",
        r"(?:SR|SN)[\s:]*([0-9]{10,24})",
    ],
    "MFG_DATE": [
        r"(?:MFG|DATEOFMFG|MFD|DOM|11)[\s:]*([A-Z0-9\-\.\/]{4,10})",
        r"(?:MFG)[\s:]*([0-9]{2}[\-\/\.][0-9]{4})",
    ],
    "EXP_DATE": [
        r"(?:EXP|DATEOFEXP|ED|USEBEFORE|17)[\s:]*([A-Z0-9\-\.\/]{4,10})",
        r"(?:EXP)[\s:]*([0-9]{2}[\-\/\.][0-9]{4})",
    ],
    "BATCH_LOT": [
        r"(?:L0T|LOT|BATCH|BN|10)[\s:]*([A-Z0-9]{4,14})",
    ],
}


def init_optimized_ocr_engine() -> RapidOCR:
    """
    Initializes a lightweight, high-speed CPU instance of RapidOCR.
    - use_angle_cls=False: Skips 180-deg orientation classifier (labels are upright)
    - det_limit_type='max': Prevents artificial 2400px upscaling of small crops
    - det_limit_side_len=736: Preserves native resolution without downsampling blur
    - rec_batch_num=4: Vectorized SIMD/AVX2 multi-line inference
    """
    return RapidOCR(
        use_angle_cls=False,
        det_model_path=None,
        det_limit_type="max",
        det_limit_side_len=736,
        rec_model_path=None,
        rec_batch_num=4,
    )


def extract_parameter_candidates(
    ocr_lines: List[Tuple[Any, str, float]]
) -> Dict[str, Dict[str, Any]]:
    """
    Parses OCR text blocks and maps them to GTIN, Serial/Unique Code, EXP, and Batch/Lot.
    Handles standard print quirks (e.g. 'L0T' for 'LOT').
    """
    found: Dict[str, Dict[str, Any]] = {}

    for item in ocr_lines:
        if not item or len(item) < 2:
            continue
        box = item[0]
        raw_line = item[1]
        conf = float(item[2]) if len(item) > 2 else 1.0

        line_norm = raw_line.strip().upper()

        for param_key, patterns in PARAM_PATTERNS.items():
            if param_key not in found:
                for pat in patterns:
                    m = re.search(pat, line_norm)
                    if m:
                        found[param_key] = {
                            "raw": raw_line,
                            "value": m.group(1),
                            "box": box,
                            "conf": conf,
                        }
                        break

    return found


class IndustrialOCREngine:
    """
    High-throughput, production-ready OCR Engine for pharmaceutical packaging.
    Extracts multi-line text blocks with bounding boxes and per-line confidence scores.
    """

    def __init__(self, min_blur_score: float = 0.0):
        # min_blur_score kept in signature for backwards compatibility
        self.ocr = init_optimized_ocr_engine()

    def process_frame(
        self,
        image_input: Union[str, np.ndarray],
        target_data: Optional[Dict[str, str]] = None,
        annotate: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes pure text extraction and structured parsing on an image or crop.

        :param image_input: File path (str) or loaded BGR/grayscale image matrix (np.ndarray).
        :param target_data: Unused (kept for API signature compatibility).
        :param annotate: Unused (pipeline renders annotations centrally).
        :return: Standardized result dictionary.
        """
        start_time = time.perf_counter()

        # 1. Ingestion
        if isinstance(image_input, str):
            raw_img = cv2.imread(image_input)
            if raw_img is None:
                return {
                    "status": "ERROR",
                    "latency_ms": 0.0,
                    "reason": f"Failed to read image at: {image_input}",
                    "raw_ocr_lines": [],
                    "extracted_text_block": "",
                    "parsed_fields": {},
                    "ocv_report": None,
                    "annotated_frame": None,
                }
        else:
            raw_img = image_input

        # 2. Text Extraction
        ocr_result, _ = self.ocr(raw_img)

        if not ocr_result:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "status": "FAIL",
                "latency_ms": latency_ms,
                "reason": "No text characters detected in frame.",
                "raw_ocr_lines": [],
                "extracted_text_block": "",
                "parsed_fields": {},
                "ocv_report": None,
                "annotated_frame": None,
            }

        # 3. Format raw detections into [(box, text, score), ...]
        raw_detections = [(line[0], str(line[1]), float(line[2])) for line in ocr_result]

        # 4. Spatial Horizontal Line Clustering
        # Merges fragmented boxes on the same horizontal row (e.g. 'GTIN' + '1870...' -> 'GTIN: 1870...')
        parsed_lines = self._cluster_horizontal_lines(raw_detections)

        # 5. Extract Structured Fields (GTIN, UID, EXP, LOT)
        parsed_fields = {}
        extracted_candidates = extract_parameter_candidates(parsed_lines)
        for k, v in extracted_candidates.items():
            if isinstance(v, dict) and "value" in v:
                parsed_fields[k] = v["value"]

        clean_text_block = "\n".join([line[1] for line in parsed_lines])
        total_latency = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "status": "SUCCESS" if parsed_lines else "FAIL",
            "latency_ms": total_latency,
            "raw_ocr_lines": [
                {"text": line[1], "confidence": round(float(line[2]), 2), "box": line[0]}
                for line in parsed_lines
            ],
            "extracted_text_block": clean_text_block,
            "parsed_fields": parsed_fields,
            "ocv_report": None,
            "annotated_frame": None,
        }

    @staticmethod
    def _cluster_horizontal_lines(
        detections: List[Tuple[Any, str, float]],
        y_tolerance_ratio: float = 0.55,
    ) -> List[Tuple[Any, str, float]]:
        """
        Groups multi-word fragments on the same vertical line into single coherent lines.
        Eliminates fragmented boxes (e.g. 8 word fragments -> 4 actual lines).
        """
        if not detections:
            return []

        items = []
        for box, text, score in detections:
            pts = np.array(box, dtype=np.float32)
            ys = pts[:, 1]
            xs = pts[:, 0]
            h = float(ys.max() - ys.min())
            cy = float((ys.min() + ys.max()) / 2.0)
            items.append({
                "box": box,
                "text": text.strip(),
                "score": float(score),
                "min_x": float(xs.min()),
                "max_x": float(xs.max()),
                "min_y": float(ys.min()),
                "max_y": float(ys.max()),
                "cy": cy,
                "h": max(1.0, h),
            })

        # Sort top to bottom by vertical center
        items.sort(key=lambda it: it["cy"])

        # Cluster boxes belonging to the same horizontal line
        lines: List[List[Dict[str, Any]]] = []
        for it in items:
            placed = False
            for line in lines:
                line_cy = sum(k["cy"] for k in line) / len(line)
                avg_h = sum(k["h"] for k in line) / len(line)
                if abs(it["cy"] - line_cy) <= avg_h * y_tolerance_ratio:
                    line.append(it)
                    placed = True
                    break
            if not placed:
                lines.append([it])

        # Within each line, sort left-to-right and combine
        merged_lines: List[Tuple[Any, str, float]] = []
        for line in lines:
            line.sort(key=lambda it: it["min_x"])
            texts = [k["text"] for k in line]

            # Format key-value cleanly (e.g. 'GTIN' + '1870...' -> 'GTIN: 1870...')
            if len(texts) == 2 and not texts[0].endswith(":") and not texts[1].startswith(":"):
                combined_text = f"{texts[0]}: {texts[1]}"
            else:
                combined_text = " ".join(texts)

            combined_score = round(sum(k["score"] for k in line) / len(line), 2)
            min_x = min(k["min_x"] for k in line)
            min_y = min(k["min_y"] for k in line)
            max_x = max(k["max_x"] for k in line)
            max_y = max(k["max_y"] for k in line)
            combined_box = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]

            merged_lines.append((combined_box, combined_text, combined_score))

        return merged_lines


# Aliases for 100% backward compatibility
IndustrialOCVEngine = IndustrialOCREngine
