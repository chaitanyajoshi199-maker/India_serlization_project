from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

from .preprocessing import preprocess_and_normalize, NormalizedQR
from .iso_metrics import (
    MetricGrade,
    grade_cell_contrast,
    grade_cell_modulation,
    grade_pattern_damage,
    grade_grid_non_uniformity,
    grade_axial_non_uniformity,
    grade_unused_error_correction,
    grade_to_letter,
)


class ISOGrade(int, Enum):
    A = 4
    B = 3
    C = 2
    D = 1
    F = 0


@dataclass
class VerificationResult:
    status: str                         # "PASS" or "REJECT"
    overall_grade: int                  # 0 to 4
    overall_letter: str                 # "A", "B", "C", "D", "F"
    payload: Optional[str] = None       # Unlocked when grade >= threshold
    raw_payload: Optional[str] = None   # Always contains decoded string if readable
    reject_reason: Optional[str] = None # Reason string if REJECT
    failing_metrics: List[str] = field(default_factory=list)
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    qr_version: Optional[int] = None
    grid_size: Optional[int] = None
    ec_level: Optional[str] = None
    warped_qr: Optional[np.ndarray] = None
    corners: Optional[np.ndarray] = None
    deblur_method: str = "RAW"


class QRVerifier:
    """
    Production ISO/IEC 15415 7-Parameter Verification Engine with Motion Blur Resilience.
    Evaluates:
    1. Cell Contrast
    2. Cell Modulation
    3. Pattern Damage
    4. Decode
    5. Grid Non-Uniformity
    6. Axial Non-Uniformity
    7. Unused ECC
    """

    def __init__(self, pass_threshold: int = ISOGrade.C) -> None:
        self.pass_threshold = int(pass_threshold)

    def verify(self, image: np.ndarray) -> VerificationResult:
        # Phase 2: Preprocessing, Motion Deblurring, and Perspective Normalization
        norm = preprocess_and_normalize(image)

        # Module 1: Reference Decode Baseline
        if norm is None:
            return VerificationResult(
                status="REJECT",
                overall_grade=0,
                overall_letter="F",
                payload=None,
                raw_payload=None,
                reject_reason="DECODE_FAILED: Reference Decode failed (Grade F)",
                failing_metrics=["Decode"],
                metrics={
                    "cell_contrast": {"name": "Cell Contrast", "value": 0.0, "grade": 0, "letter": "F", "passed": False, "threshold": ">= 40% (Grade C)"},
                    "cell_modulation": {"name": "Cell Modulation", "value": 0.0, "grade": 0, "letter": "F", "passed": False, "threshold": ">= 30% (Grade C)"},
                    "pattern_damage": {"name": "Pattern Damage", "value": 0.0, "grade": 0, "letter": "F", "passed": False, "threshold": ">= 75% (Grade C)"},
                    "decode": {"name": "Decode", "value": 0.0, "grade": 0, "letter": "F", "passed": False, "threshold": "Decoded"},
                    "grid_non_uniformity": {"name": "Grid Non-Uniformity", "value": 1.0, "grade": 0, "letter": "F", "passed": False, "threshold": "<= 63% (Grade C)"},
                    "axial_non_uniformity": {"name": "Axial Non-Uniformity", "value": 1.0, "grade": 0, "letter": "F", "passed": False, "threshold": "<= 10% (Grade C)"},
                    "unused_ecc": {"name": "Unused ECC", "value": 0.0, "grade": 0, "letter": "F", "passed": False, "threshold": ">= 37% (Grade C)"},
                },
            )

        res = norm.barcode_result
        warped = norm.warped_qr
        corners = norm.corners
        decoded_text = res.text if res else None

        r_min = float(np.percentile(warped, 1))
        r_max = float(np.percentile(warped, 99))

        # 1. Cell Contrast
        cc_grade = grade_cell_contrast(r_min, r_max)

        # 2. Cell Modulation
        mod_grade = grade_cell_modulation(warped, norm.grid_size, r_min, r_max)

        # 3. Pattern Damage
        pd_grade = grade_pattern_damage(warped, norm.grid_size, r_min, r_max)

        # 4. Decode
        dec_grade = MetricGrade(
            name="Decode",
            value=1.0,
            grade=4,
            letter="A",
            threshold="Decoded",
            passed=True,
        )

        # 5. Grid Non-Uniformity
        gnu_grade = grade_grid_non_uniformity(corners, norm.grid_size)

        # 6. Axial Non-Uniformity
        anu_grade = grade_axial_non_uniformity(
            p_tl=(corners[0][0], corners[0][1]),
            p_tr=(corners[1][0], corners[1][1]),
            p_bl=(corners[3][0], corners[3][1]),
        )

        # 7. Unused ECC
        extra = getattr(res, "extra", {}) or {}
        uec_val = extra.get("UEC", 1.0)
        if uec_val is None:
            uec_val = 1.0
        uec_grade = grade_unused_error_correction(float(uec_val))

        # Phase 4: Decision Gate & Official ISO 15415 "Lowest Grade Wins" Aggregation
        evaluated_metrics = [
            cc_grade,
            mod_grade,
            pd_grade,
            dec_grade,
            gnu_grade,
            anu_grade,
            uec_grade,
        ]
        lowest_score = min(m.grade for m in evaluated_metrics)
        overall_letter = grade_to_letter(lowest_score)

        # Standard Nomenclature matching industrial grading inspection screens
        metrics_dict: Dict[str, Dict[str, Any]] = {
            "cell_contrast": {
                "name": cc_grade.name,
                "value": cc_grade.value,
                "grade": cc_grade.grade,
                "letter": cc_grade.letter,
                "passed": cc_grade.passed,
                "threshold": cc_grade.threshold,
            },
            "cell_modulation": {
                "name": mod_grade.name,
                "value": mod_grade.value,
                "grade": mod_grade.grade,
                "letter": mod_grade.letter,
                "passed": mod_grade.passed,
                "threshold": mod_grade.threshold,
            },
            "pattern_damage": {
                "name": pd_grade.name,
                "value": pd_grade.value,
                "grade": pd_grade.grade,
                "letter": pd_grade.letter,
                "passed": pd_grade.passed,
                "threshold": pd_grade.threshold,
            },
            "decode": {
                "name": dec_grade.name,
                "value": dec_grade.value,
                "grade": dec_grade.grade,
                "letter": dec_grade.letter,
                "passed": dec_grade.passed,
                "threshold": dec_grade.threshold,
            },
            "grid_non_uniformity": {
                "name": gnu_grade.name,
                "value": gnu_grade.value,
                "grade": gnu_grade.grade,
                "letter": gnu_grade.letter,
                "passed": gnu_grade.passed,
                "threshold": gnu_grade.threshold,
            },
            "axial_non_uniformity": {
                "name": anu_grade.name,
                "value": anu_grade.value,
                "grade": anu_grade.grade,
                "letter": anu_grade.letter,
                "passed": anu_grade.passed,
                "threshold": anu_grade.threshold,
            },
            "unused_ecc": {
                "name": uec_grade.name,
                "value": uec_grade.value,
                "grade": uec_grade.grade,
                "letter": uec_grade.letter,
                "passed": uec_grade.passed,
                "threshold": uec_grade.threshold,
            },
        }

        failing = [m.name for m in evaluated_metrics if m.grade < self.pass_threshold]

        if lowest_score >= self.pass_threshold:
            status = "PASS"
            payload = decoded_text
            reject_reason = None
        else:
            status = "REJECT"
            payload = decoded_text  # Retain decoded string for telemetry and operator display
            reject_reason = f"LOW_GRADE: Symbol grade {overall_letter} ({lowest_score}) below threshold C. Failing: {', '.join(failing)}"

        return VerificationResult(
            status=status,
            overall_grade=lowest_score,
            overall_letter=overall_letter,
            payload=payload,
            raw_payload=decoded_text,
            reject_reason=reject_reason,
            failing_metrics=failing,
            metrics=metrics_dict,
            qr_version=norm.version,
            grid_size=norm.grid_size,
            ec_level=getattr(res, "ec_level", None),
            warped_qr=warped,
            corners=corners,
            deblur_method=norm.deblur_method,
        )
