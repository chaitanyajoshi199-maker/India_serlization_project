"""
Batch Synthetic Image Inspection Pipeline Runner.

Processes a batch of synthetic carton images sequentially through the full Enzene
multi-engine inspection pipeline (QR Verifier, OCR/OCV Engine, Pharmacode Profiler).

Stores the final results into:
- synthetic_images/results/res.json
- synthetic_images/results/anotated_res_image/ (compressed annotated images with HUD)
- synthetic_images/results/raw_nok_images/ (raw uncompressed images ONLY for NOK frames)

Usage:
    python batch_inspect_synthetic.py                  # Process all images in synthetic_images/
    python batch_inspect_synthetic.py --limit 10       # Process first 10 images
    python batch_inspect_synthetic.py --optical        # Enable optical quality checks gate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Ensure project root in sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from integration_pipeline import (
    EnzeneIntegratedPipeline,
    IntegratedInspectionResult,
)
from capture.camera_gateway import GatewayConfig
from qr_verifier import ISOGrade
from live_camera_feed_pipeline import annotate_inspection_frame


def run_batch_inspection(
    input_dir: Path = ROOT / "synthetic_images",
    output_dir: Path = ROOT / "synthetic_images" / "results",
    limit: Optional[int] = None,
    run_optical: bool = False,
    qr_pass_threshold: ISOGrade = ISOGrade.C,
    expected_pharma: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes batch processing of all images in input_dir through the inspection pipeline.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    ann_dir = output_dir / "anotated_res_image"
    raw_nok_dir = output_dir / "raw_nok_images"
    json_path = output_dir / "res.json"

    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    raw_nok_dir.mkdir(parents=True, exist_ok=True)

    # Collect and sort image files
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ])

    if limit is not None and limit > 0:
        image_paths = image_paths[:limit]

    total_images = len(image_paths)
    if total_images == 0:
        print(f"[ERROR] No valid images found in: {input_dir.resolve()}")
        return {"error": "NO_IMAGES_FOUND"}

    print("\n" + "=" * 90)
    print(" ENZENE MULTI-MODULE BATCH SYNTHETIC IMAGE INSPECTION")
    print("=" * 90)
    print(f" Input Directory   : {input_dir.resolve()}")
    print(f" Total Images Found: {total_images}")
    print(f" Output Directory  : {output_dir.resolve()}")
    print(f"   * JSON Results  : {json_path.resolve()}")
    print(f"   * Annotated Imgs: {ann_dir.resolve()} (Compressed JPEG)")
    print(f"   * Raw NOK Imgs  : {raw_nok_dir.resolve()} (Uncompressed PNG)")
    print(f" Optical Checks    : {'ENABLED' if run_optical else 'DISABLED'}")
    print("=" * 90 + "\n")

    # Initialize Integration Pipeline
    gw_cfg = GatewayConfig(
        target_resolution=None,
        blur_threshold=15.0,
        max_glare_pixel_ratio=0.10,
        min_brightness=15.0,
        max_brightness=250.0,
    )
    pipeline = EnzeneIntegratedPipeline(
        gateway_config=gw_cfg,
        qr_pass_threshold=qr_pass_threshold,
        enable_qr=True,
        enable_ocr=True,
        enable_pharma=True,
        target_ocr_data=None,               # Pure OCR extraction
        expected_pharmacode=expected_pharma,
        fallback_pharma_full_frame=True,
    )

    results_list: List[Dict[str, Any]] = []
    stats = {
        "total": total_images,
        "ok_count": 0,
        "nok_count": 0,
        "qr_detected": 0,
        "pharma_detected": 0,
        "ocr_detected": 0,
        "total_latency_ms": 0.0,
    }

    t_start_batch = time.perf_counter()

    for idx, img_path in enumerate(image_paths, 1):
        raw_frame = cv2.imread(str(img_path))
        if raw_frame is None or raw_frame.size == 0:
            print(f"[{idx:03d}/{total_images:03d}] [SKIP] Unable to read: {img_path.name}")
            continue

        # 1. Process Frame through Unified Pipeline
        res: IntegratedInspectionResult = pipeline.process_frame(
            frame=raw_frame,
            run_optical_checks=run_optical,
            force_ocr=True,
        )

        # 2. Build initial backend dictionary for annotation rendering
        initial_backend = res.to_backend_result(raw_frame=raw_frame)

        # 3. Render High-Visibility Industrial Overlay
        annotated_vis = annotate_inspection_frame(raw_frame, res, initial_backend)

        # 4. Finalize Backend Dictionary (Compressed annotated image + NOK-only raw image)
        backend_dict = res.to_backend_result(
            raw_frame=raw_frame,
            annotated_frame=annotated_vis,
        )

        is_ok = (backend_dict.get("status") == "OK")
        if is_ok:
            stats["ok_count"] += 1
        else:
            stats["nok_count"] += 1

        if res.qr.detected:
            stats["qr_detected"] += 1
        if res.pharma.found:
            stats["pharma_detected"] += 1
        if bool(res.ocr.raw_lines):
            stats["ocr_detected"] += 1
        stats["total_latency_ms"] += res.latencies.total_ms

        # 5. Save Compressed Annotated Image to anotated_res_image/
        ann_filename = f"{img_path.stem}_annotated.jpg"
        ann_file_path = ann_dir / ann_filename
        compressed_bytes = backend_dict.get("annotated_frame")
        if isinstance(compressed_bytes, (bytes, bytearray)):
            with open(ann_file_path, "wb") as f:
                f.write(compressed_bytes)
        else:
            cv2.imwrite(str(ann_file_path), annotated_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        # 6. Save Raw Uncompressed Image to raw_nok_images/ ONLY if frame is NOK
        raw_rel_path: Optional[str] = None
        if not is_ok:
            raw_filename = f"{img_path.stem}_raw_nok.png"
            raw_file_path = raw_nok_dir / raw_filename
            cv2.imwrite(str(raw_file_path), raw_frame)
            raw_rel_path = f"raw_nok_images/{raw_filename}"

        # 7. Compile Detailed Record for res.json
        record = {
            "frame_index": idx,
            "file_name": img_path.name,
            "timestamp": datetime.now().isoformat(),
            "status": backend_dict.get("status", "NOK"),
            "overall_status": backend_dict.get("overall_status", "NOK"),
            "qr": backend_dict.get("qr", {}),
            "pharmacode": backend_dict.get("pharmacode", {}),
            "ocr": backend_dict.get("ocr", {}),
            "annotated_image": f"anotated_res_image/{ann_filename}",
            "raw_nok_image": raw_rel_path,
            "latencies_ms": {
                "optical": round(res.latencies.optical_check_ms, 2),
                "roi": round(res.latencies.roi_extraction_ms, 2),
                "qr": round(res.latencies.qr_ms, 2),
                "ocr": round(res.latencies.ocr_ms, 2),
                "pharma": round(res.latencies.pharma_ms, 2),
                "total": round(res.latencies.total_ms, 2),
            },
        }
        results_list.append(record)

        # Terminal Progress Reporting
        status_badge = "[OK ]" if is_ok else "[NOK]"
        qr_info = f"Grade {res.qr.overall_letter}" if res.qr.detected else "None"
        pharma_info = f"Code {res.pharma.payload}" if res.pharma.found else "None"
        ocr_info = f"{len(res.ocr.raw_lines)} lines" if res.ocr.raw_lines else "None"
        print(
            f" [{idx:03d}/{total_images:03d}] {status_badge} {img_path.name[:35]:35s} | "
            f"QR: {qr_info:7s} | Pharma: {pharma_info:9s} | OCR: {ocr_info:7s} | {res.latencies.total_ms:5.1f}ms"
        )

    # 8. Persist All Inspection Telemetry to res.json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)

    total_batch_time = time.perf_counter() - t_start_batch
    avg_latency = (stats["total_latency_ms"] / max(total_images, 1))

    # 9. Summary Report
    print("\n" + "=" * 90)
    print(" BATCH INSPECTION SUMMARY REPORT")
    print("=" * 90)
    print(f" Total Processed  : {len(results_list)} / {total_images}")
    print(f" Passing (OK)     : {stats['ok_count']} ({stats['ok_count']/max(1, len(results_list))*100:.1f}%)")
    print(f" Failing (NOK)    : {stats['nok_count']} ({stats['nok_count']/max(1, len(results_list))*100:.1f}%)")
    print(f" QR Detected      : {stats['qr_detected']} / {len(results_list)}")
    print(f" Pharma Detected  : {stats['pharma_detected']} / {len(results_list)}")
    print(f" OCR Extracted    : {stats['ocr_detected']} / {len(results_list)}")
    print(f" Average Latency  : {avg_latency:.2f} ms per frame")
    print(f" Total Batch Time : {total_batch_time:.2f} seconds")
    print("-" * 90)
    print(f" Results Saved To :")
    print(f"   * JSON Report  : {json_path.resolve()}")
    print(f"   * Annotated Img: {ann_dir.resolve()}")
    print(f"   * Raw NOK Img  : {raw_nok_dir.resolve()}")
    print("=" * 90 + "\n")

    return {
        "stats": stats,
        "json_path": str(json_path.resolve()),
        "annotated_dir": str(ann_dir.resolve()),
        "raw_nok_dir": str(raw_nok_dir.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description="Batch Synthetic Image Inspection Pipeline Runner.")
    parser.add_argument(
        "--input-dir",
        default=str(ROOT / "synthetic_images"),
        help="Input directory containing synthetic images (default: synthetic_images/)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "synthetic_images" / "results"),
        help="Output directory to save res.json and annotated images (default: synthetic_images/results/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of images to inspect (e.g. --limit 10)",
    )
    parser.add_argument(
        "--optical",
        action="store_true",
        help="Enable optical gatekeeper checks (blur, exposure, glare)",
    )
    parser.add_argument(
        "--expected-pharma",
        type=int,
        default=None,
        help="Expected Pharmacode integer payload to validate against",
    )
    args = parser.parse_args()

    run_batch_inspection(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        run_optical=args.optical,
        expected_pharma=args.expected_pharma,
    )


if __name__ == "__main__":
    main()
