"""
Industrial Test Carton Image Generator.
Generates realistic pharmaceutical carton images with:
1. QR Code + Link written over it + Multi-line OCR Text block
2. Real Laetus Pharmacode with value written over it
3. Real GS1 EAN-13 Retail Barcode with standard human-readable digits underneath

Randomizes the positions of (QR+Text, Pharmacode, 1D Barcode) across the carton
without overlaps, and benchmarks the inspection pipeline on all generated images.
"""
from __future__ import annotations

import os
import random
import json
from io import BytesIO
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import qrcode
import barcode
from barcode.writer import ImageWriter
from PIL import Image, ImageDraw, ImageFont


def compute_ean13_check_digit(digits12: str) -> str:
    """Computes the standard GS1 Modulo-10 check digit for a 12-digit payload."""
    total = 0
    for idx, char in enumerate(digits12):
        d = int(char)
        weight = 3 if (idx % 2 == 1) else 1
        total += d * weight
    chk = (10 - (total % 10)) % 10
    return str(chk)


def encode_pharmacode_to_bars(val: int) -> List[int]:
    """Encodes an integer into Laetus Pharmacode bars (1=Narrow, 2=Wide)."""
    bars = []
    v = val
    while v > 0:
        if v % 2 == 1:
            bars.append(1)  # Narrow
            v = (v - 1) // 2
        else:
            bars.append(2)  # Wide
            v = (v - 2) // 2
    return list(reversed(bars))


def create_pharmacode_block(
    val: int,
    header_text: str = "",
    narrow_w: int = 3,
    wide_w: int = 9,
    space: int = 3,
    bar_h: int = 44,
    quiet_zone: int = 20,
) -> Image.Image:
    """
    Renders an isolated Pharmacode block with its value written over it.
    """
    bars = encode_pharmacode_to_bars(val)
    content_w = sum(wide_w if b == 2 else narrow_w for b in bars) + space * (len(bars) - 1)
    block_w = content_w + 2 * quiet_zone
    
    # Header text
    label = header_text if header_text else f"PHARMACODE: {val}"
    text_h = 24
    block_h = text_h + bar_h + 16

    # Realistic industrial paper white (non-glare)
    img = Image.new("RGB", (block_w, block_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Use default or basic font
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except IOError:
        font = ImageFont.load_default()

    # Center text over bars
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    tx = max(2, (block_w - tw) // 2)
    draw.text((tx, 3), label, fill=(10, 10, 10), font=font)

    # Draw solid black bars
    cur_x = quiet_zone
    bar_y1 = text_h + 4
    bar_y2 = bar_y1 + bar_h
    for b in bars:
        bw = wide_w if b == 2 else narrow_w
        draw.rectangle([cur_x, bar_y1, cur_x + bw - 1, bar_y2], fill=(15, 15, 15))
        cur_x += bw + space

    return img


def create_ean13_barcode_block(digits12: str, header_text: str = "") -> Image.Image:
    """
    Renders a standard GS1 EAN-13 Retail Barcode with 30 black bars and readable digits.
    """
    chk = compute_ean13_check_digit(digits12)
    full_code = digits12 + chk

    writer = ImageWriter()
    writer.set_options({
        "module_width": 0.33,
        "module_height": 13.0,
        "quiet_zone": 4.5,
        "font_size": 10,
        "text_distance": 3.5,
    })
    ean = barcode.get("ean13", full_code, writer=writer)
    fp = BytesIO()
    ean.write(fp)
    fp.seek(0)
    barcode_img = Image.open(fp).convert("RGB")

    bw, bh = barcode_img.size
    header_h = 22 if header_text else 0
    total_h = bh + header_h

    # Realistic industrial paper white
    block = Image.new("RGB", (bw, total_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(block)

    if header_text:
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), header_text, font=font)
        tw = bbox[2] - bbox[0]
        tx = max(4, (bw - tw) // 2)
        draw.text((tx, 2), header_text, fill=(20, 20, 20), font=font)

    block.paste(barcode_img, (0, header_h))
    return block


def create_qr_and_text_block(
    qr_payload: str,
    ocr_fields: Dict[str, str],
) -> Image.Image:
    """
    Renders the QR code with its link written over it, plus adjacent OCR text lines.
    """
    # 1. Generate QR Code
    qr = qrcode.QRCode(box_size=3, border=2)
    qr.add_data(qr_payload)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color=(240, 240, 240)).convert("RGB")
    qw, qh = qr_img.size

    # 2. Text styling
    try:
        font_link = ImageFont.truetype("arial.ttf", 12)
        font_ocr = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font_link = font_ocr = ImageFont.load_default()

    header_text = qr_payload
    lines = [f"{k}: {v}" for k, v in ocr_fields.items()]

    line_spacing = 22
    ocr_block_w = 260
    ocr_block_h = len(lines) * line_spacing + 10

    header_h = 24
    total_w = qw + 16 + ocr_block_w
    total_h = header_h + max(qh, ocr_block_h) + 6

    block = Image.new("RGB", (total_w, total_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(block)

    # Draw Link written directly over QR code
    draw.text((4, 3), header_text, fill=(15, 15, 15), font=font_link)

    # Paste QR Code below its link
    block.paste(qr_img, (4, header_h))

    # Draw OCR Text lines next to QR Code
    text_x = qw + 20
    text_y = header_h + 4
    for line in lines:
        draw.text((text_x, text_y), line, fill=(0, 0, 0), font=font_ocr)
        text_y += line_spacing

    return block


def check_overlap(
    rect1: Tuple[int, int, int, int],
    rect2: Tuple[int, int, int, int],
    margin: int = 30,
) -> bool:
    """Returns True if two bounding boxes (x, y, w, h) collide or are within margin."""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (
        (x1 + w1 + margin < x2) or
        (x2 + w2 + margin < x1) or
        (y1 + h1 + margin < y2) or
        (y2 + h2 + margin < y1)
    )


def generate_single_carton(
    carton_index: int,
    canvas_w: int = 1280,
    canvas_h: int = 960,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Assembles a complete carton frame with randomly positioned, non-overlapping components:
    - QR Code + Text Block (Link written over it)
    - Real Pharmacode (Code written over it)
    - Real EAN-13 Retail Barcode (30 bars + standard text)
    """
    # 1. Generate diverse payload data
    batch_num = random.randint(10000, 99999)
    qr_payload = f"LINK:ENZENE_{carton_index:03d}_BCH{batch_num}"
    ocr_fields = {
        "GTIN": f"1870123400{carton_index:04d}",
        "EXP": f"{random.randint(1, 12):02d}-20{random.randint(27, 31)}",
        "LOT": f"BCH{batch_num}",
        "SR": f"71234567890{carton_index:04d}",
    }

    # Pharmacode value (between 100 and 4000, 7 to 12 bars)
    pharma_val = random.choice([1500, 1954, 2797, 3001, 742, 1280, 2048, 865, 3410, 1120])
    
    # EAN-13 12 digits (starting with 890 for GS1 India)
    ean12 = f"890{random.randint(100000000, 999999999)}"
    ean_full = ean12 + compute_ean13_check_digit(ean12)

    # 2. Render Component Image Blocks
    block_qr = create_qr_and_text_block(qr_payload, ocr_fields)
    block_pharma = create_pharmacode_block(pharma_val, header_text=f"PHARMACODE: {pharma_val}")
    block_barcode = create_ean13_barcode_block(ean12, header_text="RETAIL BARCODE (EAN-13)")

    w_qr, h_qr = block_qr.size
    w_pharma, h_pharma = block_pharma.size
    w_bc, h_bc = block_barcode.size

    # 3. Canvas Background: Industrial Carton Cardboard
    # Light clean carton area inside realistic margins
    carton_pad_x = 40
    carton_pad_y = 40
    min_x, max_x = carton_pad_x + 20, canvas_w - carton_pad_x - 20
    min_y, max_y = carton_pad_y + 20, canvas_h - carton_pad_y - 20

    # Base background: clean packaging off-white
    base_color = random.randint(235, 246)
    canvas = np.full((canvas_h, canvas_w, 3), base_color, dtype=np.uint8)

    # Outer border / carton boundary
    cv2.rectangle(
        canvas,
        (carton_pad_x, carton_pad_y),
        (canvas_w - carton_pad_x, canvas_h - carton_pad_y),
        (190, 190, 190),
        2,
        lineType=cv2.LINE_AA,
    )
    # Carton Branding Banner at top
    cv2.putText(
        canvas,
        f"ENZENE BIOSCIENCES | Bevarest 100mg/4mL (r-DNA Origin)  [UNIT-0{carton_index}]",
        (carton_pad_x + 18, carton_pad_y + 26),
        cv2.FONT_HERSHEY_DUPLEX,
        0.58,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )

    # 4. Randomized Non-Overlapping Placement Algorithm
    blocks = [
        ("qr_text", block_qr, w_qr, h_qr),
        ("pharma", block_pharma, w_pharma, h_pharma),
        ("barcode", block_barcode, w_bc, h_bc),
    ]
    # Shuffle order of placement so no component gets priority
    random.shuffle(blocks)

    placed_rects: List[Tuple[int, int, int, int]] = []
    positions: Dict[str, Tuple[int, int, int, int]] = {}

    for name, blk, bw, bh in blocks:
        placed = False
        for _ in range(500):  # Maximum placement attempts
            rx = random.randint(min_x, max_x - bw)
            ry = random.randint(min_y + 35, max_y - bh)
            cand_rect = (rx, ry, bw, bh)

            if not any(check_overlap(cand_rect, r, margin=40) for r in placed_rects):
                placed_rects.append(cand_rect)
                positions[name] = cand_rect
                # Paste component into canvas
                comp_np = np.array(blk)
                canvas[ry : ry + bh, rx : rx + bw] = comp_np
                placed = True
                break

        if not placed:
            # Fallback grid position if random sampling exhausted
            ry = min_y + 40 + len(placed_rects) * 160
            rx = min_x + 30
            placed_rects.append((rx, ry, bw, bh))
            positions[name] = (rx, ry, bw, bh)
            comp_np = np.array(blk)
            canvas[ry : ry + bh, rx : rx + bw] = comp_np

    # Metadata record
    manifest = {
        "carton_index": carton_index,
        "qr": {
            "payload": qr_payload,
            "bbox": positions.get("qr_text"),
        },
        "ocr": ocr_fields,
        "pharmacode": {
            "payload": pharma_val,
            "bar_count": len(encode_pharmacode_to_bars(pharma_val)),
            "bbox": positions.get("pharma"),
        },
        "barcode": {
            "payload": ean_full,
            "format": "EAN13",
            "bbox": positions.get("barcode"),
        },
    }

    return canvas, manifest


def generate_carton_dataset(
    count: int = 10,
    output_dir: str = "output/synthetic_cartons",
) -> List[Dict[str, Any]]:
    """Generates `count` synthetic carton images and saves them to disk."""
    os.makedirs(output_dir, exist_ok=True)
    manifests: List[Dict[str, Any]] = []

    print(f"\n=======================================================")
    print(f" GENERATING {count} SYNTHETIC PHARMACEUTICAL CARTONS")
    print(f" Target Directory: {output_dir}")
    print(f"=======================================================")

    for i in range(1, count + 1):
        img, meta = generate_single_carton(i)
        filename = f"carton_{i:02d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, img)
        meta["image_path"] = filepath
        meta["filename"] = filename
        manifests.append(meta)

        p_box = meta["pharmacode"]["bbox"]
        b_box = meta["barcode"]["bbox"]
        q_box = meta["qr"]["bbox"]
        print(f"[{i:02d}/{count:02d}] {filename} -> QR at ({q_box[0]},{q_box[1]}) | Pharma: Code {meta['pharmacode']['payload']} at ({p_box[0]},{p_box[1]}) | Barcode: {meta['barcode']['payload']} at ({b_box[0]},{b_box[1]})")

    manifest_path = os.path.join(output_dir, "cartons_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifests, f, indent=2)

    print(f"\nManifest saved to: {manifest_path}")
    return manifests


def benchmark_cartons(cartons_dir: str = "output/synthetic_cartons") -> None:
    """Runs the integrated inspection pipeline on all synthetic cartons and reports accuracy."""
    import glob
    from integration_pipeline import EnzeneIntegratedPipeline

    files = sorted(glob.glob(os.path.join(cartons_dir, "carton_*.png")))
    if not files:
        print(f"No cartons found in {cartons_dir}. Please run generator first.")
        return

    print("\n" + "=" * 90)
    print(" BENCHMARKING ENZENE INSPECTION PIPELINE ON GENERATED CARTONS")
    print("=" * 90)

    p = EnzeneIntegratedPipeline(enable_ocr=True)

    passed_count = 0
    for idx, f in enumerate(files, 1):
        img = cv2.imread(f)
        res = p.process_frame(img, run_optical_checks=False)
        fname = os.path.basename(f)

        qr_ok = res.qr.status_str == "OK"
        pharma_ok = res.pharma.status_str == "OK"
        bc_ok = res.barcode.status_str == "OK"

        all_ok = qr_ok and pharma_ok and bc_ok
        if all_ok:
            passed_count += 1

        print(f"[{idx:02d}/10] {fname}:")
        print(f"   QR Code  : [{res.qr.status_str}] Payload: {res.qr.payload} (Grade {res.qr.overall_letter})")
        print(f"   Pharma   : [{res.pharma.status_str}] Code: {res.pharma.payload} ({res.pharma.bar_count} bars) bbox={res.pharma.details.get('bbox')}")
        print(f"   Barcode  : [{res.barcode.status_str}] {res.barcode.format}: {res.barcode.payload} [GS1: {res.barcode.gs1_origin}]")
        print(f"   OCR      : [{res.ocr.status_str}] Lines: {len(res.ocr.raw_lines)}")
        print("-" * 90)

    print(f"\nFINAL ACCURACY: {passed_count}/{len(files)} ({passed_count/len(files)*100:.1f}%) SUCCESS RATE")
    print("=" * 90)


if __name__ == "__main__":
    import sys
    if "--test-only" in sys.argv:
        benchmark_cartons()
    else:
        generate_carton_dataset(count=10)
        benchmark_cartons()
