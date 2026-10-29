"""
Industrial 1D Retail & Commercial Barcode Detection and Verification Module.
Supports GS1 EAN-13, EAN-8, UPC-A, Code 128, Code 39.
Engineered for ultra-low latency (< 5 ms) and zero false-positive cross-talk with Pharmacode.
"""
from .barcode_engine import Industrial1DBarcodeDetector, BarcodeDetectionResult

__all__ = ["Industrial1DBarcodeDetector", "BarcodeDetectionResult"]
