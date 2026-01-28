# src/vision/stage0.py
import cv2
import numpy as np
from dataclasses import dataclass
from src.vision.detector import MedicineDetector

@dataclass
class VisionRegions:
    label_image: np.ndarray | None
    barcode_image: np.ndarray | None
    pill_images: list


# Singleton detector instance
_detector: MedicineDetector | None = None

# Minimum area thresholds for usable detections
MIN_LABEL_AREA = 5000  # Minimum pixels for label region
MIN_BARCODE_AREA = 1000  # Minimum pixels for barcode region


def _get_detector() -> MedicineDetector:
    global _detector
    if _detector is None:
        _detector = MedicineDetector()
    return _detector


def detect_regions(image_path: str) -> VisionRegions:
    """
    Stage-0: Locate label and barcode regions using trained YOLO model.
    Falls back to heuristic cropping if the model fails or detections are too small.
    """
    detector = _get_detector()
    label_crop = None
    barcode_crop = None

    try:
        detections = detector.analyze(image_path)
        
        # Check if label detection is large enough
        if detections.label_crop is not None:
            h, w = detections.label_crop.shape[:2]
            if h * w >= MIN_LABEL_AREA:
                label_crop = detections.label_crop
            else:
                print(f"[WARN] Label detection too small ({h}x{w}={h*w} < {MIN_LABEL_AREA}), using fallback")
        
        # Check if barcode detection is large enough
        if detections.barcode_crop is not None:
            h, w = detections.barcode_crop.shape[:2]
            if h * w >= MIN_BARCODE_AREA:
                barcode_crop = detections.barcode_crop
            else:
                print(f"[WARN] Barcode detection too small ({h}x{w}={h*w} < {MIN_BARCODE_AREA}), using fallback")
                
    except Exception as e:
        print(f"[WARN] YOLO detection failed: {e}. Falling back to heuristics.")

    # Fallback to heuristic if YOLO misses regions or they're too small
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not readable")

    h, w = img.shape[:2]

    if label_crop is None:
        # For blister strips/small packages, use most of the image
        label_crop = img[int(h * 0.05):int(h * 0.70), int(w * 0.02):int(w * 0.98)]
        print(f"[FALLBACK] Heuristic label region: {label_crop.shape}")

    if barcode_crop is None:
        barcode_crop = img[int(h * 0.50):h, 0:w]
        print(f"[FALLBACK] Heuristic barcode region: {barcode_crop.shape}")

    return VisionRegions(
        label_image=label_crop,
        barcode_image=barcode_crop,
        pill_images=[]
    )
