import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class VisionRegions:
    label_image: np.ndarray | None
    barcode_image: np.ndarray | None
    pill_images: list


def detect_regions(image_path: str) -> VisionRegions:
    """
    Stage-0: Locate label and barcode regions.
    Heuristic-based (no ML) — swappable later.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not readable")

    h, w, _ = img.shape

    # --- Heuristic assumptions (valid for pharma bottles) ---
    # Barcode is usually bottom third
    barcode_region = img[int(h * 0.65):h, 0:w]

    # Label text is usually center
    label_region = img[int(h * 0.15):int(h * 0.65), int(w * 0.05):int(w * 0.95)]
    print(f"Detected label region shape: {label_region.shape}")
    print(f"Detected barcode region shape: {barcode_region.shape}")
    return VisionRegions(
        label_image=label_region,
        barcode_image=barcode_region,
        pill_images=[]  # leave empty for now
    )
