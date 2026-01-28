import cv2
from pyzbar.pyzbar import decode
from typing import Optional

from schema import Barcode


def detect_barcode_from_image(image, use_preprocessing: bool = True) -> Optional[Barcode]:
    """
    Decode barcode / DataMatrix from an image region (Stage-0 compatible).
    Applies preprocessing for challenging images (blur, glare, etc.)
    """
    try:
        if image is None:
            return None

        # Try with preprocessing first
        if use_preprocessing:
            try:
                from src.vision.preprocessing import enhance_for_barcode
                processed = enhance_for_barcode(image)
                gray_processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                results = decode(gray_processed)
                
                if results:
                    b = results[0]
                    value = b.data.decode("utf-8").strip()
                    print(f"[STAGE-0] Detected barcode (preprocessed): {value} | type={b.type}")
                    return Barcode(value=value, symbology=b.type, confidence=1.0)
            except Exception as e:
                print(f"[WARN] Barcode preprocessing failed: {e}")

        # Fallback to original image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = decode(gray)

        if not results:
            return None

        b = results[0]
        value = b.data.decode("utf-8").strip()

        print(f"[STAGE-0] Detected barcode: {value} | type={b.type}")

        return Barcode(
            value=value,
            symbology=b.type,
            confidence=1.0
        )

    except Exception as e:
        print("Barcode detection error:", e)
        return None


# -------------------------------------------------
# OPTIONAL: Backward-compatible wrapper
# -------------------------------------------------

def detect_barcode(image_path: str) -> Optional[Barcode]:
    """
    Legacy helper — reads image from path and calls Stage-0 version.
    """
    image = cv2.imread(image_path)
    return detect_barcode_from_image(image)
