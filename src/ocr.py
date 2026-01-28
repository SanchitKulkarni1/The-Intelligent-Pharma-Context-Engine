# src/ocr.py
from paddleocr import PaddleOCR
import logging
import numpy as np
from typing import Dict

logging.getLogger("ppocr").setLevel(logging.ERROR)

ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    det_db_thresh=0.1,
    det_db_box_thresh=0.3,
    show_log=False
)


def _parse_ocr_result(result) -> Dict:
    tokens = []
    full_text_lines = []

    if result and result[0]:
        for line in result[0]:
            bbox = line[0]           # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = line[1][0]
            confidence = float(line[1][1])

            tokens.append({
                "text": text,
                "confidence": confidence,
                "bbox": [
                    int(bbox[0][0]), int(bbox[0][1]),
                    int(bbox[2][0]), int(bbox[2][1])
                ]
            })

            full_text_lines.append(text)

    return {
        "engine": "PaddleOCR",
        "full_text": " ".join(full_text_lines),
        "tokens": tokens
    }


# -------------------------------------------------
# STAGE-0 COMPATIBLE OCR (IMAGE INPUT)
# -------------------------------------------------

def run_ocr_from_image(image: np.ndarray, use_preprocessing: bool = True) -> Dict:
    """
    Run OCR on an image region (Stage-0 compatible).
    Tries both preprocessed and original images, returns the better result.
    
    Args:
        image: Input image as numpy array
        use_preprocessing: If True, try preprocessing and compare with original
    """
    if image is None:
        return {"engine": "PaddleOCR", "full_text": "", "tokens": []}

    original_result = None
    preprocessed_result = None
    
    # Try original image first
    try:
        result = ocr_engine.ocr(image, cls=True)
        original_result = _parse_ocr_result(result)
    except Exception as e:
        print(f"[WARN] OCR on original failed: {e}")
    
    # Try with preprocessing
    if use_preprocessing:
        try:
            from src.vision.preprocessing import preprocess_for_ocr
            processed = preprocess_for_ocr(image)
            result = ocr_engine.ocr(processed, cls=True)
            preprocessed_result = _parse_ocr_result(result)
        except Exception as e:
            print(f"[WARN] Preprocessing failed: {e}")

    # Compare results and return the better one
    orig_tokens = len(original_result["tokens"]) if original_result else 0
    prep_tokens = len(preprocessed_result["tokens"]) if preprocessed_result else 0
    
    if prep_tokens > orig_tokens:
        print(f"[INFO] Using preprocessed result ({prep_tokens} tokens vs {orig_tokens})")
        return preprocessed_result
    elif orig_tokens > 0:
        print(f"[INFO] Using original result ({orig_tokens} tokens vs {prep_tokens})")
        return original_result
    elif preprocessed_result:
        return preprocessed_result
    elif original_result:
        return original_result
    else:
        return {"engine": "PaddleOCR", "full_text": "", "tokens": []}


# -------------------------------------------------
# LEGACY WRAPPER (OPTIONAL)
# -------------------------------------------------

def run_ocr(image_path: str) -> Dict:
    """
    Backward-compatible wrapper for file paths.
    """
    result = ocr_engine.ocr(image_path, cls=True)
    return _parse_ocr_result(result)
