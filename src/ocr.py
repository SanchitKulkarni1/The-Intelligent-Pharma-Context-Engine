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

def run_ocr_from_image(image: np.ndarray) -> Dict:
    """
    Run OCR on an image region (Stage-0 compatible).
    """
    if image is None:
        return {"engine": "PaddleOCR", "full_text": "", "tokens": []}

    result = ocr_engine.ocr(image, cls=True)
    return _parse_ocr_result(result)


# -------------------------------------------------
# LEGACY WRAPPER (OPTIONAL)
# -------------------------------------------------

def run_ocr(image_path: str) -> Dict:
    """
    Backward-compatible wrapper for file paths.
    """
    result = ocr_engine.ocr(image_path, cls=True)
    return _parse_ocr_result(result)
