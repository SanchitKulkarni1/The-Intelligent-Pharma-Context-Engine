# src/ocr.py
from paddleocr import PaddleOCR
import logging

# Turn off the noisy logging
logging.getLogger("ppocr").setLevel(logging.ERROR)

def basic_ocr_test(image_path):
    # Initialize OCR with the sensitive settings we tuned
    ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_thresh=0.1, det_db_box_thresh=0.3, show_log=False)
    
    # Run the OCR
    result = ocr.ocr(image_path, cls=True)
    
    raw_lines = []
    # PaddleOCR returns a nested list. We need to unwrap it.
    if result and result[0]:
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            
            # Print it so we can see it in the terminal (Debug)
            print(f"[{confidence:.2f}] {text}")
            
            # Add it to our list
            raw_lines.append(text)
            
    # --- CRITICAL FIX: RETURN THE LIST! ---
    return raw_lines