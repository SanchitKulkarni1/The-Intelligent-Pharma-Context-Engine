from ultralytics import YOLO
import cv2
import numpy as np
import os
from dataclasses import dataclass

@dataclass
class Detections:
    label_crop: np.ndarray | None
    barcode_crop: np.ndarray | None
    debug_image: np.ndarray     # Image with boxes drawn on it

class MedicineDetector:
    def __init__(self, model_path="runs/detect/train/weights/best.pt"):
        # Check if model exists first
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}! Did you move the runs folder?")
            
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)

    def analyze(self, image_path: str) -> Detections:
        # 1. Run Inference
        results = self.model(image_path, conf=0.01)[0]
        original_img = results.orig_img
        
        label_part = None
        barcode_part = None

        # 2. Extract Crops
        for box in results.boxes:
            # Get coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class name (the one you trained: 'label_roi' or 'barcode')
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            
            # Crop the image
            crop = original_img[y1:y2, x1:x2]

            if cls_name == 'label_roi' or cls_name == 'text_block': # Handle whatever you named it
                label_part = crop
                print(f"✅ Found Label Region: {crop.shape}")
            elif cls_name == 'barcode':
                barcode_part = crop
                print(f"✅ Found Barcode: {crop.shape}")

        # 3. Return results + debug image with boxes drawn
        return Detections(
            label_crop=label_part,
            barcode_crop=barcode_part,
            debug_image=results.plot() # This draws the boxes for you
        )

# --- Quick Test ---
if __name__ == "__main__":
    # Point this to a real image in your folder
    test_image = "medicine-bottle-1/test/images/134_png.rf.9eecd904f90506ce5378d896f4e066da.jpg" 
    
    if os.path.exists(test_image):
        detector = MedicineDetector()
        result = detector.analyze(test_image)
        
        # Save results to verify
        if result.label_crop is not None:
            cv2.imwrite("output_label.jpg", result.label_crop)
        if result.debug_image is not None:
            cv2.imwrite("output_debug.jpg", result.debug_image)
            
        print("Test complete! Check 'output_debug.jpg' to see your model in action.")
    else:
        print(f"Please add a test image at {test_image}")