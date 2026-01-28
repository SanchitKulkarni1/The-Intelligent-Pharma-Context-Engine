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
    def __init__(self, model_path="best.pt"):
        # Check if model exists first
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}! Did you move the runs folder?")
            
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)

    def analyze(self, image_path: str) -> Detections:
        # 1. Run Inference
        results = self.model(image_path, conf=0.15)[0]  # Lower threshold to capture more detections
        original_img = results.orig_img
        
        label_crops = []
        barcode_crops = []

        # 2. Extract All Crops
        for box in results.boxes:
            # Get coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class name (the one you trained: 'label_roi' or 'barcode')
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])
            
            # Crop the image
            crop = original_img[y1:y2, x1:x2]
            area = (x2 - x1) * (y2 - y1)

            if cls_name == 'label_roi' or cls_name == 'text_block':
                label_crops.append((crop, area, conf))
            elif cls_name == 'barcode':
                barcode_crops.append((crop, area, conf))

        # 3. Select the largest crop for each class
        label_part = None
        barcode_part = None
        
        if label_crops:
            # Sort by area descending, pick largest
            label_crops.sort(key=lambda x: x[1], reverse=True)
            label_part = label_crops[0][0]
            print(f"✅ Selected Label Region: {label_part.shape} (area={label_crops[0][1]}, conf={label_crops[0][2]:.2f})")
        
        if barcode_crops:
            # Sort by area descending, pick largest
            barcode_crops.sort(key=lambda x: x[1], reverse=True)
            barcode_part = barcode_crops[0][0]
            print(f"✅ Selected Barcode: {barcode_part.shape} (area={barcode_crops[0][1]}, conf={barcode_crops[0][2]:.2f})")

        # 4. Return results + debug image with boxes drawn
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