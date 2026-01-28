# src/vision/preprocessing.py
"""
Image preprocessing module for handling challenging pharmaceutical images.
Addresses: blur, glare, low contrast, rotation, and curved surfaces.
"""

import cv2
import numpy as np
from typing import Tuple


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Apply a series of preprocessing steps to improve OCR accuracy.
    
    Handles:
    - Blur/out-of-focus images
    - Specular glare from plastic/foil
    - Low contrast
    - Noise
    """
    if image is None or image.size == 0:
        return image
    
    # 1. Upscale small images (helps with tiny text)
    h, w = image.shape[:2]
    if h < 200 or w < 200:
        scale = max(200 / h, 200 / w, 2.0)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # 2. Remove glare first (on color image)
    deglared = remove_glare(image)
    
    # 3. Convert to grayscale
    if len(deglared.shape) == 3:
        gray = cv2.cvtColor(deglared, cv2.COLOR_BGR2GRAY)
    else:
        gray = deglared.copy()
    
    # 4. Aggressive denoising for blur
    denoised = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)
    
    # 5. Adaptive histogram equalization (CLAHE) for contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # 6. Strong sharpening to counteract blur
    sharpened = sharpen_image(enhanced, strength=2.0)
    
    # 7. Bilateral filter to smooth while keeping edges
    smoothed = cv2.bilateralFilter(sharpened, 9, 75, 75)
    
    # 8. Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, 4
    )
    
    # 9. Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Convert back to BGR for PaddleOCR compatibility
    result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    
    return result


def sharpen_image(image: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """Apply unsharp masking to sharpen blurry images."""
    # Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    # Unsharp mask with configurable strength
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def remove_glare(image: np.ndarray) -> np.ndarray:
    """
    Reduce specular highlights (glare) from reflective surfaces.
    Uses inpainting on overexposed regions.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Detect overexposed (glare) regions
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # Dilate the mask slightly
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Inpaint the glare regions
    result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return result


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Correct rotation/skew in the image.
    Useful for tilted labels.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=50, maxLineGap=10)
    
    if lines is None or len(lines) == 0:
        return image
    
    # Calculate average angle
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 45:  # Only consider near-horizontal lines
            angles.append(angle)
    
    if not angles:
        return image
    
    median_angle = np.median(angles)
    
    # Rotate to correct skew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), 
                              flags=cv2.INTER_CUBIC, 
                              borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def enhance_for_barcode(image: np.ndarray) -> np.ndarray:
    """
    Preprocessing specifically for barcode detection.
    Focuses on contrast and edge clarity.
    """
    if image is None or image.size == 0:
        return image
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # High contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Sharpen
    sharpened = sharpen_image(enhanced)
    
    # Otsu's thresholding for clean barcode lines
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def full_preprocess_pipeline(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline returning both OCR-optimized and barcode-optimized versions.
    
    Returns:
        Tuple of (ocr_image, barcode_image)
    """
    # Remove glare first (works on color image)
    deglared = remove_glare(image)
    
    # Deskew
    deskewed = deskew_image(deglared)
    
    # Generate optimized versions
    ocr_optimized = preprocess_for_ocr(deskewed)
    barcode_optimized = enhance_for_barcode(deskewed)
    
    return ocr_optimized, barcode_optimized
