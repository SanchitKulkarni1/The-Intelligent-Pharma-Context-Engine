# 🧪 The Intelligent Pharma-Context Engine

An end-to-end pipeline for extracting, verifying, and enriching metadata from pharmaceutical packaging images (medicine bottles, blister strips, ampules).

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)
![PaddleOCR](https://img.shields.io/badge/OCR-PaddleOCR-red.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Technical Stack](#-technical-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Pipeline Stages](#-pipeline-stages)
- [Handling Real-World Challenges](#-handling-real-world-challenges)
- [Performance Report](#-performance-report)
- [Output Schema](#-output-schema)
- [Project Structure](#-project-structure)

---

## 🎯 Overview

This system processes pharmaceutical images through five stages:

1. **Detection** – YOLOv8-based region detection (label, barcode)
2. **OCR** – Text extraction using PaddleOCR with preprocessing
3. **Entity Extraction** – NLP-based parsing of drug name, dosage, manufacturer, composition
4. **Verification** – Cross-referencing against RxNorm with fuzzy matching
5. **Enrichment** – Fetching supplemental data (storage, side effects, warnings) from openFDA

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT IMAGE                                     │
│                    (Medicine Bottle / Blister Strip)                         │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 0: DETECTION                                   │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │   YOLOv8        │    │  Heuristic      │    │  Minimum Area           │  │
│  │   (best.pt)     │───▶│  Fallback       │───▶│  Threshold (5000px)     │  │
│  │   conf=0.15     │    │  (if YOLO fails)│    │  (reject tiny crops)    │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│      LABEL REGION                 │   │      BARCODE REGION                │
└───────────────────┬───────────────┘   └───────────────────┬───────────────┘
                    │                                       │
                    ▼                                       ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│   STAGE 1: IMAGE PREPROCESSING    │   │   BARCODE PREPROCESSING            │
│   ┌─────────────────────────────┐ │   │   ┌─────────────────────────────┐ │
│   │ • Glare Removal (Inpaint)   │ │   │   │ • CLAHE Contrast            │ │
│   │ • Upscaling (<200px)        │ │   │   │ • Otsu Thresholding         │ │
│   │ • Denoising (NLMeans)       │ │   │   │ • High Sharpening           │ │
│   │ • CLAHE Contrast            │ │   │   └─────────────────────────────┘ │
│   │ • Sharpening (Unsharp Mask) │ │   └───────────────────┬───────────────┘
│   │ • Deskewing                 │ │                       │
│   └─────────────────────────────┘ │                       ▼
└───────────────────┬───────────────┘   ┌───────────────────────────────────┐
                    │                   │   STAGE 2: BARCODE DECODE          │
                    ▼                   │   (pyzbar - 1D/DataMatrix)         │
┌───────────────────────────────────┐   │   ──────────────────────────────── │
│   STAGE 3: OCR (PaddleOCR)        │   │   If NDC found → Override OCR     │
│   ┌─────────────────────────────┐ │   └───────────────────┬───────────────┘
│   │ Try preprocessed + original │ │                       │
│   │ Return MORE tokens          │ │                       │
│   └─────────────────────────────┘ │                       │
└───────────────────┬───────────────┘                       │
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 4: ENTITY EXTRACTION                              │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │ • Drug Name    → First capitalized word (heuristic)                   │ │
│   │ • Dosage       → Regex: \d+\s?(mg|ml|mcg|g|iu)                        │ │
│   │ • Manufacturer → Keywords: "Pharma", "Labs", "Inc"                    │ │
│   │ • Composition  → Known ingredients list (50+ drugs)                   │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 5: VERIFICATION (RxNorm)                          │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │ 1. NDC Barcode Override (if present) → 100% confidence                │ │
│   │ 2. Exact RxNorm Query                                                 │ │
│   │ 3. Approximate Match (handles OCR typos like "Lisinopri1")            │ │
│   │ 4. Fuzzy Scoring (RapidFuzz token_set_ratio)                          │ │
│   │ 5. LLM Re-ranker (Gemini Flash) for ambiguous cases                   │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 6: ENRICHMENT (openFDA)                           │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │ • Storage Requirements                                                │ │
│   │ • Common Side Effects                                                 │ │
│   │ • Safety Warnings                                                     │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ENRICHED JSON OUTPUT                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Technical Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Object Detection** | YOLOv8 (`best.pt`) | Fast inference, custom-trained for pharma labels |
| **OCR Engine** | PaddleOCR | Angle-aware, high accuracy on pharmacy text, supports angle classification |
| **Fuzzy Matching** | RapidFuzz | `token_set_ratio` handles word reordering and partial matches |
| **LLM Re-ranking** | Gemini Flash | Resolves ambiguous candidates using clinical reasoning |
| **Barcode Decoding** | pyzbar | Supports 1D barcodes and DataMatrix |
| **Drug Database** | RxNorm API | Authoritative normalized drug names and CUIs |
| **Enrichment** | openFDA Label API | Complete drug labeling information |
| **Image Preprocessing** | OpenCV | Denoising, sharpening, CLAHE, glare removal |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/SanchitKulkarni1/The-Intelligent-Pharma-Context-Engine.git
cd The-Intelligent-Pharma-Context-Engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# System dependency for pyzbar (Linux)
sudo apt-get install libzbar0
```

### Environment Variables

Create a `.env` file for the optional LLM re-ranker:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

---

## 🚀 Usage

### Basic Usage

```bash
python main.py path/to/image.jpeg
```

### Command Line Options

```bash
python main.py --help

# Process specific image
python main.py images/barcode1.jpeg

# Suppress JSON output (show only stages)
python main.py images/testimage.jpeg --no-json
```

### 🌐 Web UI (Streamlit)

A clean, modern web interface is available:

```bash
streamlit run app.py
```

**Features:**
- 📤 Drag-and-drop image upload
- 🎯 Sample image gallery for quick testing
- 📊 Metrics dashboard (OCR tokens, match score, enrichment status)
- 📑 Tabbed results (OCR, Entities, Verification, Enrichment, JSON)
- 📥 One-click JSON download
- ⚙️ Toggle preprocessing and LLM re-ranking

![Web UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)

### Programmatic Usage

```python
from main import run_pipeline

doc = run_pipeline("path/to/image.jpeg", output_json=False)

# Access results
print(doc.verification.matched_term)       # "lisinopril 10 MG Oral Tablet"
print(doc.verification.rxnorm_cui)          # "197884"
print(doc.enrichment.safety_warnings)       # ["Do not use if pregnant"]
```

---

## 🔬 Pipeline Stages

### Stage 0: Region Detection (YOLO)

- Uses custom-trained YOLOv8 model (`best.pt`)
- Detects `label_roi` and `barcode` classes
- Selects **largest** detection per class (handles multiple overlapping detections)
- Falls back to heuristic cropping if YOLO fails or detections are too small (<5000px)

### Stage 1: Image Preprocessing

Applied to challenging images with blur, glare, or low contrast:

| Technique | Purpose |
|-----------|---------|
| Glare Removal | Inpaints overexposed regions (>240 brightness) |
| Upscaling | Resizes small images (<200px) for better OCR |
| CLAHE | Adaptive contrast enhancement |
| Denoising | Non-local means denoising (h=15) |
| Sharpening | Unsharp mask with strength 2.0 |
| Deskewing | Hough line-based rotation correction |

**Smart Strategy**: Tries both preprocessed AND original images, returns whichever yields more OCR tokens.

### Stage 2: Text Extraction (OCR)

- PaddleOCR with angle classification enabled
- Outputs tokens with bounding boxes and confidence scores
- Full text reconstruction for entity extraction

### Stage 3: Entity Extraction

| Entity | Method |
|--------|--------|
| Drug Name | First capitalized word (heuristic) |
| Dosage | Regex: `\d+(\.\d+)?\s?(mg|ml|mcg|g|iu)` |
| Manufacturer | Keyword matching: "Pharma", "Labs", "Inc", etc. |
| Composition | Pattern matching against 50+ known active ingredients |

### Stage 4: Verification (RxNorm)

1. **Barcode Override**: If NDC detected → direct FDA lookup (100% confidence)
2. **Exact Query**: RxNorm drugs endpoint
3. **Approximate Match**: `approximateTerm` API for OCR typos (e.g., "Lisinopri1" → "Lisinopril")
4. **Fuzzy Scoring**: RapidFuzz `token_set_ratio` ≥ 50%
5. **LLM Re-ranking**: Gemini Flash for ambiguous candidates (score difference < 15%)

### Stage 5: Enrichment (openFDA)

Once verified, fetches from FDA Label API:
- Storage requirements
- Common side effects (adverse_reactions)
- Safety warnings

---

## 🔧 Handling Real-World Challenges

### 1. Physical Distortions (Curved Bottles, Glare)

```python
# src/vision/preprocessing.py
def remove_glare(image):
    """Inpaint overexposed regions"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
```

### 2. Fuzzy Entity Resolution (OCR Typos)

```python
# Example: OCR reads "Lisinopri1" instead of "Lisinopril"
from rapidfuzz import fuzz
fuzz.token_set_ratio("Lisinopri1", "Lisinopril")  # → 95

# RxNorm approximate match handles this automatically
requests.get("https://rxnav.nlm.nih.gov/REST/approximateTerm.json", 
             params={"term": "Lisinopri1"})
# Returns: "Lisinopril"
```

### 3. Layout Agnosticism

- Entity extraction uses regex patterns, **not fixed coordinates**
- Drug names inferred from capitalization and word position
- Manufacturer detected via semantic keyword matching
- No hard-coded template assumptions

### 4. Multi-Modal Validation

```python
# Barcode NDC override (highest trust)
if barcode_detected:
    drug = lookup_drug_by_ndc(barcode.value)
    confidence = 1.0  # Authoritative identifier
else:
    # Fall back to OCR + fuzzy matching
    confidence = fuzzy_score / 100
```

---

## 📊 Performance Report

### Test Results

| Image | OCR Text | Drug Matched | CUI | Confidence |
|-------|----------|--------------|-----|------------|
| `barcode1.jpeg` | DILTIAZEM HCI 30 mg TABLET | diltiazem hydrochloride 30 MG Oral Tablet [Cardizem] | 833219 | 0.95 |
| `testimage.jpeg` | HYDROCODONE/APAP QTY:30 | acetaminophen 325 MG / hydrocodone bitartrate 5 MG Oral Tablet | 1492673 | 0.95 |
| `test2.jpeg` | MORPHIN Sulfate | Morphine Sulfate | 30236 | 0.95 |
| `blur_image.jpg` | *(unreadable)* | — | — | — |

### Character Error Rate (CER)

```
CER = (S + D + I) / N
Where: S = Substitutions, D = Deletions, I = Insertions, N = Total characters
```

| Image | Ground Truth | Predicted | CER |
|-------|--------------|-----------|-----|
| `barcode1.jpeg` | DILTIAZEM HCI 30 mg TABLET | DILTIAZEM HCI 30 mg TABLET | **0.00** |
| `test2.jpeg` | MORPHINE Sulfate | MORPHIN Sulfate | **0.06** (1 deletion / 16 chars) |

### Entity Match Rate

| Entity | Matches | Total | Rate |
|--------|---------|-------|------|
| Drug Name | 3 | 3 | **100%** |
| Dosage | 2 | 3 | **67%** |
| Manufacturer | 1 | 2 | **50%** |

### Running Evaluation

```bash
# Single prediction evaluation
python evaluation.py single -p prediction.json -g ground_truth.json

# Batch evaluation
python evaluation.py batch -p ./outputs -g ground_truths.json
```

---

## 📄 Output Schema

```json
{
  "document_id": "uuid",
  "timestamp_utc": "2024-01-29T00:00:00Z",
  "barcode": {
    "value": "5107974501",
    "symbology": "CODE128",
    "confidence": 1.0
  },
  "raw_ocr": {
    "engine": "PaddleOCR",
    "full_text": "DILTIAZEM HCI 30 mg TABLET",
    "tokens": [
      {"text": "DILTIAZEM", "confidence": 0.93, "bbox": [72, 25, 196, 40]},
      {"text": "HCI", "confidence": 0.91, "bbox": [...]},
      {"text": "30 mg", "confidence": 0.94, "bbox": [...]}
    ]
  },
  "extracted_entities": {
    "drug_name": {"value": "DILTIAZEM", "confidence": 0.7, "source": "ocr"},
    "dosage": {"value": "30 mg", "confidence": 0.9, "source": "ocr"},
    "manufacturer": {"value": "UDL", "confidence": 0.75, "source": "ocr"},
    "composition": {"value": ["diltiazem"], "confidence": 0.85, "source": "ocr"}
  },
  "verification": {
    "rxnorm_cui": "833219",
    "matched_term": "diltiazem hydrochloride 30 MG Oral Tablet [Cardizem]",
    "match_score": 0.95,
    "final_canonical_name": "diltiazem hydrochloride 30 MG Oral Tablet [Cardizem]",
    "justification": "This candidate perfectly matches the ingredient, strength, and form."
  },
  "enrichment": {
    "storage_requirements": "Store at 20-25°C (68-77°F)",
    "common_side_effects": ["dizziness", "headache", "edema"],
    "safety_warnings": ["May cause hypotension", "Avoid in heart block"]
  }
}
```

---

## 📁 Project Structure

```
The-Intelligent-Pharma-Context-Engine/
├── main.py                      # CLI entry point
├── app.py                       # Streamlit Web UI
├── evaluation.py                # CER & Entity Match Rate calculator
├── schema.py                    # Pydantic models
├── best.pt                      # Trained YOLOv8 model
├── requirements.txt             # Python dependencies
├── .env                         # API keys (create this)
├── images/                      # Test images
│   ├── barcode1.jpeg
│   ├── testimage.jpeg
│   ├── test2.jpeg
│   └── blur_image.jpg
└── src/
    ├── vision/
    │   ├── detector.py          # YOLOv8 wrapper
    │   ├── stage0.py            # Region detection orchestrator
    │   └── preprocessing.py     # Image enhancement
    ├── ocr.py                   # PaddleOCR wrapper
    ├── barcode.py               # pyzbar wrapper
    ├── entity_extraction.py     # NLP entity parsing
    ├── verification.py          # RxNorm verification
    ├── enrichment.py            # openFDA enrichment
    └── utils/
        ├── ingredients.py       # RxNorm ingredient lookup
        ├── ndc.py               # NDC to drug lookup
        └── reranker.py          # Gemini LLM reranker
```

---

## 📜 License

MIT License

---

## 🙏 Acknowledgments

- **RxNorm** by NIH/NLM for drug nomenclature
- **openFDA** for drug labeling data
- **PaddleOCR** by Baidu for OCR engine
- **Ultralytics** for YOLOv8