# The Intelligent Pharma-Context Engine

An end-to-end pipeline for extracting, verifying, and enriching metadata from medicine bottle/strip images.

## 🎯 Overview

This system processes pharmaceutical images through five stages:

1. **Detection** – YOLO-based region detection (label, barcode)
2. **OCR** – Text extraction using PaddleOCR
3. **Entity Extraction** – NLP-based parsing of drug name, dosage, manufacturer, composition
4. **Verification** – Cross-referencing against RxNorm for canonical drug names
5. **Enrichment** – Fetching supplemental data (storage, side effects, warnings) from openFDA

## 🏗️ Architecture

```
                    ┌─────────────────┐
                    │   Input Image   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  YOLO Detector  │  ← best.pt (trained model)
                    │  (Stage 0)      │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼────────┐           ┌────────▼────────┐
     │  Label Region   │           │ Barcode Region  │
     └────────┬────────┘           └────────┬────────┘
              │                             │
     ┌────────▼────────┐           ┌────────▼────────┐
     │   PaddleOCR     │           │   pyzbar        │
     └────────┬────────┘           └────────┬────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │ Entity Extract  │
                    │ (Regex + NLP)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   RxNorm API    │  ← Fuzzy matching + LLM reranker
                    │  (Verification) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   openFDA API   │
                    │  (Enrichment)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   JSON Output   │
                    └─────────────────┘
```

## 🧠 Model Details

| Component | Technology | Notes |
|-----------|------------|-------|
| **Detection** | YOLOv8 (`best.pt`) | Custom-trained for label_roi and barcode detection |
| **OCR** | PaddleOCR | Angle-aware, optimized for pharmacy labels |
| **Fuzzy Matching** | RapidFuzz | Token-set ratio for drug name matching |
| **LLM Reranker** | Gemini Flash | Optional, invoked only for ambiguous cases |
| **Barcode Decode** | pyzbar | Supports 1D barcodes and DataMatrix |

## 📦 Installation

```bash
# Clone the repository
git clone <repo_url>
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

Create a `.env` file for optional LLM reranking:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

## 🚀 Usage

### Basic Usage

```bash
python main.py images/barcode1.jpeg
```

### Command Line Options

```bash
python main.py --help

# Suppress JSON output
python main.py images/testimage.jpeg --no-json
```

### Programmatic Usage

```python
from main import run_pipeline

doc = run_pipeline("path/to/image.jpeg", output_json=False)
print(doc.verification.matched_term)
print(doc.enrichment.safety_warnings)
```

## 📊 Evaluation

The `evaluation.py` script calculates metrics as per the submission requirements:

### Character Error Rate (CER)

```
CER = (S + D + I) / N
```
- S = Substitutions
- D = Deletions
- I = Insertions
- N = Total characters in ground truth

### Entity Match Rate

Percentage of correctly extracted entities (drug_name, manufacturer, dosage, composition) with fuzzy matching threshold of 85%.

### Running Evaluation

```bash
# Single prediction
python evaluation.py single -p prediction.json -g ground_truth.json

# Batch evaluation
python evaluation.py batch -p ./outputs -g ground_truths.json
```

## 🔧 Handling Real-World Challenges

### Physical Distortions (Curved/Reflective Surfaces)

- **YOLO Detection**: Trained on diverse images including curved bottles
- **Fallback Heuristics**: If YOLO fails, geometric cropping is applied
- **PaddleOCR**: Uses angle classification to handle rotated text

### Fuzzy Entity Resolution

```python
# Example: OCR reads "Lisinopri1" instead of "Lisinopril"
# RapidFuzz token_set_ratio handles this:
from rapidfuzz import fuzz
fuzz.token_set_ratio("Lisinopri1", "Lisinopril")  # → 95
```

### Layout Agnosticism

- Entity extraction uses regex patterns, not fixed coordinates
- Drug names inferred from capitalization and word position
- Manufacturer detected via keyword matching ("Pharma", "Labs", etc.)

### Multi-Modal Validation

1. **Barcode Priority**: If NDC barcode is detected, it overrides OCR-based matching
2. **Confidence Scoring**: Each extraction includes confidence values
3. **LLM Fallback**: For ambiguous cases, Gemini provides reasoning

## 📁 Project Structure

```
The-Intelligent-Pharma-Context-Engine/
├── main.py                 # Entry point
├── schema.py               # Pydantic models
├── evaluation.py           # CER & Entity Match Rate calculator
├── best.pt                 # Trained YOLO model
├── requirements.txt        # Dependencies
├── images/                 # Test images
│   ├── barcode1.jpeg
│   └── testimage.jpeg
└── src/
    ├── vision/
    │   ├── stage0.py       # YOLO + fallback region detection
    │   └── detector.py     # MedicineDetector class
    ├── ocr.py              # PaddleOCR wrapper
    ├── barcode.py          # pyzbar wrapper
    ├── entity_extraction.py# NLP entity parsing
    ├── verification.py     # RxNorm verification
    ├── enrichment.py       # openFDA enrichment
    └── utils/
        ├── ingredients.py  # RxNorm ingredient lookup
        ├── ndc.py          # NDC to drug lookup
        └── reranker.py     # Gemini LLM reranker
```

## 📄 Output Schema

```json
{
  "document_id": "uuid",
  "timestamp_utc": "2024-01-29T00:00:00Z",
  "barcode": {
    "value": "0363600231",
    "symbology": "CODE128",
    "confidence": 1.0
  },
  "raw_ocr": {
    "engine": "PaddleOCR",
    "full_text": "...",
    "tokens": [...]
  },
  "extracted_entities": {
    "drug_name": {"value": "Lisinopril", "confidence": 0.7},
    "dosage": {"value": "10 mg", "confidence": 0.9},
    "manufacturer": {"value": "Sun Pharma", "confidence": 0.75},
    "composition": {"value": ["lisinopril"], "confidence": 0.85}
  },
  "verification": {
    "rxnorm_cui": "197884",
    "matched_term": "lisinopril 10 MG Oral Tablet",
    "match_score": 0.92,
    "final_canonical_name": "lisinopril 10 MG Oral Tablet"
  },
  "enrichment": {
    "storage_requirements": "Store at 20-25°C",
    "common_side_effects": ["dizziness", "cough"],
    "safety_warnings": ["Do not use if pregnant"]
  }
}
```

## 📜 License

MIT License