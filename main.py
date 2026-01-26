from uuid import uuid4
from datetime import datetime, timezone

from schema import PharmaDocument, RawOCR, OCRToken
from src.vision.stage0 import detect_regions
from src.barcode import detect_barcode_from_image
from src.ocr import run_ocr_from_image
from src.entity_extraction import extract_entities
from src.verification import verify_drug
from src.utils.ingredients import extract_ingredients_from_rxnorm
from src.enrichment import enrich_with_fda


def build_document() -> PharmaDocument:
    return PharmaDocument(
        document_id=str(uuid4()),
        timestamp_utc=datetime.now(timezone.utc)
    )


def ocr_stage_from_crop(doc: PharmaDocument, label_crop) -> PharmaDocument:
    ocr_result = run_ocr_from_image(label_crop)

    if not ocr_result or not ocr_result.get("tokens"):
        raise RuntimeError("OCR failed: no text detected")

    doc.raw_ocr = RawOCR(
        engine=ocr_result["engine"],
        full_text=ocr_result["full_text"],
        tokens=[
            OCRToken(
                text=t["text"],
                confidence=t["confidence"],
                bbox=t["bbox"]
            )
            for t in ocr_result["tokens"]
        ]
    )

    return doc


if __name__ == "__main__":
    doc = build_document()

    # -----------------------------
    # STAGE 0: VISION (REGIONS)
    # -----------------------------
    regions = detect_regions("images/barcode1.jpeg")

    # -----------------------------
    # BARCODE (HIGHEST TRUST)
    # -----------------------------
    doc.barcode = detect_barcode_from_image(regions.barcode_image)

    # -----------------------------
    # OCR ONLY ON LABEL REGION
    # -----------------------------
    doc = ocr_stage_from_crop(doc, regions.label_image)

    # -----------------------------
    # NLP PIPELINE
    # -----------------------------
    doc = extract_entities(doc)
    doc = verify_drug(doc)

    ingredients = extract_ingredients_from_rxnorm(
        doc.verification.rxnorm_cui
    )

    doc.enrichment = enrich_with_fda(ingredients)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    print("=== RAW OCR TEXT ===")
    print(doc.raw_ocr.full_text)

    print("\n=== EXTRACTED ENTITIES ===")
    print(doc.extracted_entities)

    print("\n=== VERIFICATION ===")
    print(doc.verification)

    print("\n=== ENRICHMENT ===")
    print(doc.enrichment)
