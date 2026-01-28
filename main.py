#!/usr/bin/env python3
"""
The Intelligent Pharma-Context Engine
End-to-end pipeline: Image → Detection → OCR → Entity Extraction → Verification → Enrichment → JSON Output
"""

import sys
import json
import argparse
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
    """Initialize a new PharmaDocument with unique ID and timestamp."""
    return PharmaDocument(
        document_id=str(uuid4()),
        timestamp_utc=datetime.now(timezone.utc)
    )


def ocr_stage_from_crop(doc: PharmaDocument, label_crop) -> PharmaDocument:
    """Run OCR on the label crop and populate the document."""
    if label_crop is None:
        print("[WARN] No label crop provided, skipping OCR.")
        doc.raw_ocr = RawOCR(engine="PaddleOCR", full_text="", tokens=[])
        return doc

    ocr_result = run_ocr_from_image(label_crop)

    if not ocr_result or not ocr_result.get("tokens"):
        print("[WARN] OCR returned no tokens.")
        doc.raw_ocr = RawOCR(
            engine=ocr_result.get("engine", "PaddleOCR"),
            full_text=ocr_result.get("full_text", ""),
            tokens=[]
        )
        return doc

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


def run_pipeline(image_path: str, output_json: bool = True) -> PharmaDocument:
    """
    Execute the full pipeline on a single image.

    Args:
        image_path: Path to the medicine bottle/strip image.
        output_json: If True, print final JSON to stdout.

    Returns:
        The enriched PharmaDocument.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print('='*60)

    doc = build_document()

    # -----------------------------
    # STAGE 0: VISION (REGIONS)
    # -----------------------------
    print("\n[STAGE 0] Detecting regions with YOLO...")
    regions = detect_regions(image_path)

    # -----------------------------
    # BARCODE (HIGHEST TRUST)
    # -----------------------------
    print("\n[STAGE 1] Detecting barcode...")
    doc.barcode = detect_barcode_from_image(regions.barcode_image)
    if doc.barcode:
        print(f"  ✓ Barcode found: {doc.barcode.value} ({doc.barcode.symbology})")
    else:
        print("  ✗ No barcode detected.")

    # -----------------------------
    # OCR ON LABEL REGION
    # -----------------------------
    print("\n[STAGE 2] Running OCR on label region...")
    doc = ocr_stage_from_crop(doc, regions.label_image)
    print(f"  Full text: {doc.raw_ocr.full_text[:200]}..." if len(doc.raw_ocr.full_text) > 200 else f"  Full text: {doc.raw_ocr.full_text}")

    # -----------------------------
    # NLP: ENTITY EXTRACTION
    # -----------------------------
    print("\n[STAGE 3] Extracting entities...")
    doc = extract_entities(doc)
    if doc.extracted_entities:
        if doc.extracted_entities.drug_name:
            print(f"  Drug Name: {doc.extracted_entities.drug_name.value}")
        if doc.extracted_entities.dosage:
            print(f"  Dosage: {doc.extracted_entities.dosage.value}")
        if doc.extracted_entities.manufacturer:
            print(f"  Manufacturer: {doc.extracted_entities.manufacturer.value}")
        if doc.extracted_entities.composition:
            print(f"  Composition: {doc.extracted_entities.composition.value}")

    # -----------------------------
    # VERIFICATION (RXNORM)
    # -----------------------------
    print("\n[STAGE 4] Verifying against RxNorm...")
    doc = verify_drug(doc)
    if doc.verification and doc.verification.matched_term:
        print(f"  ✓ Matched: {doc.verification.matched_term} (score: {doc.verification.match_score})")
        print(f"  RxNorm CUI: {doc.verification.rxnorm_cui}")
    else:
        print("  ✗ No RxNorm match found.")

    # -----------------------------
    # ENRICHMENT (FDA)
    # -----------------------------
    print("\n[STAGE 5] Enriching with FDA data...")
    if doc.verification and doc.verification.rxnorm_cui:
        ingredients = extract_ingredients_from_rxnorm(doc.verification.rxnorm_cui)
        doc.enrichment = enrich_with_fda(ingredients)
        if doc.enrichment:
            print(f"  Storage: {doc.enrichment.storage_requirements}")
            print(f"  Side Effects: {len(doc.enrichment.common_side_effects)} found")
            print(f"  Warnings: {len(doc.enrichment.safety_warnings)} found")
    else:
        print("  ✗ Skipped (no verified drug).")

    # -----------------------------
    # OUTPUT
    # -----------------------------
    if output_json:
        print("\n" + "="*60)
        print("FINAL JSON OUTPUT")
        print("="*60)
        print(doc.model_dump_json(indent=2))

    return doc


def main():
    parser = argparse.ArgumentParser(
        description="The Intelligent Pharma-Context Engine",
        epilog="Example: python main.py images/barcode1.jpeg"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="images/barcode1.jpeg",
        help="Path to the medicine image (default: images/barcode1.jpeg)"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Suppress JSON output"
    )

    args = parser.parse_args()

    try:
        doc = run_pipeline(args.image, output_json=not args.no_json)
        return 0
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
