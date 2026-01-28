#!/usr/bin/env python3
"""
Evaluation Script for The Intelligent Pharma-Context Engine
Calculates Character Error Rate (CER) and Entity Match Rate.

CER = (S + D + I) / N
Where:
  S = Substitutions
  D = Deletions
  I = Insertions
  N = Total characters in ground truth
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from rapidfuzz import distance as rd


def calculate_cer(predicted: str, ground_truth: str) -> Tuple[float, Dict]:
    """
    Calculate Character Error Rate between predicted and ground truth strings.

    Returns:
        Tuple of (CER value, detailed breakdown dict)
    """
    if not ground_truth:
        return 0.0 if not predicted else 1.0, {"S": 0, "D": 0, "I": len(predicted), "N": 0}

    # Use Levenshtein distance components
    ops = rd.Levenshtein.editops(predicted, ground_truth)

    substitutions = sum(1 for op in ops if op.tag == "replace")
    deletions = sum(1 for op in ops if op.tag == "delete")
    insertions = sum(1 for op in ops if op.tag == "insert")

    n = len(ground_truth)
    cer = (substitutions + deletions + insertions) / n if n > 0 else 0.0

    return cer, {
        "S": substitutions,
        "D": deletions,
        "I": insertions,
        "N": n,
        "predicted_len": len(predicted),
        "ground_truth_len": n
    }


def calculate_entity_match_rate(
    predicted_entities: Dict,
    ground_truth_entities: Dict,
    fuzzy_threshold: float = 0.85
) -> Tuple[float, Dict]:
    """
    Calculate Entity Match Rate.

    An entity is considered matched if:
    - Exact match, OR
    - Fuzzy similarity >= threshold

    Returns:
        Tuple of (match_rate, detailed results dict)
    """
    entity_fields = ["drug_name", "manufacturer", "dosage", "composition"]
    
    matches = 0
    total = 0
    details = {}

    for field in entity_fields:
        gt_value = ground_truth_entities.get(field)
        pred_value = predicted_entities.get(field)

        if gt_value is None:
            continue  # Skip fields not in ground truth

        total += 1
        
        if pred_value is None:
            details[field] = {"status": "MISSED", "gt": gt_value, "pred": None}
            continue

        # Handle list values (like composition)
        if isinstance(gt_value, list):
            gt_set = set(str(v).lower() for v in gt_value)
            pred_set = set(str(v).lower() for v in (pred_value if isinstance(pred_value, list) else [pred_value]))
            overlap = len(gt_set & pred_set) / len(gt_set) if gt_set else 0
            if overlap >= fuzzy_threshold:
                matches += 1
                details[field] = {"status": "MATCH", "gt": gt_value, "pred": pred_value, "overlap": overlap}
            else:
                details[field] = {"status": "MISMATCH", "gt": gt_value, "pred": pred_value, "overlap": overlap}
        else:
            # String comparison
            gt_str = str(gt_value).lower().strip()
            pred_str = str(pred_value).lower().strip()
            
            if gt_str == pred_str:
                matches += 1
                details[field] = {"status": "EXACT_MATCH", "gt": gt_value, "pred": pred_value}
            else:
                # Fuzzy match
                similarity = rd.Levenshtein.normalized_similarity(pred_str, gt_str)
                if similarity >= fuzzy_threshold:
                    matches += 1
                    details[field] = {"status": "FUZZY_MATCH", "gt": gt_value, "pred": pred_value, "similarity": similarity}
                else:
                    details[field] = {"status": "MISMATCH", "gt": gt_value, "pred": pred_value, "similarity": similarity}

    match_rate = matches / total if total > 0 else 0.0
    return match_rate, details


def evaluate_single(prediction_json: Dict, ground_truth_json: Dict) -> Dict:
    """
    Evaluate a single prediction against ground truth.
    """
    results = {}

    # CER on full OCR text
    pred_text = ""
    gt_text = ground_truth_json.get("raw_ocr", {}).get("full_text", "")
    
    if prediction_json.get("raw_ocr"):
        pred_text = prediction_json["raw_ocr"].get("full_text", "")
    
    cer, cer_details = calculate_cer(pred_text, gt_text)
    results["cer"] = {"value": round(cer, 4), "details": cer_details}

    # Entity Match Rate
    pred_entities = {}
    gt_entities = ground_truth_json.get("extracted_entities", {})
    
    if prediction_json.get("extracted_entities"):
        for field in ["drug_name", "manufacturer", "dosage", "composition"]:
            entity = prediction_json["extracted_entities"].get(field)
            if entity:
                pred_entities[field] = entity.get("value") if isinstance(entity, dict) else entity

    for field in ["drug_name", "manufacturer", "dosage", "composition"]:
        entity = gt_entities.get(field)
        if entity:
            gt_entities[field] = entity.get("value") if isinstance(entity, dict) else entity

    emr, emr_details = calculate_entity_match_rate(pred_entities, gt_entities)
    results["entity_match_rate"] = {"value": round(emr, 4), "details": emr_details}

    # Verification accuracy
    pred_match = prediction_json.get("verification", {}).get("matched_term")
    gt_match = ground_truth_json.get("verification", {}).get("matched_term")
    
    if gt_match:
        if pred_match:
            similarity = rd.Levenshtein.normalized_similarity(
                str(pred_match).lower(), 
                str(gt_match).lower()
            )
            results["verification_accuracy"] = {
                "matched": similarity >= 0.85,
                "similarity": round(similarity, 4),
                "predicted": pred_match,
                "ground_truth": gt_match
            }
        else:
            results["verification_accuracy"] = {
                "matched": False,
                "similarity": 0.0,
                "predicted": None,
                "ground_truth": gt_match
            }

    return results


def evaluate_batch(predictions_dir: Path, ground_truth_file: Path) -> Dict:
    """
    Evaluate multiple predictions against a ground truth file.

    Expected ground truth format:
    {
        "image_name.jpeg": {
            "raw_ocr": {"full_text": "..."},
            "extracted_entities": {...},
            "verification": {...}
        },
        ...
    }
    """
    with open(ground_truth_file, "r") as f:
        ground_truths = json.load(f)

    all_results = {}
    total_cer = 0.0
    total_emr = 0.0
    count = 0

    for gt_name, gt_data in ground_truths.items():
        pred_file = predictions_dir / f"{Path(gt_name).stem}_prediction.json"
        
        if not pred_file.exists():
            print(f"[WARN] Prediction not found for {gt_name}")
            continue

        with open(pred_file, "r") as f:
            prediction = json.load(f)

        result = evaluate_single(prediction, gt_data)
        all_results[gt_name] = result
        
        total_cer += result["cer"]["value"]
        total_emr += result["entity_match_rate"]["value"]
        count += 1

    summary = {
        "total_samples": count,
        "average_cer": round(total_cer / count, 4) if count > 0 else 0.0,
        "average_entity_match_rate": round(total_emr / count, 4) if count > 0 else 0.0,
        "individual_results": all_results
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Pharma-Context Engine predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single prediction
  python evaluation.py --prediction output.json --ground-truth gt.json

  # Evaluate batch of predictions
  python evaluation.py --predictions-dir ./outputs --ground-truth-file gt_all.json
        """
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Evaluation mode")

    # Single evaluation
    single_parser = subparsers.add_parser("single", help="Evaluate single prediction")
    single_parser.add_argument("--prediction", "-p", required=True, help="Path to prediction JSON")
    single_parser.add_argument("--ground-truth", "-g", required=True, help="Path to ground truth JSON")

    # Batch evaluation
    batch_parser = subparsers.add_parser("batch", help="Evaluate batch of predictions")
    batch_parser.add_argument("--predictions-dir", "-p", required=True, help="Directory containing prediction JSONs")
    batch_parser.add_argument("--ground-truth-file", "-g", required=True, help="Path to ground truth file")

    args = parser.parse_args()

    if args.mode == "single":
        with open(args.prediction, "r") as f:
            prediction = json.load(f)
        with open(args.ground_truth, "r") as f:
            ground_truth = json.load(f)
        
        results = evaluate_single(prediction, ground_truth)
        print(json.dumps(results, indent=2))

    elif args.mode == "batch":
        results = evaluate_batch(Path(args.predictions_dir), Path(args.ground_truth_file))
        print(json.dumps(results, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
