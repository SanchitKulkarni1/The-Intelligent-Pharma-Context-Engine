# src/entity_extraction.py
import re
from schema import PharmaDocument, ExtractedEntities, ExtractedEntity
from typing import List, Tuple

DOSAGE_PATTERN = re.compile(r"\b\d+(\.\d+)?\s?(mg|ml|mcg|g|iu)\b", re.IGNORECASE)
COMPOSITION_PATTERN = re.compile(
    r"(acetaminophen|ibuprofen|aspirin|naproxen|lisinopril|metformin|omeprazole|hydrocodone|oxycodone|amoxicillin|azithromycin|cephalexin|ciprofloxacin|metronidazole|prednisone|gabapentin|tramadol|losartan|atorvastatin|simvastatin|levothyroxine|amlodipine|hydrochlorothiazide|furosemide|pantoprazole|cetirizine|diphenhydramine|loratadine|famotidine|ranitidine)",
    re.IGNORECASE
)

# Known pharma company keywords
COMPANY_KEYWORDS = [
    "pharma", "labs", "laboratories", "inc", "corp", "ltd",
    "healthcare", "generics", "biotech", "therapeutics"
]


def _find_dosages(text: str) -> List[str]:
    """Extract all dosage strings."""
    return DOSAGE_PATTERN.findall(text)


def _find_compositions(text: str) -> List[str]:
    """Extract known active ingredients."""
    return list(set(m.group().lower() for m in COMPOSITION_PATTERN.finditer(text)))


def _find_drug_name(tokens: List[str]) -> Tuple[str | None, float]:
    """
    Heuristic: Find the first capitalized token that looks like a drug name.
    Returns (drug_name, confidence).
    """
    for token in tokens:
        # Skip very short or numeric tokens
        if len(token) < 4 or token.isdigit():
            continue
        # Check if it's a known company keyword
        if any(kw in token.lower() for kw in COMPANY_KEYWORDS):
            continue
        # Prefer tokens that start with capital and have mixed case (common for drug names)
        if token[0].isupper():
            return token, 0.7
    # Fallback to the first reasonable token
    if tokens and len(tokens[0]) >= 3:
        return tokens[0], 0.5
    return None, 0.0


def _find_manufacturer(tokens: List[str]) -> Tuple[str | None, float]:
    """
    Look for tokens that contain company keywords.
    """
    for i, token in enumerate(tokens):
        lower = token.lower()
        if any(kw in lower for kw in COMPANY_KEYWORDS):
            # Try to capture multi-word company names (e.g., "Sun Pharma")
            if i > 0:
                return f"{tokens[i-1]} {token}", 0.75
            return token, 0.65
    return None, 0.0


def extract_entities(doc: PharmaDocument) -> PharmaDocument:
    """
    Extract structured entities from raw OCR text.
    """
    if not doc.raw_ocr or not doc.raw_ocr.full_text:
        doc.extracted_entities = ExtractedEntities()
        return doc

    text = doc.raw_ocr.full_text
    tokens = text.split()

    # Dosage
    dosage_matches = DOSAGE_PATTERN.findall(text)
    dosage = None
    if dosage_matches:
        # Re-run to get full match string, not groups
        dosage_match = DOSAGE_PATTERN.search(text)
        dosage = dosage_match.group() if dosage_match else None

    # Composition (active ingredients)
    compositions = _find_compositions(text)

    # Drug name
    drug_name, drug_conf = _find_drug_name(tokens)

    # Manufacturer
    manufacturer, manu_conf = _find_manufacturer(tokens)

    doc.extracted_entities = ExtractedEntities(
        drug_name=ExtractedEntity(
            value=drug_name,
            confidence=drug_conf
        ) if drug_name else None,

        manufacturer=ExtractedEntity(
            value=manufacturer,
            confidence=manu_conf
        ) if manufacturer else None,

        dosage=ExtractedEntity(
            value=dosage,
            confidence=0.9
        ) if dosage else None,

        composition=ExtractedEntity(
            value=compositions,
            confidence=0.85
        ) if compositions else None
    )

    return doc
