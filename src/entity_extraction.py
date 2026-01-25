import re
from schema import ExtractedEntities, ExtractedEntity

DOSAGE_PATTERN = re.compile(r"\b\d+\s?(mg|ml|mcg|g)\b", re.IGNORECASE)

def extract_entities(doc):
    text = doc.raw_ocr.full_text

    dosage_match = DOSAGE_PATTERN.search(text)
    dosage = dosage_match.group() if dosage_match else None

    tokens = text.split()
    drug_candidate = tokens[0] if tokens else None

    doc.extracted_entities = ExtractedEntities(
        drug_name=ExtractedEntity(
            value=drug_candidate,
            confidence=0.65
        ) if drug_candidate else None,

        dosage=ExtractedEntity(
            value=dosage,
            confidence=0.9
        ) if dosage else None
    )

    return doc
