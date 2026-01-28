from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class OCRToken(BaseModel):
    text: str
    confidence: float
    bbox: List[int]


class RawOCR(BaseModel):
    engine: str
    full_text: str
    tokens: List[OCRToken]


class ExtractedEntity(BaseModel):
    value: str | List[str]
    confidence: float
    source: str = "ocr"


class ExtractedEntities(BaseModel):
    drug_name: Optional[ExtractedEntity] =None
    manufacturer: Optional[ExtractedEntity] = None
    dosage: Optional[ExtractedEntity] = None
    composition: Optional[ExtractedEntity] = None

class Barcode(BaseModel):
    value: str
    symbology: Optional[str] = None
    confidence: float = 1.0

class Verification(BaseModel):
    rxnorm_cui: Optional[str]
    matched_term: Optional[str]
    match_score: Optional[float]
    final_canonical_name: Optional[str]
    justification: Optional[str] = None


class Enrichment(BaseModel):
    storage_requirements: Optional[str]
    common_side_effects: List[str] = Field(default_factory=list)
    safety_warnings: List[str] = Field(default_factory=list)


class PharmaDocument(BaseModel):
    document_id: str
    timestamp_utc: datetime
    raw_ocr: Optional[RawOCR] = None
    extracted_entities: Optional[ExtractedEntities] = None
    verification: Optional[Verification] = None
    enrichment: Optional[Enrichment] = None
    barcode: Optional[Barcode] = None
