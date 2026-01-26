import re
import requests
from rapidfuzz import fuzz
from typing import List, Dict, Optional

from schema import Verification
from src.utils.ingredients import extract_ingredients_from_rxnorm
from src.utils.reranker import rerank_rxnorm_candidates
from src.utils.ndc import lookup_drug_by_ndc


# -----------------------------
# Config
# -----------------------------

RXNORM_SEARCH_URL = "https://rxnav.nlm.nih.gov/REST/drugs.json"
RXNORM_CUI_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"

USE_LLM_RERANKER = True
LLM_TOP_K = 5

STOPWORDS = {
    "rx", "take", "tablet", "tablets", "capsule", "capsules",
    "pharmacy", "qty", "refills", "once", "daily", "hours",
    "day", "pain", "mouth", "by", "for"
}

FORMULATION_KEYWORDS = {
    "suspension", "solution", "polistirex",
    "extended release", "12 hr", "er", "mg/ml"
}


# -----------------------------
# Helpers
# -----------------------------

def normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9/]", "", token.lower())


def normalize_ndc(raw: str) -> Optional[str]:
    """
    Normalize barcode → NDC (10 or 11 digits).
    """
    digits = re.sub(r"\D", "", raw)
    if len(digits) in (10, 11):
        return digits
    return None


def extract_candidate_terms(text: str) -> List[str]:
    tokens = text.split()
    candidates = set()

    for token in tokens:
        norm = normalize_token(token)
        for part in re.split(r"[\/+]", norm):
            if len(part) >= 4 and part not in STOPWORDS:
                candidates.add(part)

    return list(candidates)


def formulation_mismatch(rx_term: str, ocr_text: str) -> bool:
    rx = rx_term.lower()
    ocr = ocr_text.lower()
    return any(k in rx and k not in ocr for k in FORMULATION_KEYWORDS)


def query_rxnorm(term: str) -> List[str]:
    try:
        r = requests.get(
            RXNORM_SEARCH_URL,
            params={"name": term},
            timeout=5
        )
        groups = r.json().get("drugGroup", {}).get("conceptGroup", [])
    except Exception:
        return []

    results = []
    for g in groups:
        for p in g.get("conceptProperties", []) or []:
            if "name" in p:
                results.append(p["name"])
    return results


def fetch_rxnorm_cui(drug_name: str) -> Optional[str]:
    try:
        r = requests.get(
            RXNORM_CUI_URL,
            params={"name": drug_name},
            timeout=5
        )
        return r.json().get("idGroup", {}).get("rxnormId", [None])[0]
    except Exception:
        return None





def infer_form(rx_name: str) -> str:
    rx = rx_name.lower()
    if "tablet" in rx:
        return "tablet"
    if "suspension" in rx or "solution" in rx:
        return "liquid"
    return "unknown"


# -----------------------------
# Main Verification
# -----------------------------

def verify_drug(doc):
    """
    Verification pipeline:
    1. Barcode (NDC) override
    2. OCR → RxNorm candidate generation
    3. Optional LLM re-ranking (ambiguity only)
    4. Deterministic fallback
    """

    # =====================================================
    # 1️⃣ BARCODE OVERRIDE (HIGHEST TRUST)
    # =====================================================
    if getattr(doc, "barcode", None):
        ndc = normalize_ndc(doc.barcode.value)
        if ndc:
            drug = lookup_drug_by_ndc(ndc)
            if drug:
                doc.verification = Verification(
                    rxnorm_cui=drug.get("rxcui"),
                    matched_term=drug["name"],
                    match_score=1.0,
                    final_canonical_name=drug["name"],
                    justification="Barcode-derived NDC used as authoritative identifier"
                )
                return doc  # 🔒 HARD STOP

    # =====================================================
    # 2️⃣ OCR → RXNORM CANDIDATES
    # =====================================================
    if not doc.raw_ocr:
        return doc

    text = doc.raw_ocr.full_text
    tokens = extract_candidate_terms(text)

    candidates: Dict[str, Dict] = {}

    for token in tokens:
        for rx_name in query_rxnorm(token):
            if formulation_mismatch(rx_name, text):
                continue

            score = fuzz.token_set_ratio(token, rx_name)
            if score >= 50:
                if rx_name not in candidates or score > candidates[rx_name]["score"]:
                    candidates[rx_name] = {
                        "name": rx_name,
                        "score": score
                    }

    if not candidates:
        doc.verification = Verification(
            rxnorm_cui=None,
            matched_term=None,
            match_score=0.0,
            final_canonical_name=None
        )
        return doc

    rxnorm_candidates = sorted(
        candidates.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    top = rxnorm_candidates[0]
    second = rxnorm_candidates[1] if len(rxnorm_candidates) > 1 else None

    # =====================================================
    # 3️⃣ OPTIONAL LLM RE-RANKING (ONLY IF AMBIGUOUS)
    # =====================================================
    chosen_name = None
    justification = None
    confidence = None

    ambiguous = (
        USE_LLM_RERANKER
        and second is not None
        and abs(top["score"] - second["score"]) < 15
    )

    if ambiguous:
        print(">>> LLM RE-RANKER INVOKED <<<")

        llm_input = [
            {
                "name": c["name"],
                "ingredients": extract_ingredients_from_rxnorm(c["name"]),
                "form": infer_form(c["name"])
            }
            for c in rxnorm_candidates[:LLM_TOP_K]
        ]

        reranked = rerank_rxnorm_candidates(
            ocr_text=text,
            rxnorm_candidates=llm_input
        )

        if reranked:
            chosen_name = reranked["name"]
            justification = reranked.get("reason")
            confidence = 0.95

    # =====================================================
    # 4️⃣ DETERMINISTIC FALLBACK
    # =====================================================
    if not chosen_name:
        chosen_name = top["name"]
        confidence = round(top["score"] / 100, 2)

    cui = fetch_rxnorm_cui(chosen_name)

    doc.verification = Verification(
        rxnorm_cui=cui,
        matched_term=chosen_name,
        match_score=confidence,
        final_canonical_name=chosen_name,
        justification=justification
    )

    return doc
