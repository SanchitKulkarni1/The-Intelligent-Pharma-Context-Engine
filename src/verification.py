# src/verification.py
import re
import requests
from rapidfuzz import fuzz, process
from schema import Verification

# Words we NEVER want to treat as drug names
STOPWORDS = {
    "rx", "take", "tablet", "tablets", "capsule", "capsules",
    "pharmacy", "qty", "refills", "once", "daily", "hours",
    "day", "pain", "mouth", "by", "for"
}

RXNORM_SEARCH_URL = "https://rxnav.nlm.nih.gov/REST/drugs.json"


def normalize_token(token: str) -> str:
    token = token.lower()
    token = re.sub(r"[^a-z0-9/]", "", token)
    return token


def extract_candidate_terms(text: str) -> list[str]:
    tokens = text.split()
    candidates = set()

    for token in tokens:
        norm = normalize_token(token)

        # Split combination drugs like hydrocodone/apap
        parts = re.split(r"[\/+]", norm)

        for part in parts:
            if len(part) < 4:
                continue
            if part in STOPWORDS:
                continue
            candidates.add(part)

    return list(candidates)


def query_rxnorm(term: str) -> list[str]:
    """
    Query RxNorm and return possible drug names.
    """
    try:
        resp = requests.get(RXNORM_SEARCH_URL, params={"name": term}, timeout=5)
        data = resp.json()
        concepts = data.get("drugGroup", {}).get("conceptGroup", [])
    except Exception:
        return []

    results = []
    for group in concepts:
        for prop in group.get("conceptProperties", []) or []:
            results.append(prop.get("name"))

    return results

FORMULATION_KEYWORDS = {
    "suspension", "solution", "polistirex",
    "extended release", "12 hr", "er",
    "mg/ml"
}

def formulation_mismatch(rx_term: str, ocr_text: str) -> bool:
    rx_term_lower = rx_term.lower()
    ocr_lower = ocr_text.lower()

    for keyword in FORMULATION_KEYWORDS:
        if keyword in rx_term_lower and keyword not in ocr_lower:
            return True
    return False



def verify_drug(doc):
    """
    Main verification stage.
    """
    text = doc.raw_ocr.full_text
    candidates = extract_candidate_terms(text)

    best_match = None
    best_score = 0
    best_term = None
    best_cui = None

    for candidate in candidates:
        rxnorm_terms = query_rxnorm(candidate)

        for rx_term in rxnorm_terms:
            if formulation_mismatch(rx_term, text):
                continue
            score = fuzz.token_set_ratio(candidate, rx_term)

            if score > best_score:
                best_score = score
                best_match = rx_term

    # Threshold: below this, we consider verification failed
    if best_score < 80:
        doc.verification = Verification(
            rxnorm_cui=None,
            matched_term=None,
            match_score=best_score,
            final_canonical_name=None
        )
        return doc

    # Fetch CUI for the matched term
    try:
        cui_resp = requests.get(
            RXNORM_SEARCH_URL,
            params={"name": best_match},
            timeout=5
        )
        cui_data = cui_resp.json()
        cui = (
            cui_data["drugGroup"]["conceptGroup"][0]
            ["conceptProperties"][0]["rxcui"]
        )
    except Exception:
        cui = None

    doc.verification = Verification(
        rxnorm_cui=cui,
        matched_term=best_match,
        match_score=round(best_score / 100, 2),
        final_canonical_name=best_match
    )

    return doc
