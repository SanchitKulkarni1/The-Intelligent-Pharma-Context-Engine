# src/verification.py
import re
from pydantic import BaseModel
import requests
from rapidfuzz import fuzz
from typing import List, Dict, Optional

from schema import Verification
from src.llm.reranker import rerank_rxnorm_candidates


RXNORM_SEARCH_URL = "https://rxnav.nlm.nih.gov/REST/drugs.json"
USE_LLM_RERANKER = True


STOPWORDS = {
    "rx", "take", "tablet", "tablets", "capsule", "capsules",
    "pharmacy", "qty", "refills", "once", "daily", "hours",
    "day", "pain", "mouth", "by", "for"
}

FORMULATION_KEYWORDS = {
    "suspension", "solution", "polistirex",
    "extended release", "12 hr", "er", "mg/ml"
}

class Verification(BaseModel):
    rxnorm_cui: Optional[str]
    matched_term: Optional[str]
    match_score: float
    final_canonical_name: Optional[str]
    justification: Optional[str] = None
    

# -----------------------------
# Helpers
# -----------------------------

def normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9/]", "", token.lower())


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


def fetch_rxnorm_cui(drug_name: str):
    try:
        r = requests.get(
            "https://rxnav.nlm.nih.gov/REST/rxcui.json",
            params={"name": drug_name},
            timeout=5
        )
        return r.json().get("idGroup", {}).get("rxnormId", [None])[0]
    except Exception:
        return None


def infer_ingredients(rx_name: str) -> List[str]:
    rx = rx_name.lower()
    ingredients = []

    if "hydrocodone" in rx:
        ingredients.append("hydrocodone")
    if "acetaminophen" in rx or "apap" in rx:
        ingredients.append("acetaminophen")
    if "pseudoephedrine" in rx:
        ingredients.append("pseudoephedrine")
    if "chlorpheniramine" in rx:
        ingredients.append("chlorpheniramine")

    return ingredients


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
    text = doc.raw_ocr.full_text
    tokens = extract_candidate_terms(text)

    candidates: Dict[str, Dict] = {}

    # 1️⃣ Collect ALL plausible RxNorm candidates
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

    rxnorm_candidates = list(candidates.values())

    # 2️⃣ Decide LLM usage
    use_llm = USE_LLM_RERANKER and len(rxnorm_candidates) > 1

    # 3️⃣ LLM Re-ranking
    if use_llm:
        print(">>> LLM RE-RANKER INVOKED <<<")

        llm_input = [
            {
                "name": c["name"],
                "ingredients": infer_ingredients(c["name"]),
                "form": infer_form(c["name"])
            }
            for c in rxnorm_candidates
        ]

    reranked = rerank_rxnorm_candidates(
        ocr_text=text,
        rxnorm_candidates=llm_input
    )

    print("LLM raw output:", reranked)

    if reranked:
        chosen_name = reranked["name"]
        print("LLM CHOSEN NAME:", chosen_name)
        print("LLM REASON:", reranked.get("reason"))
    else:
        print("LLM returned None — falling back to deterministic")
        chosen_name = max(rxnorm_candidates, key=lambda x: x["score"])["name"]


    cui = fetch_rxnorm_cui(chosen_name)

    justification=reranked.get("reason") if reranked else None
    print ("Reason for choosing:", justification)

    doc.verification = Verification(
        rxnorm_cui=cui,
        matched_term=chosen_name,
        match_score=0.95,
        final_canonical_name=chosen_name,
        justification=justification
    )

    return doc
