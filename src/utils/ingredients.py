import requests
from typing import List

RXNORM_RELATED_URL = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json"


def extract_ingredients_from_rxnorm(rxcui: str) -> List[str]:
    """
    Production-grade ingredient extraction using RxNorm relationships.
    Resolves authoritative active ingredients (IN / PIN).
    """
    if not rxcui:
        return []

    try:
        resp = requests.get(
            RXNORM_RELATED_URL.format(rxcui=rxcui),
            params={"tty": "IN+PIN"},
            timeout=5
        )
        data = resp.json()

        ingredients = []

        groups = data.get("relatedGroup", {}).get("conceptGroup", [])
        for group in groups:
            for prop in group.get("conceptProperties", []) or []:
                name = prop.get("name")
                if name:
                    ingredients.append(name.lower())

        return sorted(set(ingredients))

    except Exception:
        return []
