import re
from typing import List

KNOWN_INGREDIENTS = [
    "acetaminophen",
    "hydrocodone",
    "ibuprofen",
    "amoxicillin",
    "clavulanate",
    "metformin",
    "paracetamol",
    "pseudoephedrine",
    "chlorpheniramine"
]

def extract_ingredients_from_rxnorm(name: str) -> List[str]:
    """
    Extract active ingredients from a verified RxNorm drug name.
    """
    name_lower = name.lower()
    found = []

    for ingredient in KNOWN_INGREDIENTS:
        if ingredient in name_lower:
            found.append(ingredient)

    # Fallback: split on slash if unknown
    if not found:
        parts = re.split(r"[\/,+]", name_lower)
        found = [p.strip() for p in parts if len(p.strip()) > 4]

    return list(set(found))
