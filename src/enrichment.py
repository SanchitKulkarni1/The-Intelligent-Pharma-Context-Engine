# src/enrichment.py
import requests
from typing import Dict, List, Optional
from schema import Enrichment
from pydantic import Field

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"


def query_openfda_by_ingredient(ingredient: str, limit: int = 1) -> Optional[Dict]:
    try:
        r = requests.get(
            OPENFDA_LABEL_URL,
            params={
                "search": f'active_ingredient:"{ingredient}"',
                "limit": limit
            },
            timeout=5
        )
        return r.json().get("results", [None])[0]
    except Exception:
        return None


def enrich_with_fda(ingredients: List[str]) -> Enrichment:
    enrichment = Enrichment(
        storage_requirements=None,
        common_side_effects=[],
        safety_warnings=[]
    )

    for ingredient in ingredients:
        label = query_openfda_by_ingredient(ingredient)
        if not label:
            continue

        if label.get("storage_and_handling"):
            enrichment.storage_requirements = label["storage_and_handling"][0]

        if label.get("adverse_reactions"):
            enrichment.common_side_effects.extend(label["adverse_reactions"])

        if label.get("warnings"):
            enrichment.safety_warnings.extend(label["warnings"])

    # Deduplicate
    enrichment.common_side_effects = list(set(enrichment.common_side_effects))
    enrichment.safety_warnings = list(set(enrichment.safety_warnings))

    return enrichment
