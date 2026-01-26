import requests

def lookup_drug_by_ndc(ndc: str):
    try:
        r = requests.get(
            "https://api.fda.gov/drug/ndc.json",
            params={"search": f'product_ndc:"{ndc}"', "limit": 1},
            timeout=5
        )
        results = r.json().get("results", [])
        if not results:
            return None

        product = results[0]
        return {
            "name": product.get("generic_name"),
            "rxcui": None  # optional
        }
    except Exception:
        return None
