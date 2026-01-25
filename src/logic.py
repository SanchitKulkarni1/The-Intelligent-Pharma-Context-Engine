import requests
from rapidfuzz import process, fuzz
import re

class PharmaLogic:
    def __init__(self):
        self.base_url = "https://api.fda.gov/drug/label.json"
        # 1. THE KNOWLEDGE BASE
        # In a real app, this is a database. For your prototype, we list likely drugs.
        # I added "Hydrocodone" specifically for your test image.
        self.known_drugs = [
            "Hydrocodone", "Ibuprofen", "Amoxicillin", "Lisinopril", 
            "Metformin", "Atorvastatin", "Acetaminophen"
        ]

    def find_best_match(self, ocr_text_list):
        """
        Scans the messy OCR lines to find a drug name from our 'known_drugs' list.
        """
        print(f"[*] Scanning {len(ocr_text_list)} lines for drug names...")
        
        # We test every line against every known drug
        best_candidate = None
        highest_score = 0

        for line in ocr_text_list:
            # Skip short noise like "P9", "No R"
            if len(line) < 4: 
                continue
                
            # 'process.extractOne' finds the best match for 'line' in 'known_drugs'
            match = process.extractOne(line, self.known_drugs, scorer=fuzz.partial_ratio)
            
            if match:
                drug_name, score, index = match
                # We only trust it if confidence is > 85%
                if score > 85 and score > highest_score:
                    highest_score = score
                    best_candidate = drug_name
                    print(f"   -> Possible match: '{line}' matches '{drug_name}' (Score: {score})")

        return best_candidate

    def fetch_fda_data(self, drug_name):
        """
        2. ENRICHMENT: Queries the openFDA API for the verified drug name.
        """
        if not drug_name:
            return None

        print(f"[*] Querying FDA database for: {drug_name}...")
        
        # FDA Query Syntax: search=openfda.brand_name:"DRUG"
        search_term = f'openfda.brand_name:"{drug_name}"'
        params = {
            'search': search_term,
            'limit': 1
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if "results" in data:
                result = data['results'][0]
                openfda = result.get('openfda', {})
                
                # Extract clean fields
                return {
                    "verified_name": drug_name,
                    "manufacturer": openfda.get('manufacturer_name', ['Unknown'])[0],
                    "generic_name": openfda.get('generic_name', ['Unknown'])[0],
                    "pharm_class": openfda.get('pharm_class_epc', ['Unknown'])[0],
                    "boxed_warning": "YES" if "boxed_warning" in result else "No",
                    "storage": result.get('storage_and_handling', ['Not listed'])[0][:150] + "..."
                }
        except Exception as e:
            print(f"[!] API Error: {e}")
            return None