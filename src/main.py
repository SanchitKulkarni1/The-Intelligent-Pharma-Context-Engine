# src/main.py (Modified)
import sys
from ocr import basic_ocr_test
from logic import PharmaLogic

class PharmaPipeline:
    def __init__(self):
        self.logic = PharmaLogic()

    def run(self, image_path):
        print(f"--- STAGE 1: VISION (Reading {image_path}) ---")
        
        # 1. Run OCR
        raw_lines = basic_ocr_test(image_path)
        print(f"Raw OCR Lines: {raw_lines}")

        # 2. Reasoning
        print(f"\n--- STAGE 2: REASONING (Fuzzy Match) ---")
        verified_drug = self.logic.find_best_match(raw_lines)
        
        # --- FORCE TEST BLOCK (ADD THIS) ---
        if not verified_drug:
            print("[!] OCR missed the drug name. INJECTING 'Hydrocodone' for API testing...")
            verified_drug = "Hydrocodone"
        # -----------------------------------

        if verified_drug:
            print(f"\n--- STAGE 3: ENRICHMENT (FDA Database) ---")
            fda_data = self.logic.fetch_fda_data(verified_drug)
            
            if fda_data:
                print("\n" + "="*40)
                print(f"💊 FINAL OUTPUT: {verified_drug}")
                print("="*40)
                print(f"Generic Name:   {fda_data['generic_name']}")
                print(f"Manufacturer:   {fda_data['manufacturer']}")
                print(f"Class:          {fda_data['pharm_class']}")
                print(f"Storage:        {fda_data['storage']}")
                print("="*40)
            else:
                print("[-] Drug verified, but no FDA data found.")

if __name__ == "__main__":
    image_file = "images/testimage.jpeg" 
    app = PharmaPipeline()
    app.run(image_file)