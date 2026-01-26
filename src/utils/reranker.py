# src/llm/reranker.py
import os
import json
import re
from typing import List, Dict, Optional

from google import genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-flash-latest"

client = genai.Client(api_key=API_KEY)


# -----------------------------
# Helpers
# -----------------------------

def extract_json(text: str) -> Optional[dict]:
    """
    Safely extract JSON object from LLM output.
    """
    if not text:
        return None

    # Remove markdown fences
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"```.*?\n", "", text)
        text = text.replace("```", "")

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        return None

    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


# -----------------------------
# Main Reranker
# -----------------------------

def rerank_rxnorm_candidates(
    ocr_text: str,
    rxnorm_candidates: List[Dict]
) -> Optional[Dict]:
    """
    Use LLM to select the most clinically appropriate RxNorm candidate.
    """

    prompt = f"""
You are a clinical medication normalization assistant.

OCR TEXT:
{ocr_text}

RXNORM CANDIDATES:
{json.dumps(rxnorm_candidates, indent=2)}

TASK:
Select the SINGLE best RxNorm candidate.

RULES:
- Prefer candidates whose ingredients are a SUBSET of the OCR evidence
- Penalize candidates introducing extra ingredients
- Use indication clues like "for pain"
- Ignore brand noise

Return ONLY valid JSON:
{{
  "name": "<exact candidate name>",
  "reason": "<short justification>"
}}
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
    except Exception as e:
        print("LLM call failed:", e)
        return None

    raw_text = response.text
    print(">>> GEMINI RAW RESPONSE <<<")
    print(raw_text)

    parsed = extract_json(raw_text)
    print(">>> GEMINI PARSED JSON <<<")
    print(parsed)

    if not parsed:
        return None

    if "name" not in parsed:
        return None

    return parsed
