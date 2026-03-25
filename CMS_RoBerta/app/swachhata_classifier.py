"""
Swachhata complaint classifier using Google Gemini.

Routing logic:
  - If any configured keyword is found (case-insensitive substring) in the complaint text,
    the complaint is sent to Gemini instead of the RoBERTa ensemble.
  - Gemini classifies the complaint into one of the 20 Swachhata categories.
  - Keywords are loaded from swachhata_keywords.json (configurable without code changes).
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 20 Swachhata categories — label_ids are fixed as specified by PMC.
# NOTE: These IDs overlap with RoBERTa label_ids; use isSwachhata flag to
# distinguish which namespace a label_id belongs to in API responses.
# ---------------------------------------------------------------------------
SWACHHATA_CATEGORIES: Dict[int, str] = {
    1:  "Removal of Dead Animals",
    2:  "Dustbins Not Cleaned",
    3:  "Garbage Dump",
    4:  "Garbage Vehicle Not Arrived",
    5:  "Sweeping Not Done",
    6:  "No Electricity in Public Toilet",
    7:  "No Water Supply in Public Toilet",
    8:  "Blockage in Public Toilet",
    9:  "Uncleaning Public Toilet",
    10: "Open Manholes or Drains",
    11: "Overflow of Sewerage or Storm Water",
    12: "Stagnant Water on Road/ Open Area",
    13: "Improper Disposal of Fecal Waste/Septage",
    14: "Removal of Debris/Construction Material",
    15: "Burning of Garbage in Open Space",
    16: "Open Defecation",
    26: "Overflow of Septic Tanks",
    27: "Yellow Spot (Public Urination Spot)",
    28: "Cleanliness Target Unit (Dirty Spot)",
    29: "Unsafe Manhole Entry",
}

# Reverse map: category name -> label_id (for already_learned overrides)
SWACHHATA_NAME_TO_ID: Dict[str, int] = {v: k for k, v in SWACHHATA_CATEGORIES.items()}

_KEYWORDS_FILE = Path(__file__).resolve().parent / "swachhata_keywords.json"
_gemini_client = None

_CATEGORIES_BLOCK = "\n".join(
    f"  label_id {lid}: {name}" for lid, name in SWACHHATA_CATEGORIES.items()
)

_CLASSIFICATION_PROMPT = f"""\
You are a municipal complaint classification assistant for Swachhata (cleanliness/sanitation) related complaints.

Classify the given complaint into exactly one of these 20 categories:

{_CATEGORIES_BLOCK}

Rules:
- Choose the single most appropriate category.
- confidence should be a float between 0.0 (not sure) and 1.0 (very sure).
- Respond with ONLY a valid JSON object — no markdown fences, no explanation.

Expected format:
{{"label": "<exact category name from list above>", "label_id": <integer>, "confidence": <float>}}
"""


def load_keywords(path: Optional[Path] = None) -> List[str]:
    """
    Load keyword list from JSON file. Returns lowercase strings for case-insensitive matching.
    Falls back to an empty list if the file is missing (logs a warning).
    """
    p = Path(path) if path else _KEYWORDS_FILE
    if not p.exists():
        logger.warning("Swachhata keywords file not found at %s — no keyword routing will occur.", p)
        return []
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    keywords = [kw.strip().lower() for kw in data.get("keywords", []) if kw.strip()]
    logger.info("Loaded %d Swachhata keywords from %s", len(keywords), p)
    return keywords


def contains_swachhata_keyword(text: str, keywords: List[str]) -> bool:
    """
    Return True if any keyword appears as a case-insensitive substring in text.
    Short-circuits on the first match.
    """
    if not keywords:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def _get_gemini_client():
    """Lazy-initialise the Gemini client (singleton per process).

    Uses the current google-genai SDK (google.genai).
    Model is read from the GEMINI_MODEL env var (default: gemini-2.0-flash).
    """
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is not installed. Run: pip install google-genai"
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Add it to your .env file or export it before starting the server."
        )

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip()
    _gemini_client = genai.Client(api_key=api_key)
    # Store model name alongside client so generate_content can use it
    _gemini_client._swachhata_model = model_name
    logger.info("Gemini client initialised (model=%s).", model_name)
    return _gemini_client


def classify_swachhata(text: str) -> dict:
    """
    Classify a single complaint into one of the 20 Swachhata categories via Gemini.

    Returns:
        {
            "label": str,       # canonical category name
            "label_id": int,    # PMC-assigned Swachhata label_id
            "confidence": float # 0.0 – 1.0 as reported by Gemini
        }

    Raises:
        RuntimeError: if the Gemini API call fails, returns unparseable JSON,
                      or returns a label_id outside the 20 Swachhata categories.
    """
    client = _get_gemini_client()
    model_name = getattr(client, "_swachhata_model", "gemini-2.0-flash")
    full_prompt = f"{_CLASSIFICATION_PROMPT}\n\nComplaint: {text}"

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
        )
        raw = response.text.strip()
    except Exception as exc:
        raise RuntimeError(f"Gemini API request failed: {exc}") from exc

    # Strip accidental markdown code fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Gemini returned non-JSON response: {raw[:200]!r}"
        ) from exc

    # Validate and normalise label_id
    try:
        label_id = int(result["label_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Gemini response missing or invalid 'label_id': {result}"
        ) from exc

    if label_id not in SWACHHATA_CATEGORIES:
        raise RuntimeError(
            f"Gemini returned label_id={label_id} which is not in the 20 Swachhata "
            f"categories. Valid ids: {sorted(SWACHHATA_CATEGORIES.keys())}"
        )

    # Always use the canonical category name from our map (ignore Gemini's spelling)
    canonical_label = SWACHHATA_CATEGORIES[label_id]

    # Normalise confidence
    try:
        confidence = round(float(max(0.0, min(1.0, result.get("confidence", 1.0)))), 4)
    except (TypeError, ValueError):
        confidence = None

    return {
        "label": canonical_label,
        "label_id": label_id,
        "confidence": confidence,
    }
