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
import re
import math
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# ---------------------------------------------------------------------------
# Fallback (Gemini-free) classifier:
# - Builds category keyword buckets from `swachhata_keywords.json`
# - Scores each of the 20 categories using:
#   1) weighted keyword substring hits (phrase length + token IDF)
#   2) lightweight token overlap
#   3) char-level TF-IDF cosine similarity to category keyword "documents"
#   4) targeted disambiguation boosts for the few fine-grained ambiguous groups
# - No model training required; only vectorizer fitting at runtime.
# ---------------------------------------------------------------------------
_fallback_cache = {
    "built": False,
    "category_keywords": None,  # Dict[int, List[str]] lowercased
    "category_token_sets": None,  # Dict[int, set[str]]
    "token_idf": None,  # Dict[str, float]
    "cat_doc_texts": None,  # Dict[int, str]
    "tfidf_word": None,  # TfidfVectorizer
    "tfidf_word_mat": None,  # np.ndarray
    "tfidf_char": None,  # TfidfVectorizer
    "tfidf_char_mat": None,  # np.ndarray
    "cat_ids": None,  # List[int]
}


def _tokenize(text: str) -> List[str]:
    """
    Tokenize for both Romanized Marathi and English and also Devanagari.
    We keep it simple and robust: unicode word chars via `\\w`.
    """
    # Lower first; we also keep digits because some complaints mention timings/days.
    # `\\w` covers Devanagari letters too under Python's unicode rules.
    return re.findall(r"\w+", text.lower())


def _build_fallback_cache() -> None:
    """
    Build cached keyword buckets + TF-IDF indices for fallback classification.
    This is deterministic and does not require any external LLM or training.
    """
    if _fallback_cache["built"]:
        return

    # Load the flat keyword list and reconstruct per-category buckets.
    # Convention: in swachhata_keywords.json, each category block starts with the
    # exact category name string (anchor), followed by related phrases/variants.
    if not _KEYWORDS_FILE.exists():
        # Fallback should still work (but confidence will be low) if keyword JSON is missing.
        _fallback_cache["built"] = True
        _fallback_cache["category_keywords"] = {lid: [] for lid in SWACHHATA_CATEGORIES}
        _fallback_cache["category_token_sets"] = {lid: set() for lid in SWACHHATA_CATEGORIES}
        _fallback_cache["token_idf"] = {}
        _fallback_cache["cat_doc_texts"] = {lid: "" for lid in SWACHHATA_CATEGORIES}
        _fallback_cache["cat_ids"] = sorted(SWACHHATA_CATEGORIES.keys())
        # Avoid sklearn "empty vocabulary" errors by fitting on a dummy string.
        dummy = "dummy"
        _fallback_cache["tfidf_word"] = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True)
        _fallback_cache["tfidf_word_mat"] = _fallback_cache["tfidf_word"].fit_transform([dummy] * len(_fallback_cache["cat_ids"])).toarray()
        _fallback_cache["tfidf_char"] = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
        _fallback_cache["tfidf_char_mat"] = _fallback_cache["tfidf_char"].fit_transform([dummy] * len(_fallback_cache["cat_ids"])).toarray()
        return

    with open(_KEYWORDS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    raw_keywords = [kw.strip() for kw in data.get("keywords", []) if isinstance(kw, str) and kw.strip()]

    # Map normalized category name -> label_id
    name_to_id_norm = {name.lower().strip(): lid for lid, name in SWACHHATA_CATEGORIES.items()}

    category_keywords: Dict[int, List[str]] = {lid: [] for lid in SWACHHATA_CATEGORIES}
    current_id: Optional[int] = None

    for kw in raw_keywords:
        kw_norm = kw.lower().strip()
        if kw_norm in name_to_id_norm:
            current_id = name_to_id_norm[kw_norm]
            # Include the anchor itself: it can help if the complaint contains the canonical phrase.
            category_keywords[current_id].append(kw_norm)
            continue
        if current_id is not None:
            category_keywords[current_id].append(kw_norm)

    cat_ids = sorted(category_keywords.keys())

    # Token sets per category (for fast token overlap scoring and df computation)
    category_token_sets: Dict[int, set] = {}
    for lid in cat_ids:
        toks: set = set()
        for kw in category_keywords[lid]:
            toks.update(_tokenize(kw))
        category_token_sets[lid] = toks

    # IDF over category-level token document frequencies
    df: Dict[str, int] = {}
    for lid in cat_ids:
        for tok in category_token_sets[lid]:
            df[tok] = df.get(tok, 0) + 1
    n_cats = len(cat_ids)
    token_idf: Dict[str, float] = {}
    for tok, dfi in df.items():
        # Smooth to avoid div-by-zero and keep values stable.
        token_idf[tok] = math.log((n_cats + 1.0) / (dfi + 1.0)) + 1.0

    # Category "documents" for TF-IDF.
    cat_doc_texts: Dict[int, str] = {lid: " ".join(category_keywords[lid]) for lid in cat_ids}

    # Fit TF-IDF on the 20 category documents once at runtime.
    # Use both word-level and char-level features; char handles Romanized Marathi spelling variants.
    docs = [cat_doc_texts[lid] for lid in cat_ids]

    tfidf_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True)
    tfidf_word_mat = tfidf_word.fit_transform(docs).toarray()

    tfidf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
    tfidf_char_mat = tfidf_char.fit_transform(docs).toarray()

    _fallback_cache.update(
        {
            "built": True,
            "category_keywords": category_keywords,
            "category_token_sets": category_token_sets,
            "token_idf": token_idf,
            "cat_doc_texts": cat_doc_texts,
            "tfidf_word": tfidf_word,
            "tfidf_word_mat": tfidf_word_mat,
            "tfidf_char": tfidf_char,
            "tfidf_char_mat": tfidf_char_mat,
            "cat_ids": cat_ids,
        }
    )


def _has_any(text_lower: str, patterns: List[str]) -> bool:
    return any(p in text_lower for p in patterns)


def classify_swachhata_fallback(text: str) -> dict:
    """
    Deterministic Gemini-free fallback for the 20 Swachhata categories.
    Returns the same schema as Gemini classification plus a source flag:
      {"label": str, "label_id": int, "confidence": float, "isLLM": bool}
    """
    _build_fallback_cache()

    text = (text or "").strip()
    text_lower = text.lower()
    complaint_tokens_set = set(_tokenize(text_lower))

    cat_ids = _fallback_cache["cat_ids"]
    category_keywords: Dict[int, List[str]] = _fallback_cache["category_keywords"]
    category_token_sets: Dict[int, set] = _fallback_cache["category_token_sets"]
    token_idf: Dict[str, float] = _fallback_cache["token_idf"]

    # Layer 1: weighted substring keyword hits + token overlap.
    layer1_scores: Dict[int, float] = {lid: 0.0 for lid in cat_ids}
    for lid in cat_ids:
        kw_list = category_keywords.get(lid, [])
        token_set = category_token_sets.get(lid, set())

        phrase_score = 0.0
        for kw in kw_list:
            if kw and kw in text_lower:
                kw_toks = _tokenize(kw)
                if not kw_toks:
                    continue
                # Phrase length and IDF together: longer & rarer tokens => higher confidence.
                phrase_score += sum(token_idf.get(t, 0.0) for t in kw_toks) * (0.6 + 0.4 * len(kw_toks))

        token_score = 0.0
        # Token overlap (fast and helps when exact phrases don't match but tokens do).
        for tok in complaint_tokens_set:
            if tok in token_set:
                token_score += token_idf.get(tok, 0.0)

        # Blend inside Layer 1.
        layer1_scores[lid] = 0.75 * phrase_score + 0.25 * token_score

    # Layer 2: disambiguation boosts for ambiguous sub-groups.
    # These are tuned from your keyword vocabulary structure.
    bonus: Dict[int, float] = {lid: 0.0 for lid in cat_ids}

    # Toilet sub-categories (6,7,8,9) disambiguation
    has_light = _has_any(
        text_lower,
        ["light", "electric", "power", "andhaar", "dark", "veej", "वीज", "लाईट", "वीज नाही", "अंधार", "अंधार आहे", "अंधार आहे", "toilet la light", "शौचालयात लाईट", "सार्वजनिक शौचालयात लाईट", "लाईट नाही"],
    )
    has_water = _has_any(
        text_lower,
        ["water", "pani", "paani", "tap", "supply", "moot", "mootar", "pani nahi", "पाणी", "पाणी नाही", "टॉयलेटमध्ये पाणी", "शौचालयात पाणी", "पाणी येत नाही", "पाणी बंद", "मुताऱ्यात पाणी"],
    )
    has_block = _has_any(
        text_lower,
        ["blocked", "choked", "chokup", "tumb", "tumbale", "jam", "clog", "overflow blockage", "drain blocked", "tuंब", "तुंब", "चोकअप", "टुंबली", "ड्रेनेज बंद", "तुंबला", "मुतारी तुंबली", "शौचालय बंद पडले", "टॉयलेट ड्रेनेज बंद"],
    )
    has_dirty = _has_any(
        text_lower,
        ["dirty", "unclean", "filthy", "smelly", "durgandhi", "durgandhi", "durgandhi yet", "ghani", "ghan", "ghan aahe", "aswachha", "ghān", "घाण", "दुर्गंध", "दुर्गंधी", "पब्लिक टॉयलेट घाण", "public toilet smelly", "अस्वच्छ", "स्वच्छता नाही", "घाण जागा"],
    )
    if any(lid in [6, 7, 8, 9] for lid in cat_ids):
        # Apply within the group even if the complaint is partial.
        if has_light:
            bonus[6] += 0.25
        if has_water:
            bonus[7] += 0.25
        if has_block:
            bonus[8] += 0.25
        if has_dirty:
            bonus[9] += 0.25

    # Manhole group (10 vs 29)
    has_open = _has_any(
        text_lower,
        ["open", "cover", "uncovered", "missing", "zhakan", "zhakan nahi", "cover missing", "cover gone", "manhole cover", "झाकण", "झाकण नाही", "झाकण गायब", "मॅनहोल उघडे", "उघडे चेंबर", "उघडे चेंबर", "drain ughda", "nala ughda"],
    )
    has_unsafe = _has_any(
        text_lower,
        ["fell", "fall", "accident", "danger", "trap", "barricade", "unsafe", "person fell", "manhole accident", "अपघात", "पडणे", "पडला", "धोकादायक", "असुरक्षित", "बॅरिकेड", "बॅरिकेड नाही", "manhole unsafe aahe barricade nahi"],
    )
    if has_open and 10 in bonus:
        bonus[10] += 0.22
    if has_unsafe and 29 in bonus:
        bonus[29] += 0.22

    # Overflow group (11 sewer vs 26 septic)
    has_sewerish = _has_any(
        text_lower,
        ["sewer", "sewage", "nala", "drainage", "gutter", "storm water", "sandpani", "सांडपाणी", "गटाराचे पाणी", "नाला ओसंडत", "ड्रेनेजचे पाणी रस्त्यावर"],
    )
    has_septic = _has_any(
        text_lower,
        ["septic", "septage", "septik", "tanki", "tank", "septi", "septic tank", "सेप्टिक", "सेप्टेज", "सेप्टिक टाकी", "सेप्टिक टँक"],
    )
    if has_sewerish:
        bonus[11] += 0.22
    if has_septic:
        bonus[26] += 0.22

    # Stagnant vs overflow (12 vs 11): both can mention "water on road"
    has_stagnant = _has_any(
        text_lower,
        ["waterlogging", "standing water", "water pool", "water accumulated", "sathale", "pani sathale", "पाणी साचले", "पाणी भरले", "निचरा नाही", "पाणी उभे", "पाणी जात नाही", "उभे आहे रस्त्यावर"],
    )
    has_overflow = _has_any(
        text_lower,
        ["overflow", "osandat", "overflow road", "overflowing", "flooding", "drain overflow", "ओव्हरफ्लो", "ओव्हरफ्लो रस्त्यावर", "ड्रेनेज ओव्हरफ्लो", "ओसंडत", "नाला ओसंडत", "sewerage overflow"],
    )
    if has_stagnant:
        bonus[12] += 0.22
    if has_overflow:
        bonus[11] += 0.18

    # Layer 3: TF-IDF cosine similarity (word + char) against category documents.
    # We don't need exact per-token IDF here; it complements substring matching.
    tfidf_word = _fallback_cache["tfidf_word"]
    tfidf_word_mat = _fallback_cache["tfidf_word_mat"]
    tfidf_char = _fallback_cache["tfidf_char"]
    tfidf_char_mat = _fallback_cache["tfidf_char_mat"]

    # Transform complaint into the fitted spaces and compute cosine similarities.
    q_word = tfidf_word.transform([text_lower]).toarray()
    q_char = tfidf_char.transform([text_lower]).toarray()

    sims_word = cosine_similarity(q_word, tfidf_word_mat).flatten().tolist()
    sims_char = cosine_similarity(q_char, tfidf_char_mat).flatten().tolist()

    # Combine: normalize Layer1 scores to [0,1] and blend with TF-IDF similarities.
    l1_vals = [layer1_scores[lid] for lid in cat_ids]
    l1_min = min(l1_vals) if l1_vals else 0.0
    l1_max = max(l1_vals) if l1_vals else 0.0
    l1_den = (l1_max - l1_min) or 1e-9

    final_raw_scores: Dict[int, float] = {}
    for idx, lid in enumerate(cat_ids):
        l1_norm = (layer1_scores[lid] - l1_min) / l1_den
        tfidf_sim = 0.45 * float(sims_word[idx]) + 0.55 * float(sims_char[idx])
        # Apply disambiguation bonus as an additive term in the same [0,1] scale.
        final_raw_scores[lid] = 0.62 * l1_norm + 0.38 * tfidf_sim + bonus.get(lid, 0.0)

    # Confidence: use the margin between top-1 and top-2 scores.
    # If top-1 and top-2 are close, confidence should be low.
    best_id = max(final_raw_scores, key=final_raw_scores.get) if final_raw_scores else min(cat_ids)
    all_scores = sorted(final_raw_scores.values(), reverse=True) if final_raw_scores else [0.0]
    if not all_scores:
        confidence = 0.0
    elif len(all_scores) == 1:
        confidence = 1.0
    else:
        winner = all_scores[0]
        second = all_scores[1]
        s_min = min(all_scores)
        s_max = max(all_scores)
        denom = (s_max - s_min) or 1e-9
        confidence = round(float((winner - second) / denom), 4)

    canonical_label = SWACHHATA_CATEGORIES[best_id]
    return {
        "label": canonical_label,
        "label_id": best_id,
        "confidence": confidence,
        "isLLM": False,
    }

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
    Routing gate for Swachhata path.

    Strategy (high recall with controlled false positives):
    1) Exact keyword substring (current behavior).
    2) Token-overlap score against the Swachhata keyword vocabulary.
    3) Word+char TF-IDF similarity against the 20 category keyword documents.

    This catches paraphrases/romanized spellings that miss exact substring checks.
    """
    if not text or not keywords:
        return False

    text_lower = text.lower().strip()

    # 1) Exact match (fast path)
    if any(kw in text_lower for kw in keywords):
        return True

    # 2) + 3) Similarity-based gate
    try:
        _build_fallback_cache()

        complaint_tokens = set(_tokenize(text_lower))
        if not complaint_tokens:
            return False

        # Domain trigger guard: similarity routing is allowed only when text
        # appears sanitation-related. This prevents over-routing generic complaints
        # (e.g., street lights, water supply, tax) into Swachhata.
        sanitation_triggers = {
            "garbage", "waste", "dustbin", "bin", "dump", "dumping",
            "sweeping", "sweep", "cleaning", "cleanliness", "dirty", "filthy", "smelly",
            "toilet", "toilets", "washroom", "sandas", "shauchalay", "mootari",
            "manhole", "drain", "drainage", "nala", "gutter", "sewer", "sewage",
            "septic", "septage", "fecal", "feces", "defecation", "urination",
            "debris", "rubble", "carcass", "animal",
            "कचरा", "कचरापेटी", "घंटागाडी", "झाडू", "सफाई", "स्वच्छता", "घाण",
            "शौचालय", "टॉयलेट", "संडास", "मुतारी",
            "मॅनहोल", "चेंबर", "ड्रेनेज", "गटार", "नाला", "सांडपाणी",
            "सेप्टिक", "सेप्टेज", "मैला", "मलविसर्जन", "लघवी", "लघुशंका",
            "राडारोडा", "मेलेला", "मृत",
            "kachra", "kachrapeti", "ghanta", "jhadu", "safai", "swachhata",
            "sandas", "toilet", "mootari", "manhole", "nala", "gutter",
            "septic", "septage", "maila", "laghavi", "laghushanka", "radaroda", "mela",
        }
        has_domain_trigger = bool(complaint_tokens.intersection(sanitation_triggers))
        if not has_domain_trigger:
            return False

        category_token_sets: Dict[int, set] = _fallback_cache["category_token_sets"]
        token_idf: Dict[str, float] = _fallback_cache["token_idf"]
        cat_ids = _fallback_cache["cat_ids"]

        # Token-overlap signal (IDF-weighted).
        best_overlap = 0.0
        best_overlap_count = 0
        for lid in cat_ids:
            overlap_tokens = complaint_tokens.intersection(category_token_sets.get(lid, set()))
            if not overlap_tokens:
                continue
            overlap_score = sum(token_idf.get(t, 0.0) for t in overlap_tokens)
            if overlap_score > best_overlap:
                best_overlap = overlap_score
                best_overlap_count = len(overlap_tokens)

        # TF-IDF similarity signal (handles spelling variation better).
        tfidf_word = _fallback_cache["tfidf_word"]
        tfidf_word_mat = _fallback_cache["tfidf_word_mat"]
        tfidf_char = _fallback_cache["tfidf_char"]
        tfidf_char_mat = _fallback_cache["tfidf_char_mat"]
        q_word = tfidf_word.transform([text_lower]).toarray()
        q_char = tfidf_char.transform([text_lower]).toarray()
        sims_word = cosine_similarity(q_word, tfidf_word_mat).flatten().tolist()
        sims_char = cosine_similarity(q_char, tfidf_char_mat).flatten().tolist()
        best_tfidf = max((0.45 * float(sw) + 0.55 * float(sc)) for sw, sc in zip(sims_word, sims_char))

        # Conservative thresholds to avoid routing non-Swachhata complaints by mistake.
        overlap_gate = best_overlap >= 2.4 and best_overlap_count >= 2
        tfidf_gate = best_tfidf >= 0.24
        return overlap_gate or tfidf_gate
    except Exception:
        # Safe fallback: if enhanced gate fails for any reason, keep old behavior.
        return False


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

    Notes:
        If Gemini fails for any reason (missing API key, network issues, or invalid JSON),
        we automatically fall back to a deterministic Gemini-free classifier built from
        the curated 539-keyword vocabulary in `swachhata_keywords.json`.
    """
    try:
        client = _get_gemini_client()
        model_name = getattr(client, "_swachhata_model", "gemini-2.0-flash")
        full_prompt = f"{_CLASSIFICATION_PROMPT}\n\nComplaint: {text}"

        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt,
        )
        raw = response.text.strip()
    except Exception as exc:
        logger.warning("Gemini Swachhata classification failed; using fallback. Error: %s", exc)
        try:
            return classify_swachhata_fallback(text)
        except Exception as fallback_exc:
            # Preserve the original failure mode if fallback also errors.
            raise RuntimeError(f"Both Gemini and fallback Swachhata classification failed: {fallback_exc}") from exc

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
        logger.warning("Gemini returned non-JSON response; using fallback. Error: %s", exc)
        return classify_swachhata_fallback(text)

    # Validate and normalise label_id
    try:
        label_id = int(result["label_id"])
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Gemini response missing/invalid label_id; using fallback. Error: %s", exc)
        return classify_swachhata_fallback(text)

    if label_id not in SWACHHATA_CATEGORIES:
        logger.warning("Gemini returned invalid label_id=%s; using fallback.", label_id)
        return classify_swachhata_fallback(text)

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
        "isLLM": True,
    }
