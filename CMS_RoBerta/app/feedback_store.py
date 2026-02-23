"""
Feedback history and category counts for adaptive routing.
- Don't ask for feedback twice for the same complaint (by hash).
- Track feedback count per category for adaptive ensemble weights.
"""
import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# DB next to model dir (or use env); default same parent as app
_DB_PATH: Optional[Path] = None


def _get_db_path() -> Path:
    global _DB_PATH
    if _DB_PATH is not None:
        return _DB_PATH
    base = Path(__file__).resolve().parent.parent
    _DB_PATH = base / "model" / "feedback.db"
    return _DB_PATH


def set_feedback_db_path(path: Path) -> None:
    global _DB_PATH
    _DB_PATH = Path(path)


def init_db() -> None:
    """Create tables if they don't exist."""
    path = _get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback_history (
            complaint_hash TEXT PRIMARY KEY,
            complaint_text TEXT NOT NULL,
            corrected_category TEXT NOT NULL,
            feedback_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS category_feedback_count (
            category TEXT PRIMARY KEY,
            feedback_count INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def _normalize_complaint_text(text: str) -> str:
    """Strip and collapse all whitespace so same complaint always matches."""
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"\s+", " ", t)  # collapse spaces, newlines, tabs to single space
    return t


def get_complaint_hash(text: str) -> str:
    """SHA256 hash of normalized complaint text (for dedup)."""
    normalized = _normalize_complaint_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def has_feedback(complaint_hash: str) -> bool:
    """True if we already have feedback for this complaint (don't ask again)."""
    return get_feedback_category(complaint_hash) is not None


def get_feedback_category(complaint_hash: str) -> Optional[str]:
    """Return corrected_category if we have feedback for this hash, else None."""
    conn = sqlite3.connect(str(_get_db_path()))
    cur = conn.execute(
        "SELECT corrected_category FROM feedback_history WHERE complaint_hash = ? LIMIT 1",
        (complaint_hash,),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def add_feedback(complaint_hash: str, complaint_text: str, corrected_category: str) -> None:
    """Record one feedback and increment category count. Idempotent per hash (INSERT OR IGNORE)."""
    conn = sqlite3.connect(str(_get_db_path()))
    conn.execute(
        """
        INSERT OR IGNORE INTO feedback_history (complaint_hash, complaint_text, corrected_category)
        VALUES (?, ?, ?)
        """,
        (complaint_hash, complaint_text[:10000], corrected_category),
    )
    conn.execute(
        """
        INSERT INTO category_feedback_count (category, feedback_count)
        VALUES (?, 1)
        ON CONFLICT(category) DO UPDATE SET feedback_count = feedback_count + 1
        """,
        (corrected_category,),
    )
    conn.commit()
    conn.close()


def add_feedback_batch(entries: list) -> None:
    """Record multiple feedbacks. entries: list of (complaint_hash, complaint_text, corrected_category)."""
    if not entries:
        return
    conn = sqlite3.connect(str(_get_db_path()))
    for complaint_hash, complaint_text, corrected_category in entries:
        conn.execute(
            """
            INSERT OR IGNORE INTO feedback_history (complaint_hash, complaint_text, corrected_category)
            VALUES (?, ?, ?)
            """,
            (complaint_hash, (complaint_text or "")[:10000], corrected_category),
        )
        conn.execute(
            """
            INSERT INTO category_feedback_count (category, feedback_count)
            VALUES (?, 1)
            ON CONFLICT(category) DO UPDATE SET feedback_count = feedback_count + 1
            """,
            (corrected_category,),
        )
    conn.commit()
    conn.close()


def get_category_feedback_counts() -> Dict[str, int]:
    """Return dict category -> feedback_count for adaptive weights."""
    conn = sqlite3.connect(str(_get_db_path()))
    cur = conn.execute("SELECT category, feedback_count FROM category_feedback_count")
    counts = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    return counts


def get_all_feedback() -> List[Tuple[str, str]]:
    """Return all (complaint_text, corrected_category) for offline SGD training (low-confidence cases only)."""
    conn = sqlite3.connect(str(_get_db_path()))
    cur = conn.execute(
        "SELECT complaint_text, corrected_category FROM feedback_history ORDER BY feedback_timestamp"
    )
    rows = cur.fetchall()
    conn.close()
    return [(row[0], row[1]) for row in rows]
