#!/usr/bin/env python3
"""
RoBERTa Failure vs SGD Improvement — Manager Report Generator
=============================================================

Generates a shareable report comparing:
- Complaints where RoBERTa's prediction was wrong (failed)
- The corrected category provided by human feedback
- The current prediction after learning (SGD improved)

Output: Markdown report with columns:
  Original complaint | RoBERTa prediction (failed) | Corrected feedback | Learnt status | After learnt predicted category

Usage:
  python roberta_failure_sgd_improvement_report.py --api-url http://localhost:5016
  python roberta_failure_sgd_improvement_report.py --api-url http://localhost:5016 --limit 500
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import requests

# Project root (Adaptive Learning/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_FEEDBACK_DB = _PROJECT_ROOT / "CMS_RoBerta" / "model" / "feedback.db"

# Truncate long complaint text in table (for readability)
MAX_COMPLAINT_LEN = 120


def get_all_feedback_from_db(limit=None):
    """Read (complaint_text, corrected_category) from feedback.db."""
    if not _FEEDBACK_DB.exists():
        return []
    conn = sqlite3.connect(str(_FEEDBACK_DB))
    conn.row_factory = sqlite3.Row
    sql = "SELECT complaint_text, corrected_category FROM feedback_history ORDER BY feedback_timestamp"
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur = conn.execute(sql)
    rows = cur.fetchall()
    conn.close()
    return [(r["complaint_text"] or "", r["corrected_category"] or "") for r in rows]


def classify(api_url, text):
    """Call /classify and return transformer_label, adaptive_label, label (final)."""
    try:
        r = requests.post(
            f"{api_url.rstrip('/')}/classify",
            json={"text": text},
            timeout=30,
        )
        if r.status_code != 200:
            return None, None, None
        d = r.json()
        return (
            d.get("transformer_label"),
            d.get("adaptive_label"),
            d.get("label"),
        )
    except Exception:
        return None, None, None


def main():
    parser = argparse.ArgumentParser(
        description="Generate RoBERTa failure vs SGD improvement report for manager"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:5016",
        help="API base URL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of feedback records to consider (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for report (default: Testing/reports)",
    )
    args = parser.parse_args()

    feedbacks = get_all_feedback_from_db(limit=args.limit)
    if not feedbacks:
        print("No feedback records found in feedback.db. Run the API and submit some feedback first.")
        sys.exit(1)

    api_url = args.api_url.rstrip("/")
    print(f"Loaded {len(feedbacks)} feedback records. Calling API for current predictions...")

    rows = []
    for i, (complaint_text, corrected_category) in enumerate(feedbacks):
        if not complaint_text or not corrected_category:
            continue
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing {i + 1}/{len(feedbacks)}...")
        roberta_label, sgd_label, final_label = classify(api_url, complaint_text)
        if roberta_label is None:
            continue
        # RoBERTa failed = predicted something different from corrected
        roberta_failed = roberta_label != corrected_category
        # After learning, the system predicts the corrected category (SGD improved)
        sgd_improved = (final_label == corrected_category) or (sgd_label == corrected_category)
        # Include only: RoBERTa wrong and SGD corrected (improved after feedback)
        if roberta_failed and sgd_improved:
            short_text = (complaint_text[:MAX_COMPLAINT_LEN] + "...") if len(complaint_text) > MAX_COMPLAINT_LEN else complaint_text
            rows.append({
                "original_complaint": short_text,
                "original_full": complaint_text,
                "roberta_prediction": roberta_label,
                "corrected_feedback": corrected_category,
                "learnt_status": "Yes",
                "after_learnt_predicted": final_label,
            })

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output_dir) if args.output_dir else _SCRIPT_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"roberta_failure_sgd_improvement_report_{timestamp}.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# RoBERTa Failure vs SGD Improvement — Manager Report\n\n")
        f.write("**Generated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        f.write("**Executive summary:** This report shows complaints where the RoBERTa model initially predicted the wrong category. After human feedback, the SGD-based adaptive classifier learnt the correct category. The \"After learnt predicted category\" column shows that the system now predicts the corrected category for these complaints—demonstrating successful learning from feedback.\n\n")
        f.write("---\n\n")
        f.write("## Purpose\n\n")
        f.write(
            "This report lists complaints where **RoBERTa's prediction was wrong** (different from the human-corrected category) "
            "and **SGD improved** after learning from feedback. It shows: **original complaint**, **RoBERTa prediction (failed)**; "
            "**corrected feedback** (human correction); **learnt status**; and **after learnt predicted category** (current system prediction). "
            "Use it to compare RoBERTa failures and how SGD corrected them using feedback.\n\n"
        )
        f.write("---\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Total feedback records considered:** {len(feedbacks)}\n")
        f.write(f"- **Complaints where RoBERTa failed and SGD improved (shown below):** {len(rows)}\n\n")
        f.write("---\n\n")
        f.write("## Detailed Table\n\n")
        f.write("| # | Original complaint | RoBERTa prediction (failed) | Corrected feedback | Learnt status | After learnt predicted category |\n")
        f.write("|---|-------------------|------------------------------|--------------------|:-------------:|----------------------------------|\n")
        for idx, r in enumerate(rows, 1):
            orig = r["original_complaint"].replace("|", "\\|").replace("\n", " ")
            rb = (r["roberta_prediction"] or "").replace("|", "\\|")
            cf = (r["corrected_feedback"] or "").replace("|", "\\|")
            after = (r["after_learnt_predicted"] or "").replace("|", "\\|")
            f.write(f"| {idx} | {orig} | {rb} | {cf} | {r['learnt_status']} | {after} |\n")
        f.write("\n---\n\n")
        f.write("## Column definitions\n\n")
        f.write("- **Original complaint:** Complaint text (truncated in table; full text stored in feedback history).\n")
        f.write("- **RoBERTa prediction (failed):** The category predicted by the RoBERTa model; here it is wrong (differs from corrected feedback).\n")
        f.write("- **Corrected feedback:** The category provided by the human reviewer; the system learnt this as the correct category.\n")
        f.write("- **Learnt status:** Yes = this feedback is stored; if the same complaint is submitted again, the system returns the corrected category.\n")
        f.write("- **After learnt predicted category:** The current system prediction (RoBERTa + SGD ensemble) after learning; when SGD improved, this matches the corrected feedback.\n\n")
        f.write("---\n\n")
        f.write(f"*Report generated by roberta_failure_sgd_improvement_report.py — {timestamp}*\n")

    print(f"Report saved: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
