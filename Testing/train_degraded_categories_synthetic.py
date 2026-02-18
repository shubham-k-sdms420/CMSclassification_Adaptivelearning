#!/usr/bin/env python3
"""
Send synthetic_pmc_data.csv to the /feedback API to train the SGD classifier on the 7 degraded
categories, then evaluate before/after accuracy and append results to sgd_learning_metrics.md.
"""
import argparse
import csv
import random
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import requests

API_URL = "http://localhost:5015"
SYNTHETIC_CSV = "synthetic_pmc_data.csv"
METRICS_MD = "sgd_learning_metrics.md"
EVAL_SAMPLES_PER_CAT = 10
DEFAULT_BATCH_SIZE = 100
DEFAULT_EPOCHS = 3

DEGRADED_CATEGORIES = [
    "RTI Related Complaint",
    "TDR",
    "Abhyagat Kaksha",
    "Environment (H.O)",
    "Garbage Depot Complaint",
    "PMC Security Complaints",
    "Unauthorized slaughterhouse / crude meat",
]


def classify(text: str) -> dict:
    try:
        r = requests.post(f"{API_URL}/classify", json={"text": text}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Classify error: {e}")
        return {}


def send_feedback(complaint_text: str, correct_category: str) -> bool:
    """Send single feedback (kept for compatibility)."""
    try:
        r = requests.post(
            f"{API_URL}/feedback",
            json={"complaint_text": complaint_text, "correct_category": correct_category},
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("status") == "learned"
    except Exception as e:
        print(f"Feedback error: {e}")
        return False


def send_batch_feedback(feedbacks: list, batch_size: int = 100) -> int:
    """Send feedback in batches for better SGDClassifier learning efficiency.

    Args:
        feedbacks: List of dicts with 'complaint_text' and 'correct_category'
        batch_size: Number of samples per batch (larger = stronger gradient per update)

    Returns:
        Number of successfully sent samples
    """
    sent = 0
    for i in range(0, len(feedbacks), batch_size):
        batch = feedbacks[i:i + batch_size]
        try:
            r = requests.post(
                f"{API_URL}/feedback/batch",
                json={"feedbacks": batch},
                timeout=60,
            )
            r.raise_for_status()
            result = r.json()
            if result.get("status") == "learned":
                sent += len(batch)
                print(f"  Sent batch {i//batch_size + 1}: {len(batch)} samples (total: {sent}/{len(feedbacks)})")
            else:
                print(f"  Warning: Batch {i//batch_size + 1} returned unexpected status: {result}")
        except Exception as e:
            print(f"  Error sending batch {i//batch_size + 1}: {e}")
        time.sleep(0.1)
    return sent


def load_synthetic(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            cat = (row.get("category") or "").strip()
            if text and cat:
                rows.append((text, cat))
    return rows


def evaluate_samples(samples_by_cat: dict) -> dict:
    """Returns {category: {"correct": n, "total": m, "accuracy": pct}}"""
    result = defaultdict(lambda: {"correct": 0, "total": 0})
    for cat, texts in samples_by_cat.items():
        for text in texts:
            pred = classify(text)
            label = pred.get("label", "")
            result[cat]["total"] += 1
            if label == cat:
                result[cat]["correct"] += 1
            time.sleep(0.05)
    for cat in result:
        t = result[cat]["total"]
        c = result[cat]["correct"]
        result[cat]["accuracy"] = round(c / t * 100, 2) if t else 0
    return dict(result)


def main():
    parser = argparse.ArgumentParser(description="Train SGD on synthetic data and report accuracy change.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of passes over synthetic data (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Samples per batch (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling each epoch")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    csv_path = base / SYNTHETIC_CSV
    md_path = base / METRICS_MD

    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"API not reachable at {API_URL}: {e}")
        print("Start the API first: python3 -m uvicorn CMS_RoBerta.app.main:app --host 0.0.0.0 --port 5015")
        return

    rows = load_synthetic(csv_path)
    if not rows:
        print("No rows in synthetic CSV.")
        return

    # Build list per category and pick evaluation samples (first N per category)
    by_cat = defaultdict(list)
    for text, cat in rows:
        by_cat[cat].append(text)

    eval_samples = {}
    for cat in DEGRADED_CATEGORIES:
        eval_samples[cat] = by_cat[cat][:EVAL_SAMPLES_PER_CAT]

    print("=== Before: Evaluating on synthetic samples (first 10 per category) ===")
    before = evaluate_samples(eval_samples)

    total_sent = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs}: Sending synthetic data (batch size: {args.batch_size}) ===")
        feedbacks = [{"complaint_text": text, "correct_category": cat} for text, cat in rows]
        random.seed(args.seed + epoch)
        random.shuffle(feedbacks)
        sent = send_batch_feedback(feedbacks, batch_size=args.batch_size)
        total_sent += sent
    print(f"  Total feedback samples sent (all epochs): {total_sent} (={len(rows)} x {args.epochs} epochs)")

    print("\n=== After: Re-evaluating on same samples ===")
    after = evaluate_samples(eval_samples)

    # Build new section for markdown
    lines = [
        "",
        "---",
        "",
        "## Accuracy improvement on degraded categories (synthetic data feedback)",
        "",
        f"After sending **{total_sent}** feedback updates ({len(rows)} samples × {args.epochs} epochs) from `synthetic_pmc_data.csv` to the feedback API (batch size: {args.batch_size}), the 7 previously degraded categories were re-evaluated on 10 samples per category (70 samples total).",
        "",
        "| Category | Accuracy (before synthetic feedback) | Accuracy (after synthetic feedback) | Change |",
        "|----------|--------------------------------------|--------------------------------------|--------|",
    ]
    total_before_correct = total_after_correct = total_n = 0
    for cat in DEGRADED_CATEGORIES:
        b = before.get(cat, {})
        a = after.get(cat, {})
        before_acc = b.get("accuracy", 0)
        after_acc = a.get("accuracy", 0)
        total_n += b.get("total", 0)
        total_before_correct += b.get("correct", 0)
        total_after_correct += a.get("correct", 0)
        ch = after_acc - before_acc
        ch_str = f"{ch:+.2f}%" if ch != 0 else "0%"
        lines.append(f"| {cat} | {before_acc}% | {after_acc}% | {ch_str} |")

    overall_before = round(total_before_correct / total_n * 100, 2) if total_n else 0
    overall_after = round(total_after_correct / total_n * 100, 2) if total_n else 0
    overall_change = overall_after - overall_before

    lines.extend([
        "",
        f"**Overall (70 samples):** Before = {overall_before}%, After = {overall_after}%, **Change = {overall_change:+.2f}%**",
        "",
        "**Conclusion:** " + (
            "Accuracy **increased** on the degraded categories after training with synthetic data."
            if overall_change > 0 else
            "Accuracy **did not increase** on the degraded categories after this synthetic feedback run; consider more epochs, larger batches, or ENSEMBLE_ADAPTIVE_WEIGHT (see below)."
            if overall_change < 0 else
            "Accuracy remained **unchanged** on the evaluated samples."
        ),
        "",
        "### How to improve accuracy further",
        "",
        "1. **Multiple epochs**: Run with more passes, e.g. `python3 train_degraded_categories_synthetic.py --epochs 5`.",
        "2. **Larger batches**: Use `--batch-size 200` so each SGD update sees more samples.",
        "3. **Give SGD more influence**: The ensemble uses 70% transformer / 30% SGD. When they disagree, the transformer often wins. Start the API with a higher adaptive weight so SGD feedback matters more:",
        "   ```bash",
        "   ENSEMBLE_ADAPTIVE_WEIGHT=0.45 ENSEMBLE_TRANSFORMER_WEIGHT=0.55 python3 -m uvicorn CMS_RoBerta.app.main:app --host 0.0.0.0 --port 5015",
        "   ```",
        "   Then re-run this script and re-evaluate.",
        "4. **More synthetic data**: Add more diverse examples per degraded category (e.g. 200+ per category).",
        "",
        f"*Section generated by train_degraded_categories_synthetic.py on {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
    ])

    # Append to metrics file (before the final --- and report line)
    if md_path.exists():
        content = md_path.read_text(encoding="utf-8")
        # Insert before last "---" and "*Report generated"
        if "---\n*Report generated" in content:
            content = content.replace(
                "\n---\n*Report generated by Shubham Kambale*",
                "\n" + "\n".join(lines) + "\n---\n*Report generated by Shubham Kambale*",
            )
        else:
            content = content.rstrip() + "\n" + "\n".join(lines) + "\n"
        md_path.write_text(content, encoding="utf-8")
        print(f"\nAppended section to {md_path}")
    else:
        section_path = base / "accuracy_improvement_degraded_categories.md"
        section_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nMetrics file not found; wrote section to {section_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
