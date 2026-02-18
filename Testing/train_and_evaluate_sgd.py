#!/usr/bin/env python3
"""
Train SGD classifier using clean_samples.csv via /feedback API,
then evaluate and track metrics showing learning progress.

Usage:
    python3 train_and_evaluate_sgd.py [--api-url http://localhost:5015] [--batch-size 10] [--sample-limit 100]
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import requests


class SGDLearningTracker:
    """Track SGD classifier learning metrics and generate reports."""

    def __init__(self, api_url: str = "http://localhost:5015"):
        self.api_url = api_url.rstrip("/")
        self.stats = {
            "before_training": defaultdict(lambda: {"correct": 0, "total": 0, "confidences": []}),
            "after_training": defaultdict(lambda: {"correct": 0, "total": 0, "confidences": []}),
            "feedback_sent": defaultdict(int),
            "similar_complaints": defaultdict(list),  # category -> list of (text, before_conf, after_conf)
        }
        self.all_predictions = []  # for detailed report

    def classify(self, text: str) -> Dict:
        """Call /classify API."""
        try:
            resp = requests.post(
                f"{self.api_url}/classify",
                json={"text": text},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Error classifying: {e}")
            return {}

    def send_feedback(self, complaint_text: str, correct_category: str) -> bool:
        """Call /feedback API."""
        try:
            resp = requests.post(
                f"{self.api_url}/feedback",
                json={
                    "complaint_text": complaint_text,
                    "correct_category": correct_category,
                },
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("status") == "learned"
        except requests.exceptions.HTTPError as e:
            # Get detailed error message from response
            try:
                error_detail = resp.json().get("detail", str(e))
                print(f"Error sending feedback: {resp.status_code} - {error_detail}")
            except:
                print(f"Error sending feedback: {resp.status_code} - {e}")
            return False
        except Exception as e:
            print(f"Error sending feedback: {e}")
            return False

    def process_sample(
        self,
        complaint: str,
        correct_category: str,
        phase: str = "before",
    ) -> Dict:
        """Classify a complaint and record metrics."""
        result = self.classify(complaint)
        if not result:
            return {}

        predicted = result.get("label", "")
        confidence = result.get("confidence", 0.0)
        is_correct = predicted == correct_category

        self.stats[f"{phase}_training"][correct_category]["total"] += 1
        if is_correct:
            self.stats[f"{phase}_training"][correct_category]["correct"] += 1
        self.stats[f"{phase}_training"][correct_category]["confidences"].append(confidence)

        return {
            "complaint": complaint[:100] + "..." if len(complaint) > 100 else complaint,
            "correct_category": correct_category,
            "predicted": predicted,
            "confidence": confidence,
            "is_correct": is_correct,
            "phase": phase,
        }

    def train_and_evaluate(
        self,
        csv_path: Path,
        batch_size: int = 10,
        sample_limit: int = None,
        test_after_every: int = 50,
    ):
        """
        Train SGD on clean_samples.csv and evaluate progress.

        Args:
            csv_path: Path to clean_samples.csv
            batch_size: Send feedback in batches (with small delay)
            sample_limit: Limit number of samples (None = all)
            test_after_every: Re-evaluate on first N samples every N feedbacks
        """
        print(f"Loading samples from {csv_path}...")
        samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append((row["complaint"], row["category"]))

        if sample_limit:
            samples = samples[:sample_limit]
        print(f"Processing {len(samples)} samples...")

        # Test set: first 100 samples (or 10% if more than 1000)
        test_size = min(100, max(10, len(samples) // 10))
        test_samples = samples[:test_size]
        train_samples = samples[test_size:]

        print(f"\n=== Phase 1: Baseline evaluation (before training) ===")
        print(f"Evaluating on {len(test_samples)} test samples...")
        for complaint, category in test_samples:
            self.process_sample(complaint, category, phase="before")
            time.sleep(0.1)  # small delay

        # Training loop
        print(f"\n=== Phase 2: Training SGD classifier ===")
        print(f"Training on {len(train_samples)} samples...")
        trained_count = 0
        for i, (complaint, category) in enumerate(train_samples, 1):
            # Classify before feedback (to see what it would predict)
            before = self.process_sample(complaint, category, phase="before")

            # Send feedback
            if self.send_feedback(complaint, category):
                trained_count += 1
                self.stats["feedback_sent"][category] += 1

            # Track similar complaints (group by category, store first 5)
            if len(self.stats["similar_complaints"][category]) < 5:
                self.stats["similar_complaints"][category].append(
                    (complaint[:150], before.get("confidence", 0.0))
                )

            # Progress update
            if i % batch_size == 0:
                print(f"  Trained {i}/{len(train_samples)} samples... ({trained_count} successful)")
                time.sleep(0.5)  # small delay between batches

            # Re-evaluate test set periodically
            if i % test_after_every == 0 and i > 0:
                print(f"\n  Re-evaluating test set after {i} training samples...")
                for test_complaint, test_category in test_samples[:20]:  # quick check on first 20
                    self.process_sample(test_complaint, test_category, phase="after")
                time.sleep(1)

        # Final evaluation
        print(f"\n=== Phase 3: Final evaluation (after training) ===")
        print(f"Re-evaluating on {len(test_samples)} test samples...")
        for complaint, category in test_samples:
            self.process_sample(complaint, category, phase="after")
            time.sleep(0.1)

        print(f"\nTraining complete! Sent {trained_count} feedback samples.")
        return self.generate_report()

    def generate_report(self) -> Dict:
        """Generate metrics report."""
        report = {
            "categories": {},
            "summary": {},
        }

        all_categories = set(
            list(self.stats["before_training"].keys())
            + list(self.stats["after_training"].keys())
        )

        for category in sorted(all_categories):
            before = self.stats["before_training"][category]
            after = self.stats["after_training"][category]

            before_acc = (
                before["correct"] / before["total"] * 100 if before["total"] > 0 else 0
            )
            after_acc = (
                after["correct"] / after["total"] * 100 if after["total"] > 0 else 0
            )
            improvement = after_acc - before_acc

            before_conf = (
                sum(before["confidences"]) / len(before["confidences"])
                if before["confidences"]
                else 0
            )
            after_conf = (
                sum(after["confidences"]) / len(after["confidences"])
                if after["confidences"]
                else 0
            )

            report["categories"][category] = {
                "before_accuracy": round(before_acc, 2),
                "after_accuracy": round(after_acc, 2),
                "improvement": round(improvement, 2),
                "before_avg_confidence": round(before_conf, 4),
                "after_avg_confidence": round(after_conf, 4),
                "feedback_samples": self.stats["feedback_sent"][category],
                "test_samples": before["total"],
                "similar_complaints": self.stats["similar_complaints"][category],
            }

        # Overall summary
        total_before_correct = sum(
            s["correct"] for s in self.stats["before_training"].values()
        )
        total_before_total = sum(
            s["total"] for s in self.stats["before_training"].values()
        )
        total_after_correct = sum(
            s["correct"] for s in self.stats["after_training"].values()
        )
        total_after_total = sum(
            s["total"] for s in self.stats["after_training"].values()
        )

        report["summary"] = {
            "overall_before_accuracy": round(
                total_before_correct / total_before_total * 100
                if total_before_total > 0
                else 0,
                2,
            ),
            "overall_after_accuracy": round(
                total_after_correct / total_after_total * 100
                if total_after_total > 0
                else 0,
                2,
            ),
            "total_feedback_samples": sum(self.stats["feedback_sent"].values()),
            "total_test_samples": total_before_total,
        }

        return report


def generate_markdown_report(report: Dict, output_path: Path):
    """Generate a markdown report with tables and learning patterns."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# SGD Classifier Learning Metrics Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Overall Accuracy (Before Training):** {report['summary']['overall_before_accuracy']}%\n")
        f.write(f"- **Overall Accuracy (After Training):** {report['summary']['overall_after_accuracy']}%\n")
        f.write(f"- **Improvement:** {report['summary']['overall_after_accuracy'] - report['summary']['overall_before_accuracy']:.2f} percentage points\n")
        f.write(f"- **Total Feedback Samples Sent:** {report['summary']['total_feedback_samples']}\n")
        f.write(f"- **Test Samples Evaluated:** {report['summary']['total_test_samples']}\n\n")

        # Accuracy table
        f.write("## Per-Category Accuracy Metrics\n\n")
        f.write("| Category | Before Accuracy | After Accuracy | Improvement | Feedback Samples | Test Samples |\n")
        f.write("|----------|----------------|---------------|-------------|------------------|--------------|\n")

        for category, metrics in sorted(
            report["categories"].items(),
            key=lambda x: x[1]["improvement"],
            reverse=True,
        ):
            f.write(
                f"| {category} | {metrics['before_accuracy']}% | "
                f"{metrics['after_accuracy']}% | {metrics['improvement']:+.2f}% | "
                f"{metrics['feedback_samples']} | {metrics['test_samples']} |\n"
            )

        # Learning patterns - similar complaints
        f.write("\n## Learning Patterns: Similar Complaints\n\n")
        f.write(
            "This section shows how the SGD classifier learned from similar complaints "
            "within each category. The confidence scores show the model's certainty "
            "before training (baseline).\n\n"
        )

        for category, metrics in sorted(report["categories"].items()):
            if metrics["similar_complaints"]:
                f.write(f"### {category}\n\n")
                f.write(f"**Feedback Samples:** {metrics['feedback_samples']}  \n")
                f.write(f"**Accuracy Improvement:** {metrics['improvement']:+.2f}%  \n\n")
                f.write("**Sample Complaints Used for Training:**\n\n")
                for i, (complaint, conf) in enumerate(metrics["similar_complaints"], 1):
                    f.write(f"{i}. *{complaint}...*  \n")
                    f.write(f"   - Initial confidence: {conf:.2%}\n\n")
                f.write("\n")

        # Confidence analysis
        f.write("## Average Confidence Analysis\n\n")
        f.write("Average ensemble confidence (0–1) on test samples **before** vs **after** the SGD classifier was trained with feedback.\n\n")
        f.write("| Category | Avg Confidence (Before Feedback Training) | Avg Confidence (After Feedback Training) | Change |\n")
        f.write("|----------|------------------------------------------|------------------------------------------|--------|\n")

        for category, metrics in sorted(report["categories"].items()):
            conf_change = metrics["after_avg_confidence"] - metrics["before_avg_confidence"]
            f.write(
                f"| {category} | {metrics['before_avg_confidence']:.4f} | "
                f"{metrics['after_avg_confidence']:.4f} | {conf_change:+.4f} |\n"
            )

        f.write("\n---\n")
        f.write("*Report generated by Shubham Kambale*\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train SGD classifier and generate learning metrics"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:5015",
        help="API base URL (default: http://localhost:5015)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Feedback batch size (default: 10)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit number of samples (default: all)",
    )
    parser.add_argument(
        "--csv",
        default="clean_samples.csv",
        help="Input CSV file (default: clean_samples.csv)",
    )
    parser.add_argument(
        "--output",
        default="sgd_learning_metrics.md",
        help="Output markdown report (default: sgd_learning_metrics.md)",
    )

    args = parser.parse_args()

    csv_path = Path(__file__).parent / args.csv
    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        return

    tracker = SGDLearningTracker(api_url=args.api_url)

    # Check API is running
    try:
        resp = requests.get(f"{args.api_url}/health", timeout=5)
        resp.raise_for_status()
        print(f"✓ API is running at {args.api_url}\n")
    except Exception as e:
        print(f"Error: API not reachable at {args.api_url}")
        print(f"  {e}")
        print("\nPlease start the API first:")
        print("  python3 -m uvicorn CMS_RoBerta.app.main:app --host 0.0.0.0 --port 5015")
        return

    # Train and evaluate
    report = tracker.train_and_evaluate(
        csv_path=csv_path,
        batch_size=args.batch_size,
        sample_limit=args.sample_limit,
    )

    # Save JSON report
    json_path = Path(__file__).parent / "sgd_learning_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved JSON report: {json_path}")

    # Generate markdown report
    md_path = Path(__file__).parent / args.output
    generate_markdown_report(report, md_path)
    print(f"✓ Saved markdown report: {md_path}")


if __name__ == "__main__":
    main()
