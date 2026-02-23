#!/usr/bin/env python3
"""
Automated Feedback Simulation Script for PMC SGD Classifier
=============================================================

This script simulates the online learning feedback loop by:
1. Making predictions via /classify endpoint
2. Automatically providing "feedback" via /feedback endpoint
3. Tracking accuracy improvement over time
4. Generating learning curves and reports

This tests the incremental learning capability without manual UI interaction.

Usage:
    python feedback_simulator.py --api-url http://localhost:5015 --csv training_samples_sgd_7800.csv
"""

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


class FeedbackSimulator:
    """Simulates automated feedback loop for SGD training"""

    def __init__(self, api_url, csv_path, batch_size=10, delay=0.1, feedback_all=False, category_map_path=None, categories_file=None):
        self.api_url = api_url.rstrip("/")
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.delay = delay  # Delay between requests (seconds)
        self.feedback_all = feedback_all  # If True, send feedback for every record (train on all)
        self.category_map_path = category_map_path  # CSV: from_category,to_category (map to API labels)
        self.categories_file = categories_file  # If set, only train on these categories (one per line, CSV names)

        # Metrics tracking
        self.metrics = {
            "feedbacks_given": 0,
            "predictions_correct": 0,
            "predictions_total": 0,
            "accuracy_over_time": [],
            "category_accuracy": defaultdict(lambda: {"correct": 0, "total": 0}),
            "confidence_distribution": [],
            "learning_curve": [],
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "initial_accuracy": None,
            "final_accuracy": None,
            "improvement": None,
        }

        # Load data
        self.df = None
        self.load_data()

    def load_data(self):
        """Load training/feedback data"""
        print("=" * 80)
        print("LOADING FEEDBACK DATA")
        print("=" * 80)

        self.df = pd.read_csv(self.csv_path)

        # Check required columns
        if "complaint_text" not in self.df.columns and "text" in self.df.columns:
            self.df["complaint_text"] = self.df["text"]
        if "category" not in self.df.columns and "label" in self.df.columns:
            self.df["category"] = self.df["label"]

        if "complaint_text" not in self.df.columns or "category" not in self.df.columns:
            raise ValueError("CSV must have 'complaint_text' (or 'text') and 'category' (or 'label') columns")

        # Optionally keep only rows for selected categories (e.g. weak categories only)
        if self.categories_file:
            path = Path(self.categories_file)
            if path.exists():
                allowed = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
                self.df = self.df[self.df["category"].astype(str).str.strip().isin(allowed)].reset_index(drop=True)
                print(f"Filtered to {len(self.df):,} rows in {len(allowed)} categories (weak-categories only)")
                if len(self.df) == 0:
                    raise ValueError("No rows left after category filter. Check category names in " + self.categories_file)
            else:
                print("Categories file not found:", self.categories_file)

        # Apply category mapping so CSV labels match API's 78 labels (fixes false 0% categories)
        if self.category_map_path:
            map_df = pd.read_csv(self.category_map_path)
            if "from_category" in map_df.columns and "to_category" in map_df.columns:
                category_map = dict(zip(map_df["from_category"].str.strip(), map_df["to_category"].str.strip()))
                self.df["category"] = self.df["category"].astype(str).str.strip().replace(category_map)
                print(f"Applied category mapping from {self.category_map_path} ({len(category_map)} rules)")
            else:
                print("Category map CSV must have columns: from_category, to_category")

        # Shuffle for realistic feedback order
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Loaded {len(self.df):,} samples")
        print(f"Categories: {self.df['category'].nunique()}")
        print(f"Batch size: {self.batch_size}")
        print(f"Total batches: {len(self.df) // self.batch_size}")

    def test_api_connection(self):
        """Test if API is reachable"""
        print("\n" + "=" * 80)
        print("TESTING API CONNECTION")
        print("=" * 80)

        try:
            # Test classify endpoint
            # First request can be slow while models load (RoBERTa + SGD)
            response = requests.post(
                f"{self.api_url}/classify",
                json={"text": "Test connection"},
                timeout=60,
            )

            if response.status_code == 200:
                print(f"API is reachable at {self.api_url}")
                return True
            else:
                print(f"API returned status {response.status_code}")
                return False

        except requests.exceptions.ConnectionError:
            print(f"Cannot connect to {self.api_url}")
            print("  Make sure your API is running!")
            return False
        except requests.exceptions.ReadTimeout:
            print(f"Connection to {self.api_url} timed out (waited 60s).")
            print("  The API may still be loading models (RoBERTa + SGD). Try again in a minute.")
            print("  Or start the API first, wait until it logs 'Uvicorn running', then run this script.")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False

    def classify_complaint(self, complaint_text):
        """Send complaint to /classify endpoint"""
        try:
            # Allow time for inference (transformer + adaptive classifier)
            response = requests.post(
                f"{self.api_url}/classify",
                json={"text": complaint_text},
                timeout=30,
            )

            if response.status_code == 200:
                return response.json()
            else:
                return None

        except Exception as e:
            print(f"  Error classifying: {e}")
            return None

    def send_feedback(self, complaint_text, correct_category):
        """Send feedback to /feedback endpoint"""
        try:
            payload = {
                "complaint_text": complaint_text,
                "correct_category": correct_category,
            }

            # Allow time for partial_fit + model save
            response = requests.post(
                f"{self.api_url}/feedback",
                json=payload,
                timeout=30,
            )

            return response.status_code == 200

        except Exception as e:
            print(f"  Error sending feedback: {e}")
            return False

    def evaluate_batch(self, batch_df):
        """Evaluate current model performance on a batch"""
        correct = 0
        total = len(batch_df)

        for _, row in batch_df.iterrows():
            complaint = row.get("complaint_text") or row.get("text", "")
            result = self.classify_complaint(complaint)

            if result:
                predicted = result.get("label")
                true_label = row.get("category") or row.get("label", "")
                if predicted == true_label:
                    correct += 1

                # Track confidence
                confidence = result.get("confidence", 0)
                self.metrics["confidence_distribution"].append(
                    {"confidence": confidence, "correct": predicted == true_label}
                )

            time.sleep(self.delay / 2)  # Small delay

        accuracy = correct / total if total > 0 else 0
        return accuracy, correct, total

    def run_feedback_loop(self, max_samples=None):
        """Main feedback simulation loop"""
        print("\n" + "=" * 80)
        print("STARTING FEEDBACK SIMULATION")
        print("=" * 80)

        if not self.test_api_connection():
            print("\nCannot proceed without API connection!")
            return

        # Limit samples if specified
        if max_samples:
            self.df = self.df.head(max_samples)

        total_samples = len(self.df)

        # Initial evaluation (before any feedback)
        print("\nInitial Model Performance (before feedback)...")
        eval_sample = self.df.sample(min(200, len(self.df)), random_state=42)
        initial_acc, _, _ = self.evaluate_batch(eval_sample)
        print(f"   Baseline Accuracy: {initial_acc * 100:.2f}%")

        self.metrics["learning_curve"].append(
            {"feedback_count": 0, "accuracy": initial_acc, "timestamp": time.time()}
        )

        # Process in batches
        print(f"\nProcessing {total_samples:,} samples in batches of {self.batch_size}...")

        batches = [
            self.df[i : i + self.batch_size]
            for i in range(0, len(self.df), self.batch_size)
        ]

        for batch_idx, batch_df in enumerate(tqdm(batches, desc="Training Progress")):
            # Process each complaint in batch
            for _, row in batch_df.iterrows():
                complaint = row.get("complaint_text") or row.get("text", "")
                correct_category = row.get("category") or row.get("label", "")

                if not complaint or not correct_category:
                    continue

                # 1. Get prediction
                result = self.classify_complaint(complaint)

                if result:
                    predicted = result.get("label")
                    confidence = result.get("confidence", 0)

                    # Track accuracy
                    self.metrics["predictions_total"] += 1
                    is_correct = predicted == correct_category

                    if is_correct:
                        self.metrics["predictions_correct"] += 1

                    # Track per-category accuracy
                    self.metrics["category_accuracy"][correct_category]["total"] += 1
                    if is_correct:
                        self.metrics["category_accuracy"][correct_category]["correct"] += 1

                    # 2. Provide feedback (simulate human correction)
                    # With --feedback-all: send feedback for every record (all 7,800 used for training).
                    # Otherwise: only wrong predictions or confidence < 0.85 (realistic simulation).
                    if self.feedback_all or not is_correct or confidence < 0.85:
                        feedback_sent = self.send_feedback(complaint, correct_category)

                        if feedback_sent:
                            self.metrics["feedbacks_given"] += 1

                time.sleep(self.delay)

            # Evaluate every N batches
            if (batch_idx + 1) % 10 == 0:
                eval_sample = self.df.sample(min(200, len(self.df)), random_state=42)
                current_acc, _, _ = self.evaluate_batch(eval_sample)

                self.metrics["learning_curve"].append(
                    {
                        "feedback_count": self.metrics["feedbacks_given"],
                        "accuracy": current_acc,
                        "timestamp": time.time(),
                    }
                )

                # Progress update
                print(
                    f"\n  Batch {batch_idx + 1}/{len(batches)}: "
                    f"Accuracy={current_acc*100:.2f}%, "
                    f"Feedbacks={self.metrics['feedbacks_given']}"
                )

        # Final evaluation
        print("\nFinal Model Performance (after feedback)...")
        eval_sample = self.df.sample(min(500, len(self.df)), random_state=42)
        final_acc, _, _ = self.evaluate_batch(eval_sample)
        print(f"   Final Accuracy: {final_acc * 100:.2f}%")

        self.metrics["learning_curve"].append(
            {
                "feedback_count": self.metrics["feedbacks_given"],
                "accuracy": final_acc,
                "timestamp": time.time(),
            }
        )

        self.metrics["final_accuracy"] = final_acc
        self.metrics["initial_accuracy"] = initial_acc
        self.metrics["improvement"] = final_acc - initial_acc

    def generate_plots(self, output_dir="plots"):
        """Generate visualization plots"""
        print("\n" + "=" * 80)
        print("GENERATING PLOTS")
        print("=" * 80)

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 1. Learning Curve
        if len(self.metrics["learning_curve"]) > 1:
            plt.figure(figsize=(12, 6))

            feedbacks = [
                point["feedback_count"] for point in self.metrics["learning_curve"]
            ]
            accuracies = [
                point["accuracy"] * 100 for point in self.metrics["learning_curve"]
            ]

            plt.plot(feedbacks, accuracies, marker="o", linewidth=2, markersize=8)
            plt.xlabel("Number of Feedbacks", fontsize=12)
            plt.ylabel("Accuracy (%)", fontsize=12)
            plt.title(
                "SGD Learning Curve: Accuracy Improvement Over Time",
                fontsize=14,
                fontweight="bold",
            )
            plt.grid(True, alpha=0.3)

            # Annotate start and end
            if len(accuracies) > 0:
                plt.annotate(
                    f"Start: {accuracies[0]:.1f}%",
                    xy=(feedbacks[0], accuracies[0]),
                    xytext=(10, -20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )

                plt.annotate(
                    f"End: {accuracies[-1]:.1f}%",
                    xy=(feedbacks[-1], accuracies[-1]),
                    xytext=(-60, 20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )

            plot_path = output_path / f"learning_curve_{self.metrics['timestamp']}.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Learning curve saved: {plot_path}")

        # 2. Confidence Distribution
        if self.metrics["confidence_distribution"]:
            plt.figure(figsize=(12, 6))

            correct_confs = [
                item["confidence"]
                for item in self.metrics["confidence_distribution"]
                if item["correct"]
            ]
            wrong_confs = [
                item["confidence"]
                for item in self.metrics["confidence_distribution"]
                if not item["correct"]
            ]

            if correct_confs:
                plt.hist(
                    correct_confs,
                    bins=20,
                    alpha=0.7,
                    label="Correct Predictions",
                    color="green",
                )
            if wrong_confs:
                plt.hist(
                    wrong_confs,
                    bins=20,
                    alpha=0.7,
                    label="Wrong Predictions",
                    color="red",
                )

            plt.xlabel("Confidence", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.title("Prediction Confidence Distribution", fontsize=14, fontweight="bold")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plot_path = output_path / f"confidence_dist_{self.metrics['timestamp']}.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Confidence distribution saved: {plot_path}")

    def generate_report(self, output_dir="reports"):
        """Generate markdown report"""
        print("\n" + "=" * 80)
        print("GENERATING REPORT")
        print("=" * 80)

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        report_file = output_path / f"feedback_simulation_report_{self.metrics['timestamp']}.md"

        with open(report_file, "w", encoding="utf-8") as f:
            # Header
            f.write(
                "# PMC SGD Classifier - Automated Feedback Simulation Report\n\n"
            )
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Samples Processed:** {self.metrics['predictions_total']:,}\n")
            f.write(f"- **Feedbacks Provided:** {self.metrics['feedbacks_given']:,}\n")
            
            # Handle cases where accuracy metrics might not be set
            initial_acc = self.metrics.get('initial_accuracy')
            final_acc = self.metrics.get('final_accuracy')
            improvement = self.metrics.get('improvement')
            
            if initial_acc is not None:
                f.write(f"- **Initial Accuracy:** {initial_acc * 100:.2f}%\n")
            else:
                f.write("- **Initial Accuracy:** N/A (simulation did not complete)\n")
            
            if final_acc is not None:
                f.write(f"- **Final Accuracy:** {final_acc * 100:.2f}%\n")
            else:
                f.write("- **Final Accuracy:** N/A (simulation did not complete)\n")
            
            if improvement is not None:
                f.write(f"- **Improvement:** {improvement * 100:+.2f}%\n")
            else:
                f.write("- **Improvement:** N/A (simulation did not complete)\n")
            
            f.write(f"- **API Endpoint:** {self.api_url}\n\n")

            f.write("---\n\n")

            # Learning Progress
            f.write("## Learning Progress\n\n")
            f.write("| Feedbacks Given | Accuracy | Improvement |\n")
            f.write("|----------------|----------|-------------|\n")

            for i, point in enumerate(self.metrics["learning_curve"]):
                if i == 0:
                    improvement = "Baseline"
                else:
                    prev_acc = self.metrics["learning_curve"][i - 1]["accuracy"]
                    improvement = f"{(point['accuracy'] - prev_acc) * 100:+.2f}%"

                f.write(
                    f"| {point['feedback_count']:,} | {point['accuracy'] * 100:.2f}% | {improvement} |\n"
                )

            f.write("\n")

            # Category-wise Performance
            f.write("## Category-wise Performance\n\n")

            category_stats = []
            for category, stats in self.metrics["category_accuracy"].items():
                if stats["total"] > 0:
                    accuracy = stats["correct"] / stats["total"]
                    category_stats.append((category, accuracy, stats["total"]))

            # Sort by accuracy
            category_stats.sort(key=lambda x: x[1])

            f.write("### Bottom 20 Categories (Need Attention)\n\n")
            f.write("| Category | Accuracy | Samples |\n")
            f.write("|----------|----------|--------|\n")

            for cat, acc, total in category_stats[:20]:
                f.write(f"| {cat[:40]} | {acc * 100:.1f}% | {total} |\n")

            f.write("\n### Top 20 Categories (Strong Performance)\n\n")
            f.write("| Category | Accuracy | Samples |\n")
            f.write("|----------|----------|--------|\n")

            for cat, acc, total in reversed(category_stats[-20:]):
                f.write(f"| {cat[:40]} | {acc * 100:.1f}% | {total} |\n")

            f.write("\n")

            # Confidence Analysis
            if self.metrics["confidence_distribution"]:
                f.write("## Confidence Analysis\n\n")

                correct_confs = [
                    item["confidence"]
                    for item in self.metrics["confidence_distribution"]
                    if item["correct"]
                ]
                wrong_confs = [
                    item["confidence"]
                    for item in self.metrics["confidence_distribution"]
                    if not item["correct"]
                ]

                if correct_confs:
                    f.write(f"- **Correct Predictions:**\n")
                    f.write(f"  - Average Confidence: {np.mean(correct_confs) * 100:.2f}%\n")
                    f.write(f"  - Median Confidence: {np.median(correct_confs) * 100:.2f}%\n")
                    f.write(f"\n")
                if wrong_confs:
                    f.write(f"- **Wrong Predictions:**\n")
                    f.write(f"  - Average Confidence: {np.mean(wrong_confs) * 100:.2f}%\n")
                    f.write(f"  - Median Confidence: {np.median(wrong_confs) * 100:.2f}%\n")
                    f.write(f"\n")

            # Recommendations
            f.write("## Key Insights\n\n")

            improvement_val = self.metrics.get("improvement")
            if improvement_val is None:
                f.write("⚠️ **Simulation Incomplete:** The simulation did not complete successfully. Check API connection and logs.\n\n")
            else:
                improvement = improvement_val * 100

                if improvement > 10:
                    f.write(
                        "✅ **Excellent Learning:** Model improved significantly through feedback.\n\n"
                    )
                elif improvement > 5:
                    f.write("✅ **Good Learning:** Model showed measurable improvement.\n\n")
                elif improvement > 0:
                    f.write(
                        "⚠️ **Modest Learning:** Some improvement observed, consider more feedback.\n\n"
                    )
                else:
                    f.write(
                        "❌ **No Improvement:** Model did not learn effectively. Review data quality.\n\n"
                    )

            # Bottom performers
            weak_categories = [cat for cat, acc, _ in category_stats[:10]]
            if weak_categories:
                f.write("### Categories Needing More Training Data:\n\n")
                for cat in weak_categories:
                    f.write(f"- {cat}\n")
                f.write("\n")

            f.write("### Next Steps:\n\n")
            f.write("1. Review low-performing categories for data quality issues\n")
            f.write("2. Collect more real-world examples for weak categories\n")
            f.write("3. Continue incremental learning with production feedback\n")
            f.write("4. Monitor confidence distribution in production\n")
            f.write("5. Retrain periodically with accumulated feedback\n\n")

            # Footer
            f.write("---\n\n")
            f.write(
                f"*Generated by Automated Feedback Simulator - {self.metrics['timestamp']}*\n"
            )

        print(f"Report saved: {report_file}")

        # Save metrics JSON
        json_file = output_path / f"feedback_metrics_{self.metrics['timestamp']}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2, default=str)

        print(f"Metrics JSON saved: {json_file}")

        return report_file

    def run_complete_simulation(self, max_samples=None):
        """Run complete simulation pipeline"""
        print("\n" + "=" * 80)
        print("AUTOMATED FEEDBACK SIMULATION")
        print("=" * 80)

        start_time = time.time()

        # Run feedback loop
        self.run_feedback_loop(max_samples)

        # Generate outputs
        self.generate_plots()
        report_file = self.generate_report()

        elapsed_time = time.time() - start_time

        # Final summary
        print("\n" + "=" * 80)
        print("SIMULATION COMPLETED")
        print("=" * 80)
        print(f"\nTotal Time: {elapsed_time:.1f} seconds")
        
        improvement_val = self.metrics.get('improvement')
        if improvement_val is not None:
            print(f"Improvement: {improvement_val * 100:+.2f}%")
        else:
            print("Improvement: N/A (simulation did not complete)")
        
        print(f"Report: {report_file}")
        print(f"Plots: ./plots/")
        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Automated Feedback Simulation for SGD Classifier"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:5015",
        help="API base URL (default: http://localhost:5015)",
    )
    parser.add_argument(
        "--csv", type=str, required=True, help="Path to CSV with training data"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing (default: 10)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between requests in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process (default: all)",
    )
    parser.add_argument(
        "--feedback-all",
        action="store_true",
        help="Send feedback for every record so all 7,800 are used for training (default: only wrong or low-confidence)",
    )
    parser.add_argument(
        "--category-map",
        type=str,
        default=None,
        help="Path to CSV with from_category,to_category to map CSV labels to API labels (fixes 0%% for name mismatch)",
    )
    parser.add_argument(
        "--categories-file",
        type=str,
        default=None,
        help="Path to file with category names (one per line). If set, only these categories are used (e.g. weak_categories.txt).",
    )

    args = parser.parse_args()

    # Initialize simulator
    simulator = FeedbackSimulator(
        api_url=args.api_url,
        csv_path=args.csv,
        batch_size=args.batch_size,
        delay=args.delay,
        feedback_all=args.feedback_all,
        category_map_path=args.category_map,
        categories_file=args.categories_file,
    )

    # Run simulation
    simulator.run_complete_simulation(max_samples=args.max_samples)


if __name__ == "__main__":
    main()
