#!/usr/bin/env python3
"""
Build clean_samples.csv from All_Complaint_data.csv and report3.csv
containing only complaints whose category is in the 'Categories needing improvement' list,
for use in testing the SGD classifier.
"""
import csv
from pathlib import Path

# Categories needing improvement (accuracy < 70%, support > 0) - exact names from your report
CATEGORIES_NEEDING_IMPROVEMENT = {
    "Garden Cleaning & maintenance",
    "Garbage Depot Complaint",
    "Public Health Related",
    "Bhavan",
    "Environment (H.O)",
    "License (Parwana)",
    "Abhyagat Kaksha",
    "Garden Civil maintenance work",
    "City Development Plan",
    "Property Tax-TAT(Title Transfer)",
    "Traffic-Planning",
    "PMC Properties",
    "Garden Electric maintenance work",
    "Unauthorized banners / advertisements / Permit / license in Pune City",
    "General Administration",
    "Primary Education",
    "Information Technology",
    "Sports",
    "TDR",
    "Lashkar Water Supply",
    "Communicable Disease",
    "Unauthorized slaughterhouse / crude meat",
    "Urban Poor Scheme",
    "Heritage Cell (Bhavan HO)",
    "RTI Related Complaint",
    "PMC Security Complaints",
    "Vehicle department",
}


def normalize_category(s: str) -> str:
    return (s or "").strip()


def in_improvement_list(category: str) -> bool:
    return normalize_category(category) in CATEGORIES_NEEDING_IMPROVEMENT


def main():
    import random
    from collections import defaultdict
    
    base = Path(__file__).resolve().parent
    all_data_path = base / "All_Complaint_data.csv"
    report3_path = base / "report3.csv"
    out_path = base / "clean_samples.csv"
    
    # Target: 3000-4000 samples total, balanced across categories
    TARGET_TOTAL = 3500  # middle of range
    SAMPLES_PER_CATEGORY = TARGET_TOTAL // len(CATEGORIES_NEEDING_IMPROVEMENT)  # ~130 per category

    # Collect all samples per category
    samples_by_category = defaultdict(list)
    seen = set()   # (text_stripped[:200], category) to avoid exact dupes

    def add(complaint: str, category: str):
        complaint = (complaint or "").strip()
        if not complaint:
            return
        key = (complaint[:200], category)
        if key in seen:
            return
        seen.add(key)
        samples_by_category[category].append(complaint)

    # ---- All_Complaint_data.csv: comma, "com_description", "category_name"
    if all_data_path.exists():
        with open(all_data_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cat = normalize_category(row.get("category_name", ""))
                if in_improvement_list(cat):
                    add(row.get("com_description", ""), cat)

    # ---- report3.csv: semicolon, "com_description", "category_name"
    if report3_path.exists():
        with open(report3_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                cat = normalize_category(row.get("category_name", ""))
                if in_improvement_list(cat):
                    add(row.get("com_description", ""), cat)

    # Sample balanced set: take up to SAMPLES_PER_CATEGORY from each category
    rows_out = []
    for category in CATEGORIES_NEEDING_IMPROVEMENT:
        samples = samples_by_category[category]
        if len(samples) > SAMPLES_PER_CATEGORY:
            # Randomly sample if we have more than needed
            selected = random.sample(samples, SAMPLES_PER_CATEGORY)
        else:
            # Take all if we have fewer
            selected = samples
        for complaint in selected:
            rows_out.append((complaint, category))

    # Shuffle to mix categories
    random.shuffle(rows_out)

    # Write clean_samples.csv: complaint, category
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["complaint", "category"])
        for complaint, category in rows_out:
            writer.writerow([complaint, category])

    print(f"Written {len(rows_out)} rows to {out_path}")
    # Per-category counts
    from collections import Counter
    counts = Counter(cat for _, cat in rows_out)
    print(f"\nSamples per category (target: ~{SAMPLES_PER_CATEGORY}):")
    for cat in sorted(CATEGORIES_NEEDING_IMPROVEMENT):
        count = counts.get(cat, 0)
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
