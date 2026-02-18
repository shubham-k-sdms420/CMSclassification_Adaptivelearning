# SGD Classifier Training and Evaluation

This directory contains scripts to train the SGD (adaptive) classifier using feedback samples and track learning metrics.

## Files

- **`clean_samples.csv`** - Contains complaints and their correct categories for the 27 categories needing improvement (24,365 samples)
- **`build_clean_samples.py`** - Script that generated `clean_samples.csv` from source data
- **`train_and_evaluate_sgd.py`** - Main script to train SGD via `/feedback` API and generate metrics report
- **`sgd_learning_metrics.md`** - Generated markdown report with accuracy tables and learning patterns
- **`sgd_learning_metrics.json`** - Generated JSON report with detailed metrics

## Quick Start

### 1. Ensure API is Running

```bash
# From project root
python3 -m uvicorn CMS_RoBerta.app.main:app --host 0.0.0.0 --port 5015
```

### 2. Run Training Script

```bash
cd Testing
python3 train_and_evaluate_sgd.py
```

**Options:**
- `--api-url http://localhost:5015` - API base URL (default: http://localhost:5015)
- `--batch-size 10` - Feedback batch size (default: 10)
- `--sample-limit 100` - Limit number of samples for quick test (default: all)
- `--csv clean_samples.csv` - Input CSV file (default: clean_samples.csv)
- `--output sgd_learning_metrics.md` - Output report file (default: sgd_learning_metrics.md)

**Example (test with 100 samples):**
```bash
python3 train_and_evaluate_sgd.py --sample-limit 100
```

### 3. View Results

After training completes, check:
- **`sgd_learning_metrics.md`** - Human-readable report with tables
- **`sgd_learning_metrics.json`** - Machine-readable detailed metrics

## How It Works

1. **Baseline Evaluation**: Tests the current model on a held-out test set (before training)
2. **Training Phase**: Sends feedback samples to `/feedback` API to train the SGD classifier
3. **Final Evaluation**: Re-tests on the same test set (after training) to measure improvement

The script tracks:
- **Accuracy per category** (before vs after)
- **Confidence scores** (average before vs after)
- **Learning patterns** (sample complaints used for training)
- **Improvement metrics** (how much each category improved)

## Report Structure

The generated `sgd_learning_metrics.md` contains:

1. **Summary** - Overall accuracy before/after, total samples
2. **Per-Category Accuracy Metrics** - Table showing:
   - Before accuracy
   - After accuracy
   - Improvement (percentage points)
   - Feedback samples sent
   - Test samples evaluated
3. **Learning Patterns** - Shows sample complaints used for training per category
4. **Average Confidence Analysis** - How confidence scores changed

## Notes

- The script splits data: first 100 samples (or 10%) are used for testing, rest for training
- Training happens incrementally - SGD learns from each feedback sample
- The script periodically re-evaluates the test set during training to show progress
- Make sure the API is running and accessible before starting training
