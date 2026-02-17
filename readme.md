# Adaptive Learning Complaint Classification System

A hybrid ensemble system for classifying municipal complaints (PMC - Pune Municipal Corporation). The system classifies bilingual (English/Marathi) complaints into 78 categories and **continuously learns from human feedback without retraining the transformer model**.

**Current system:** **CMS_RoBerta API** (port 5015) — XLM-RoBERTa + hybrid ensemble (SGDClassifier). Two-case routing: **Accept** (confidence >80%) or **Human feedback** (≤80%); submit corrections via `POST /feedback`. See `CMS_RoBerta/README.md`.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [API Reference](#api-reference)
6. [Adaptive Learning Flow](#adaptive-learning-flow)
7. [Safety / Routing](#safety--routing)
8. [Configuration and Tuning](#configuration-and-tuning)
9. [Troubleshooting](#troubleshooting)
10. [Legacy Content (Reference Only)](#legacy-content-reference-only)

---

## How It Works

### The Core Idea

The **CMS_RoBerta** system uses a **hybrid ensemble** of two models that work together:

1. **XLM-RoBERTa (transformer)** — A pre-trained multilingual model fine-tuned on complaint data. It understands semantic meaning and context in both English and Marathi.
2. **SGD Classifier (adaptive)** — A lightweight linear classifier that learns from TF-IDF features (word/ngram patterns). This is the **only** component that adapts from human feedback.

### Classification Flow

```
Complaint Text ("Street light not working near my house")
        │
        ├─────────────────┬─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ XLM-RoBERTa  │  │ SGD Classifier│  │   Combine   │
│ (transformer)│  │  (adaptive)  │  │  predictions│
│              │  │              │  │             │
│ Label:       │  │ Label:       │  │ Final label │
│ "Street      │  │ "Street      │  │ + confidence│
│  Lights"     │  │  Lights"     │  │             │
│ Conf: 94%    │  │ Conf: 88%    │  │ Conf: 92%   │
└──────┬───────┘  └──────┬───────┘  └──────┬──────┘
       │                 │                 │
       └─────────────────┴─────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Confidence > 80%?  │
              └──────────┬───────────┘
                         │
            ┌────────────┼────────────┐
            │                        │
            ▼                        ▼
      ┌──────────┐            ┌──────────────┐
      │  Accept  │            │Human feedback│
      │ (use it) │            │ (needs review│
      └──────────┘            └──────┬───────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │ POST /feedback   │
                            │ → SGD learns     │
                            │ → Save to disk   │
                            └──────────────────┘
```

### Key Points

1. **Two models, one decision:** Both models see the same text and make independent predictions. Their results are combined into a single final category and confidence score.

2. **Confidence-based routing:** 
   - **> 80% confidence** → **Accept** (use the prediction automatically)
   - **≤ 80% confidence** → **Human feedback** (human picks the correct category)

3. **Adaptive learning:** Only the **SGD classifier** learns from human corrections. The transformer (RoBERTa) is **never retrained** — it stays fixed. When you submit feedback via `POST /feedback`, the SGD classifier updates immediately and saves to `CMS_RoBerta/model/adaptive_classifier.pkl`.

4. **Human-in-the-Loop (HITL):** The system combines machine intelligence (RoBERTa + SGD) for clear cases with human judgment for edge cases. Every correction makes the system better on similar complaints.

For detailed scenarios and examples, see **`docs/working-of-the-system.md`**.

---

## Architecture

```
  Request (text) → main.py (FastAPI)
                        │
                        ▼
              ┌─────────────────────┐
              │  ensemble.py        │  RoBERTa + SGDClassifier
              │  (hybrid ensemble)  │  → label, confidence, routing
              └─────────┬───────────┘
                        │
           confidence > 80%? ── Yes → Accept
                        │
                       No → Human feedback → POST /feedback
                                                │
                                                ▼
                        ┌───────────────────────────────────┐
                        │  adaptive_classifier.py            │
                        │  partial_fit(); save to model/     │
                        └───────────────────────────────────┘
```

---

## Project Structure

```
├── requirements.txt              # Python dependencies
├── readme.md                     # This file
│
├── CMS_RoBerta/                  # XLM-RoBERTa complaint API (current system)
│   ├── app/
│   │   ├── main.py               # FastAPI: /classify, /classify/batch, /feedback
│   │   ├── ensemble.py           # Hybrid ensemble (transformer + adaptive classifier)
│   │   ├── adaptive_classifier.py # SGDClassifier + TF-IDF (online learning)
│   │   └── __init__.py
│   ├── model/                    # XLM-RoBERTa weights + label2id.json
│   │   └── adaptive_classifier.pkl  # optional; created when /feedback is used
│   └── README.md                 # CMS_RoBerta setup and API
│
└── docs/                         # Documentation
    └── hybrid_ensemble_integration.md  # Ensemble + two-case routing design
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the CMS_RoBerta API

No index build or evaluate scripts are required. From project root:

```bash
python3 -m uvicorn CMS_RoBerta.app.main:app --host 0.0.0.0 --port 5015
```

### 3. Classify a Complaint (port 5015)

```bash
curl -X POST http://localhost:5015/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Street light not working near my house"}'
```

### 4. Submit Feedback (when routing is human_feedback)

```bash
curl -X POST http://localhost:5015/feedback \
  -H "Content-Type: application/json" \
  -d '{"complaint_text": "Street light not working", "correct_category": "Street Lights"}'
```

### 5. Access the Web UI

Open your browser to: `http://localhost:5015/ui`

---

## API Reference (CMS_RoBerta — port 5015)

### POST /classify

Classify a single complaint. Returns ensemble label, confidence, and routing (accept vs human_feedback).

```json
// Request
{"text": "Street light not working near my house"}

// Response
{
    "label": "Street Lights",
    "label_id": 42,
    "confidence": 0.92,
    "routing": "accept",
    "transformer_label": "Street Lights",
    "transformer_confidence": 0.94,
    "adaptive_label": "Street Lights",
    "adaptive_confidence": 0.88,
    "agreement": true
}
```

When confidence ≤ 80%, `routing` is `"human_feedback"`; use the feedback form or POST /feedback with the correct category.

### POST /classify/batch

```json
// Request
{"texts": ["complaint 1", "complaint 2"]}

// Response
{"predictions": [{ "label": "...", "confidence": 0.9, "routing": "accept", ... }, ...]}
```

### POST /feedback

Submit human correction to update the adaptive (SGD) classifier.

```json
// Request
{"complaint_text": "Street light not working", "correct_category": "Street Lights"}

// Response
{"status": "learned", "message": "Adaptive classifier updated with feedback"}
```

### GET /labels

Returns the list of 78 possible category labels (from the model's label2id).

### GET /health

Returns `{"status": "ok", "model_loaded": true}`.

### GET /ui

Serves the testing UI for single/batch classify and feedback.

---

## Adaptive Learning Flow

### Current system (CMS_RoBerta)

1. **Classify**: For each complaint, the ensemble (XLM-RoBERTa + SGD classifier) returns a label and confidence.
2. **Routing**: If confidence **> 80%** → **Accept** (use the prediction). If **≤ 80%** → **Human feedback** (human picks the correct category).
3. **Learning**: When routing is human_feedback, the user submits the correct category via POST /feedback. Only the **adaptive (SGD) classifier** is updated with `partial_fit`; the transformer is never retrained. The updated classifier is saved to `CMS_RoBerta/model/adaptive_classifier.pkl`.

See `docs/working-of-the-system.md` for full details.

---

## Safety / Routing

The current system has **two outcomes only**: **Accept** (confidence > 80%) or **Human feedback** (≤ 80%). 

- **Accept**: The system is confident enough to use the prediction automatically. No human review needed.
- **Human feedback**: The system is unsure (low confidence or models disagree). A human provides the correct category, and the adaptive classifier learns from it.

There is no multi-layer safety table or drift monitoring; low-confidence predictions are sent to humans and learning happens only when feedback is submitted via POST /feedback.

---

## Configuration and Tuning

CMS_RoBerta loads the transformer and label set from `CMS_RoBerta/model/`. 

**Key settings:**
- **Confidence threshold**: The 80% threshold is defined in `CMS_RoBerta/app/ensemble.py` as `CONFIDENCE_ACCEPT = 0.80`. To change it, edit that file.
- **Model weights**: Default is 70% RoBERTa, 30% SGD. Adjustable in `AdaptiveEnsemble.__init__()` (`transformer_weight`, `adaptive_weight`).
- **No .env required**: CMS_RoBerta doesn't use `.env` for normal operation. The model directory and code settings are sufficient.

---

## Troubleshooting

### "Model not loaded" error
- Ensure `CMS_RoBerta/model/` contains the transformer model files (config.json, pytorch_model.bin, tokenizer files, label2id.json).
- Check that the model directory path is correct (default: `CMS_RoBerta/model/`).

### "UI file not found" error
- Ensure you're running the server from the project root directory (`/home/stark/Desktop/Adaptive Learning/`).
- The UI file should be at `CMS_RoBerta/app/ui.html`.

### Low confidence on all predictions
- The adaptive classifier may not be trained yet. Submit feedback via `/feedback` to train it.
- Check if both models agree (`agreement: true`). Disagreement lowers confidence.

### Server won't start
- Check that port 5015 is not already in use: `lsof -i :5015`
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version (3.8+): `python3 --version`

### Slow classification
- First request may be slow (model loading). Subsequent requests should be faster.
- If using GPU, ensure CUDA is available: `python3 -c "import torch; print(torch.cuda.is_available())"`

---

## Legacy Content (Reference Only)

The sections below describe the **removed KNN/FAISS pipeline** and are kept for reference only. The current system is **CMS_RoBerta** only.

### Legacy: KNN/FAISS Pipeline (Removed)

The previous system used embedding-based similarity search with FAISS. That pipeline and its `modules/` package have been removed. Only CMS_RoBerta is in use.

### Legacy: Configuration Reference

The following `.env` configuration tables applied only to the removed KNN pipeline. **CMS_RoBerta does not use them**.

<details>
<summary>Click to expand legacy configuration reference</summary>

#### Embedding Model

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL_NAME` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence Transformer model name |
| `EMBEDDING_DIMENSION` | `384` | Vector dimension (auto-detected) |
| `EMBEDDING_BATCH_SIZE` | `256` | Texts per batch during encoding |
| `DEVICE` | `cpu` | `cpu` or `cuda` for GPU |

#### Data Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAIN_DATA_PATH` | `./data/train.jsonl` | Training data JSONL file |
| `TEST_DATA_PATH` | `./data/test.jsonl` | Test data JSONL file |
| `VAL_DATA_PATH` | `./data/val.jsonl` | Validation data JSONL file |
| `DATA_INPUT_FIELD` | `input` | JSONL field name for complaint text |
| `DATA_OUTPUT_FIELD` | `output` | JSONL field name for category label |

#### Vector Store (FAISS)

| Variable | Default | Description |
|----------|---------|-------------|
| `FAISS_INDEX_TYPE` | `Flat` | `Flat` (exact) or `IVF` (approximate, faster) |
| `FAISS_IVF_NLIST` | `100` | Number of IVF clusters (only if IVF) |
| `DISTANCE_METRIC` | `cosine` | `cosine` or `euclidean` |

#### Classifier

| Variable | Default | Description |
|----------|---------|-------------|
| `K_NEIGHBORS` | `15` | Number of nearest neighbors to consider |
| `VOTING_STRATEGY` | `weighted` | `weighted` (closer=more weight) or `uniform` |
| `MIN_CATEGORY_VOTES` | `3` | Min votes from a category to trust it |

#### Safety Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIDENCE_THRESHOLD_HIGH` | `0.85` | Above this → eligible for auto-learn |
| `CONFIDENCE_THRESHOLD_MEDIUM` | `0.65` | Above this → classify (don't learn) |
| `SIMILARITY_THRESHOLD` | `0.60` | Min similarity to known category examples |
| `MAX_AUTO_LEARN_PER_DAY` | `20` | Rate limit: max auto-learned per day |

#### Ensemble Model

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_ENSEMBLE` | `true` | Enable/disable the backup SGD model |
| `ENSEMBLE_AGREEMENT_REQUIRED` | `true` | Both models must agree for auto-learn |

#### Drift Monitoring

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_DRIFT_MONITORING` | `true` | Enable/disable drift checks |
| `DRIFT_ACCURACY_THRESHOLD` | `0.75` | Below this → rollback triggered |
| `DRIFT_WINDOW_SIZE` | `500` | Samples to evaluate per drift check |

#### API Server

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Server bind host |
| `API_PORT` | `5000` | Server port |
| `API_DEBUG` | `false` | Flask debug mode |

</details>
