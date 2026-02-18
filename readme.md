# Adaptive Learning Complaint Classification System

**Automatically sort citizen complaints into the right department—in English or Marathi—and improve over time when staff correct mistakes.**

This project is an AI-powered system for **Pune Municipal Corporation (PMC)**. When a citizen submits a complaint (e.g. *"Street light not working near my house"*), the system suggests which of **78 complaint categories** it belongs to (e.g. Street Lights, Garbage, Drainage). If the system is confident, it uses that suggestion automatically; if not, a staff member chooses the correct category and the system **learns from that choice** so similar complaints are handled better next time. No need to retrain the main AI model—learning happens from everyday corrections.

---

## What’s in this README

| Section | Best for |
|--------|----------|
| [Overview](#overview) | Everyone — what the project does and why it matters |
| [Key features](#key-features) | Everyone — main capabilities at a glance |
| [How it works](#how-it-works) | Technical readers — flow and components |
| [Quick start](#quick-start) | Developers — run the API and try it |
| [API reference](#api-reference-cms_roberta--port-5015) | Integrators — endpoints and examples |
| [Troubleshooting](#troubleshooting) | Developers — common issues and fixes |

---

## Overview

- **What it does:** Reads complaint text (English or Marathi), suggests one of 78 PMC categories, and either **accepts** that suggestion (when confident) or asks for **human feedback** (when unsure) which is a step of Adaptive learning .
- **Why it matters:** Speeds up complaint routing, keeps categories consistent, and reduces manual work while still letting staff correct and teach the system.
- **How it learns:** When a human picks the correct category for an uncertain complaint, that correction is used to update a small “adaptive” part of the system. The main AI model is not retrained.

The live system is the **CMS_RoBerta API** (runs on port **5015**). It combines a fixed multilingual AI model with a lightweight classifier that learns from feedback. For setup and deployment details, see **`CMS_RoBerta/README.md`**.

---

## Key features

- **Bilingual:** Works with complaints in **English** and **Marathi**.
- **78 categories:** Matches PMC’s complaint categories (Street Lights, Garbage, Drainage, Roads, etc.).
- **Confidence-based routing:** High confidence → use the suggestion; low confidence → send to a human, then learn from the correction.
- **Adaptive learning:** Improves from human corrections without full model retraining.
- **REST API:** Integrate with your CMS or tools via `/classify`, `/classify/batch`, and `/feedback`.
- **Web UI:** Simple testing interface at `http://localhost:5015/ui` for trying classifications and submitting feedback.

**Who is this for?** Municipal staff and product owners can use the Overview and Key features to understand what the system does. Developers and integrators can use Quick Start and API Reference to run and connect to the service.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Terms explained](#terms-explained)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Prerequisites](#prerequisites)
6. [Quick Start](#quick-start)
7. [API Reference](#api-reference-cms_roberta--port-5015)
8. [Adaptive Learning Flow](#adaptive-learning-flow)
9. [Safety / Routing](#safety--routing)
10. [Configuration and Tuning](#configuration-and-tuning)
11. [Documentation](#documentation)
12. [Troubleshooting](#troubleshooting)
13. [License & contributing](#license--contributing)

---

## How It Works

### In simple terms

A complaint is sent to **two AI components** at once. Both suggest a category and how sure they are. The system **combines** their answers into one suggestion and one **confidence** (e.g. 90%). If that confidence is **above 80%**, the system **accepts** the suggestion. If it’s **80% or below**, the system asks a **human** to choose the correct category and then **learns** from that choice for future similar complaints.

### The Core Idea (technical)

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

## Terms explained

| Term | Meaning |
|------|--------|
| **Category** | One of the 78 complaint types (e.g. Street Lights, Garbage, Drainage). |
| **Confidence** | How sure the system is (0–100%). Higher = more likely the suggestion is correct. |
| **Accept** | The system uses its suggestion automatically (no human review). Happens when confidence > 80%. |
| **Human feedback** | The system is unsure; a person picks the correct category. Happens when confidence ≤ 80%. |
| **Feedback (API)** | Sending the correct category for a complaint so the system can learn (e.g. via `POST /feedback`). |
| **Adaptive learning** | The system gets better over time by learning from human corrections, without retraining the main AI model. |

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
│   ├── model/                    # config, tokenizer, label2id; add model.safetensors (see CMS_RoBerta/README.md)
│   │   └── adaptive_classifier.pkl  # optional; created when /feedback is used
│   └── README.md                 # CMS_RoBerta setup and API
│
└── docs/                         # Documentation
    ├── working-of-the-system.md  # How the system works (plain language)
    ├── hybrid_ensemble_integration.md
    └── API_DOCUMENTATION.md
```

---

## Prerequisites

- **Python 3.8 or newer** (to run the API).
- **Model files**: Download the model files from [Google Drive](https://drive.google.com/drive/folders/1qA-Sg2pVO9TXpux_LRNsJ-gVYbksNnGS?usp=sharing) and place them in `CMS_RoBerta/model/`:
  - `model.safetensors` (2.09 GB) - Transformer model weights
  - `adaptive_classifier.pkl` (298 KB) - Pre-trained adaptive classifier (optional, will be created automatically if missing)
- For **Docker** deployment: Docker installed and the model files available (e.g. mounted or copied into the image).

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download and add model files

1. Download the model files from [Google Drive](https://drive.google.com/drive/folders/1qA-Sg2pVO9TXpux_LRNsJ-gVYbksNnGS?usp=sharing):
   - `model.safetensors` (2.09 GB) - Transformer model weights
   - `adaptive_classifier.pkl` (298 KB) - Pre-trained adaptive classifier (optional)

2. Place the downloaded files in `CMS_RoBerta/model/` directory:
   ```bash
   # Example: After downloading, move files to the model directory
   cp ~/Downloads/model.safetensors CMS_RoBerta/model/
   cp ~/Downloads/adaptive_classifier.pkl CMS_RoBerta/model/  # Optional
   ```

**Note:** The `adaptive_classifier.pkl` file is optional—it will be created automatically when you first use the `/feedback` endpoint. However, if you download the pre-trained version, it will start with better initial performance.

### 3. Start the CMS_RoBerta API

From the project root (no separate “build index” or evaluate step needed):

```bash
python3 -m uvicorn CMS_RoBerta.app.main:app --host 0.0.0.0 --port 5015
```

### 4. Classify a complaint (port 5015)

```bash
curl -X POST http://localhost:5015/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Street light not working near my house"}'
```

### 5. Submit feedback (when routing is human_feedback)

```bash
curl -X POST http://localhost:5015/feedback \
  -H "Content-Type: application/json" \
  -d '{"complaint_text": "Street light not working", "correct_category": "Street Lights"}'
```

### 6. Use the web UI

Open a browser and go to: **http://localhost:5015/ui** to try single/batch classification and submit feedback.

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

Low-confidence predictions are always sent to humans; the system learns only when someone submits the correct category via the feedback API.

---

## Configuration and Tuning

CMS_RoBerta loads the transformer and label set from `CMS_RoBerta/model/`. 

**Key settings:**
- **Confidence threshold**: The 80% threshold is defined in `CMS_RoBerta/app/ensemble.py` as `CONFIDENCE_ACCEPT = 0.80`. To change it, edit that file.
- **Model weights**: Default is 70% RoBERTa, 30% SGD. Adjustable in `AdaptiveEnsemble.__init__()` (`transformer_weight`, `adaptive_weight`).
- **No .env required**: CMS_RoBerta doesn't use `.env` for normal operation. The model directory and code settings are sufficient.

---

## Documentation

| Document | Description |
|----------|-------------|
| **`CMS_RoBerta/README.md`** | Setup, deployment (Docker/local), and API overview for the classification service. |
| **`docs/working-of-the-system.md`** | How the system works in plain language: Accept vs human feedback, learning from corrections, example scenarios. |
| **`docs/API_DOCUMENTATION.md`** | Detailed API documentation. |
| **`docs/hybrid_ensemble_integration.md`** | Design of the ensemble and two-case routing. |

---

## Troubleshooting

### "Model not loaded" error
- Ensure `CMS_RoBerta/model/` contains the transformer model files: `config.json`, tokenizer files (`tokenizer_config.json`, `tokenizer.json`), `label2id.json`, and model weights (`model.safetensors` or `pytorch_model.bin`). Weights are not in the repo; add them to `model/` before running (see `CMS_RoBerta/README.md`).
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

## License & contributing

- **License:** See the LICENSE file in the repository (if present). Otherwise, use and distribution are subject to your organization’s policy.
- **Contributing:** For changes or extensions, follow your team’s process (e.g. internal repo, pull requests, or contact the maintainers).

