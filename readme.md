# Adaptive Learning Complaint Classification System

**Automatically sort citizen complaints into the right department—in English or Marathi—and improve over time when staff correct mistakes.**

This project is an AI-powered system for **Pune Municipal Corporation (PMC)**. When a citizen submits a complaint (e.g. *"Street light not working near my house"*), the system suggests one of **78 complaint categories** (e.g. Street Lights, Garbage, Drainage). If the system is confident, it uses that suggestion; if not, a staff member chooses the correct category and the system **learns from that choice**. No full model retrain—learning happens from everyday corrections via the adaptive (SGD) classifier.

---

## What’s in this README

| Section                                                         | Best for                                  |
| --------------------------------------------------------------- | ----------------------------------------- |
| [Overview](#overview)                                              | Everyone — what the project does         |
| [Key features](#key-features)                                      | Main capabilities                         |
| [How it works](#how-it-works)                                      | Flow and components                       |
| [Download SGD classifier model](#download-sgd-classifier-model)    | Setting up the learnt adaptive model      |
| [Quick start](#quick-start)                                        | Run the API and try it                    |
| [Current accuracy &amp; retraining](#current-accuracy--retraining) | Model performance and how it is retrained |
| [API reference](#api-reference-cms_roberta--port-5016)             | Endpoints and examples                    |
| [Troubleshooting](#troubleshooting)                                | Common issues and fixes                   |

---

## Overview

- **What it does:** Takes complaint text (English or Marathi), suggests one of 78 PMC categories, and either **accepts** the suggestion (when confident) or asks for **human feedback** (when unsure).
- **Why it matters:** Speeds up complaint routing, keeps categories consistent, and reduces manual work while letting staff correct and teach the system.
- **How it learns:** When a human picks the correct category for an uncertain complaint, that correction updates the **adaptive (SGD) classifier** only. The main transformer (RoBERTa) is not retrained.

The live system is the **CMS_RoBerta API** (port **5016**): a fixed multilingual transformer plus a lightweight SGD classifier that learns from feedback. See **`CMS_RoBerta/README.md`** for service setup.

---

## Key features

- **Bilingual:** English and Marathi.
- **78 categories:** PMC complaint categories (Street Lights, Garbage, Drainage, etc.).
- **Confidence-based routing:** High confidence → accept; low → human feedback, then learn from correction.
- **Adaptive learning:** Improves from human corrections without full model retraining.
- **REST API:** `/classify`, `/classify/batch`, `/feedback`.
- **Web UI:** `http://localhost:5016/ui` for testing and feedback.

---

## How it works

The system uses a **hybrid ensemble**:

1. **XLM-RoBERTa (transformer)** — Fixed multilingual model; understands meaning in English and Marathi.
2. **SGD classifier (adaptive)** — Lightweight classifier on TF-IDF features; **the only part that learns** from human feedback.

**Flow:** Both models predict; results are combined into one label and confidence. If confidence **> 80%** → **Accept**. If **≤ 80%** → **Human feedback**; staff submits the correct category via `POST /feedback`, the SGD classifier updates with `partial_fit` and saves to `CMS_RoBerta/model/adaptive_classifier.pkl`.

---

## Download SGD classifier model

The **learnt adaptive model** (SGD classifier) is stored as **`adaptive_classifier.pkl`**. You can use the pre-trained version that was trained on 7,800 samples and then on 20 weak categories (see [Current accuracy &amp; retraining](#current-accuracy--retraining)).

### 1. Download the model file

- **SGD classifier (adaptive model) — download link:**
  [adaptive_classifier.pkl (Google Drive)](https://drive.google.com/file/d/10bwLoCl8lQgqHnxbkzuLtXJe3rdJ74L0/view?usp=sharing)

Download this file to your machine (e.g. via browser or `gdown` if you use Google Drive from the command line).

### 2. Place it in the project

Put the downloaded file in the API model directory so the app can load it:

```bash
# From the project root: Adaptive Learning/
cp /path/to/your/downloaded/adaptive_classifier.pkl CMS_RoBerta/model/
```

Example if the file is in your Downloads folder:

```bash
cp ~/Downloads/adaptive_classifier.pkl CMS_RoBerta/model/
```

### 3. Verify

Ensure the file exists:

```bash
ls -l CMS_RoBerta/model/adaptive_classifier.pkl
```

When you start the API, it will load this file. If `adaptive_classifier.pkl` is missing, the API can still run: the adaptive part will be created when the first feedback is submitted, but starting with the downloaded file gives you the current learnt model and better accuracy from day one.

**Path summary:** the SGD classifier model must be at **`CMS_RoBerta/model/adaptive_classifier.pkl`** (relative to the project root).

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add model files to `CMS_RoBerta/model/`

- **Transformer (RoBERTa):** Download the main model files (e.g. `model.safetensors`, `config.json`, tokenizer, `label2id.json`) from [Google Drive](https://drive.google.com/drive/folders/1qA-Sg2pVO9TXpux_LRNsJ-gVYbksNnGS?usp=sharing) and place them in **`CMS_RoBerta/model/`**. See **`CMS_RoBerta/README.md`** for the full list of required files.
- **SGD classifier (adaptive):** Download **adaptive_classifier.pkl** from [Google Drive](https://drive.google.com/file/d/10bwLoCl8lQgqHnxbkzuLtXJe3rdJ74L0/view?usp=sharing) and copy it to **`CMS_RoBerta/model/adaptive_classifier.pkl`** (see [Download SGD classifier model](#download-sgd-classifier-model)).

### 3. Start the API

From the project root:

```bash
python3 -m uvicorn CMS_RoBerta.app.main:app --host 0.0.0.0 --port 5016
```

### 4. Try it

- **Classify:**`curl -X POST http://localhost:5016/classify -H "Content-Type: application/json" -d '{"text": "Street light not working near my house"}'`
- **Web UI:** open **http://localhost:5016/ui**

---

## Run with Docker

Run the API in a container so you don’t need to install Python or dependencies on the host.

**1. Put model files on the host**  
Ensure **`CMS_RoBerta/model/`** contains the required files (see [Quick start](#quick-start)): transformer files from [Google Drive](https://drive.google.com/drive/folders/1qA-Sg2pVO9TXpux_LRNsJ-gVYbksNnGS?usp=sharing), and **adaptive_classifier.pkl** from [Google Drive](https://drive.google.com/file/d/10bwLoCl8lQgqHnxbkzuLtXJe3rdJ74L0/view?usp=sharing). The container mounts this folder, so the same files are used and feedback updates are saved on the host.

**2. Build and run** (from project root `Adaptive Learning/`):

```bash
docker compose up --build
# Or in background:
docker compose up -d --build
```

API: **http://localhost:5016** — UI: http://localhost:5016/ui — Health: http://localhost:5016/health

**3. Optional (without Compose):**

```bash
docker build -t cms-roberta-api:latest .
docker run -d --name cms-complaint-classifier -p 5016:5016 \
  -v $(pwd)/CMS_RoBerta/model:/app/CMS_RoBerta/model \
  -e MODEL_DIR=/app/CMS_RoBerta/model \
  cms-roberta-api:latest
```

**4. Stop / logs:** `docker compose down` | `docker compose logs -f`

See **`DOCKER.md`** for more detail.

---

## Current accuracy & retraining

### Current model accuracy

The **learnt model** (`CMS_RoBerta/model/adaptive_classifier.pkl`) was trained in two stages:

1. **First run:** Full feedback simulation on **7,800 records** → **65%** accuracy overall (20 categories were weak due to label name mismatch).
2. **Second run:** Training only on the **20 weak categories** (2,000 records) with category mapping and feedback → **100%** on those 2,000.

**Combined overall accuracy** on the full 7,800 records: **~90.6%** (7,070 correct / 7,800). Details and formula: **`Testing/reports/updated_overall_accuracy.md`**.

### How the model is retrained

- **In production:** Every time a user submits the correct category via **`POST /feedback`**, the API updates the SGD classifier with `partial_fit` and saves **`adaptive_classifier.pkl`** again. So the model keeps improving with real corrections.
- **Bulk / testing:** The **Testing** folder contains a **feedback simulator** and scripts to:
  - Run automated feedback over a CSV (e.g. 7,800 samples).
  - Apply **category mapping** so CSV labels match the API’s 78 labels.
  - Train only on **weak categories** (see **`Testing/weak_categories.txt`** and **`Testing/README.md`**).

Steps and commands for retraining and improving accuracy (full run, weak-only run, category mapping) are in **`Testing/README.md`**. The learnt model file used by the API is always **`CMS_RoBerta/model/adaptive_classifier.pkl`**.

---

## API reference (CMS_RoBerta — port 5016)

### POST /classify

Classify one complaint. Returns label, confidence, and routing (accept vs human_feedback).

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

When confidence ≤ 80%, `routing` is `"human_feedback"`; use the UI or `POST /feedback` with the correct category.

### POST /classify/batch

Request: `{"texts": ["complaint 1", "complaint 2"]}`
Response: `{"predictions": [{ "label": "...", "confidence": 0.9, "routing": "accept", ... }, ...]}`

### POST /feedback

Send the correct category so the SGD classifier can learn.

```json
// Request
{"complaint_text": "Street light not working", "correct_category": "Street Lights"}

// Response
{"status": "learned", "message": "Adaptive classifier updated with feedback"}
```

### GET /labels

Returns the 78 category labels.

### GET /health

Returns `{"status": "ok", "model_loaded": true}`.

### GET /ui

Serves the web UI for classification and feedback.

---

## Project structure

```
├── requirements.txt
├── readme.md                     # This file
├── CMS_RoBerta/                  # Classification API (port 5016)
│   ├── app/
│   │   ├── main.py               # FastAPI: /classify, /classify/batch, /feedback
│   │   ├── ensemble.py           # RoBERTa + SGD ensemble
│   │   ├── adaptive_classifier.py # SGD classifier (learns from feedback)
│   │   └── ui.html
│   └── model/                    # Put transformer + label2id here
│       └── adaptive_classifier.pkl  # SGD classifier (download from Drive link above)
├── Testing/                      # Feedback simulator, weak-category training, reports
│   ├── README.md                 # Steps to improve accuracy and retrain
│   ├── feedback_simulator.py
│   ├── training_samples_sgd_7800.csv
│   ├── category_mapping.csv
│   ├── weak_categories.txt
│   └── reports/
└── docs/
    ├── working-of-the-system.md
    ├── API_DOCUMENTATION.md
    └── hybrid_ensemble_integration.md
```

---

## Troubleshooting

- **Model not loaded:** Ensure `CMS_RoBerta/model/` has the transformer files (config, tokenizer, `label2id.json`, weights). Add **adaptive_classifier.pkl** from [Google Drive](https://drive.google.com/file/d/10bwLoCl8lQgqHnxbkzuLtXJe3rdJ74L0/view?usp=sharing) to the same folder.
- **UI not found:** Run the server from the project root so `CMS_RoBerta/app/ui.html` is found.
- **Low confidence:** The adaptive classifier may be untrained. Use the downloaded **adaptive_classifier.pkl** or submit feedback via `/feedback` (or run the Testing feedback simulator).
- **Port in use:** Check port 5016: `lsof -i :5016`.

---

## Documentation

| Document                                                  | Description                                                                     |
| --------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **`CMS_RoBerta/README.md`**                       | API setup and deployment.                                                       |
| **`Testing/README.md`**                           | Feedback simulator, category mapping, weak-only training, accuracy improvement. |
| **`Testing/reports/updated_overall_accuracy.md`** | Final overall accuracy (90.6%) and how it was computed.                         |
| **`docs/working-of-the-system.md`**               | How the system works.                                                           |
| **`docs/API_DOCUMENTATION.md`**                   | Detailed API docs.                                                              |

---

## License & contributing

- **License:** See the LICENSE file in the repository (if present).
- **Contributing:** Follow your organization’s process for changes and extensions.
