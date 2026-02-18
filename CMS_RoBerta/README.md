# CMS Complaint Classification API

FastAPI service that serves the fine-tuned **XLM-RoBERTa-Large** (Stage 2) complaint classification model for PMC CMS, with a **hybrid ensemble** (transformer + adaptive SGDClassifier) and **two-case routing**:

- **Good confidence > 80%** → **Accept** (use prediction; no human review).
- **Low confidence ≤ 80%** → **Human feedback** (queue for review; human selects category; system learns via `/feedback`).

Runs  on **port 5015** and is intended for deployment on your company server via Docker.

---

## Setup from pmc_stage2_output (local trained model)

**✓ Model ready:** The workspace is pre-populated with the Stage 2 model from `/home/stark/CMS_v2/pmc_stage2_output` and is ready for deployment.

If you need to refresh or set up from scratch, run these commands:

```bash
# Copy missing tokenizer files (model.safetensors and config.json should already be in model/)
cp /home/stark/CMS_v2/pmc_stage2_output/final_model/tokenizer_config.json "/home/stark/CMS Classification model/model/"
cp /home/stark/CMS_v2/pmc_stage2_output/final_model/tokenizer.json "/home/stark/CMS Classification model/model/"

# Add label mapping (required for API)
cp /home/stark/CMS_v2/category_to_id.json "/home/stark/CMS Classification model/model/label2id.json"
```

**Or use symlinks** (no extra disk usage):

```bash
cd "/home/stark/CMS Classification model/model"
ln -sf /home/stark/CMS_v2/pmc_stage2_output/final_model/tokenizer_config.json .
ln -sf /home/stark/CMS_v2/pmc_stage2_output/final_model/tokenizer.json .
cp /home/stark/CMS_v2/category_to_id.json ./label2id.json
```

---

## Download Model Files from Google Drive

**Download the model files from [Google Drive](https://drive.google.com/drive/folders/1qA-Sg2pVO9TXpux_LRNsJ-gVYbksNnGS?usp=sharing):**

- **`model.safetensors`** (2.09 GB) - Transformer model weights
- **`adaptive_classifier.pkl`** (298 KB) - Pre-trained adaptive classifier (optional)

After downloading, place these files in `CMS_RoBerta/model/` directory:

```bash
# Example: After downloading, move files to the model directory
cp ~/Downloads/model.safetensors CMS_RoBerta/model/
cp ~/Downloads/adaptive_classifier.pkl CMS_RoBerta/model/  # Optional
```

**Note:** The `adaptive_classifier.pkl` file is optional—it will be created automatically when you first use the `/feedback` endpoint. However, if you download the pre-trained version, it will start with better initial performance.

### Required files inside `model/`

Place these files in `CMS Classification model/model/`:

| File | Description |
|------|-------------|
| `config.json` | Model config (from `trainer.save_model`) |
| `pytorch_model.bin` or `model.safetensors` | Model weights |
| `tokenizer_config.json` | Tokenizer config |
| `tokenizer.json` | Tokenizer (or `vocab.txt` + `special_tokens_map.json` for older models) |
| **`label2id.json`** | Label name → id mapping (use `category_to_id.json` from CMS_v2) |

So your layout should look like:

```
CMS_RoBerta/
  app/
    main.py              # FastAPI app (ensemble + routing + /feedback)
    __init__.py
    ensemble.py          # Hybrid ensemble (XLM-RoBERTa + adaptive classifier)
    adaptive_classifier.py  # SGDClassifier + TF-IDF (online learning)
  model/
    config.json
    pytorch_model.bin   # or model.safetensors
    tokenizer_config.json
    tokenizer.json
    label2id.json
    adaptive_classifier.pkl  # optional; created when /feedback is used
  requirements.txt
  Dockerfile
  README.md
```

**From Drive:**  
Either copy the entire **`trained_model/model`** folder contents into **`model/`**, or download **`pmc_model.zip`**, unzip it, and put the contents (same files as above) into **`model/`**.

---

## Run with Docker (deployment)

1. **Put the model in place**  
   Ensure all required files are under `model/` as above.

2. **Build the image** (from the `CMS Classification model` directory):

   ```bash
   cd "CMS Classification model"
   docker build -t cms-classification-api .
   ```

3. **Run the container** (mount the model, expose port 5015):

   ```bash
   docker run -d \
     --name cms-classification \
     -p 5015:5015 \
     -v "$(pwd)/model:/app/model:ro" \
     cms-classification-api
   ```

   If your model is elsewhere, e.g. `/opt/cms/model`:

   ```bash
   docker run -d \
     --name cms-classification \
     -p 5015:5015 \
     -v /opt/cms/model:/app/model:ro \
     cms-classification-api
   ```

4. **Check health:**

   ```bash
   curl http://localhost:5015/health
   ```

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/labels` | List of possible classification labels |
| POST | `/classify` | Classify a single complaint; returns `label`, `label_id`, `confidence`, `routing` (`accept` or `human_feedback`) |
| POST | `/classify/batch` | Classify multiple texts; each prediction includes `confidence` and `routing` |
| POST | `/feedback` | Submit human correction: `{"complaint_text": "...", "correct_category": "..."}` to update the adaptive classifier |

### Example: single classification

```bash
curl -X POST http://localhost:5015/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Illegal construction noise after 10 PM near my society."}'
```

Response:

```json
{"label": "Building Permission", "label_id": 9, "confidence": 0.92, "routing": "accept", "probabilities": null}
```

Use `routing`: **`accept`** (good confidence >80%) or **`human_feedback`** (low confidence; send to human, then call `/feedback` with the correct category).

With probabilities:

```bash
curl -X POST http://localhost:5015/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Road potholes near main market.", "return_probabilities": true}'
```

### Example: batch classification

```bash
curl -X POST http://localhost:5015/classify/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Complaint one...", "Complaint two..."], "return_probabilities": false}'
```

### Example: submit human feedback (for low-confidence cases)

When `routing` is `human_feedback`, have a human choose the correct category, then:

```bash
curl -X POST http://localhost:5015/feedback \
  -H "Content-Type: application/json" \
  -d '{"complaint_text": "Road potholes near main market.", "correct_category": "Road, pavement, divider, pits, repair / new speed breaker / zebra crossing"}'
```

The adaptive classifier is updated immediately and persisted to `model/adaptive_classifier.pkl`.

---

## Run locally (without Docker)

From the `CMS Classification model` directory:

```bash
pip install -r requirements.txt
# Ensure model files are in ./model/
python -m uvicorn app.main:app --host 0.0.0.0 --port 5015
```

Or:

```bash
python -m app.main
```

---

## Deploying only this folder

You can push or copy only the **`CMS Classification model`** folder to your server:

1. Copy the folder (including `model/` with the downloaded files).
2. On the server: `cd "CMS Classification model"`, then build and run Docker as above.

If you prefer not to put the model in the repo, copy only the code (no `model/`), then on the server place the model files in a directory and use the same `-v /path/to/model:/app/model` when running the container.

---

## Using GitHub and large model files

This repository is designed so that **code and configs are stored in GitHub, but large model weights are not**:

- `.gitignore` explicitly ignores:
  - `*.safetensors`
  - `model/*.bin`
- This is required because:
  - GitHub enforces a **100 MB per-file limit** for normal repos.
  - The fine-tuned XLM‑RoBERTa weights (`model.safetensors` / `pytorch_model.bin`) are typically **hundreds of MBs or GBs**.

### What gets pushed to GitHub

- All **application code** (`app/`, `Dockerfile`, `requirements.txt`, etc.).
- All **non‑huge model metadata** that is safe/small enough, for example:
  - `model/config.json`
  - `model/tokenizer_config.json`
  - `model/tokenizer.json` (or vocab/special token files)
  - `model/label2id.json`

### How to provide the actual model for deployment

At runtime the API expects the weights to be present under `model/` (e.g. `model/model.safetensors` or `model/pytorch_model.bin`).  
Since these files are **not stored in GitHub**, they must be supplied by your deployment process. Common patterns:

1. **Bake into the Docker image**
   - During image build, copy the model weights into `/app/model`:
     ```dockerfile
     COPY model/model.safetensors /app/model/model.safetensors
     ```
   - This can be done from a local path, an internal artifact store, or any storage DevOps has access to.

2. **Download on container start**
   - Store the weights in S3 / GCS / Azure Blob / internal storage / Hugging Face Hub.
   - On container startup, download into `/app/model` before the API starts:
     ```bash
     mkdir -p /app/model
     curl -L "<YOUR_MODEL_URL>" -o /app/model/model.safetensors
     ```

3. **Mount via a volume**
   - Keep the model files on persistent storage.
   - Mount that path into the container as `/app/model` (similar to the Docker `-v /path/to/model:/app/model` examples above).

As long as the final container has the expected files under `/app/model` (mapped from local `model/`), **the API will load and serve the model correctly**, regardless of the fact that the large weight files themselves are not tracked in Git.
