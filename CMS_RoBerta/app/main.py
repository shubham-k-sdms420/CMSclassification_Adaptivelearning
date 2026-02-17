"""
CMS Complaint Classification API — FastAPI service for XLM-RoBERTa complaint classification
with hybrid ensemble (transformer + adaptive SGDClassifier). Two-case routing:
  - Good confidence > 80% -> Accept
  - Low confidence <= 80% -> Human feedback
Serves on port 5015. Loads model from MODEL_DIR (default: ../model).
"""
import json
import os
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .ensemble import AdaptiveEnsemble

# Path to the trained model (saved by train_indicbert.py: trainer.save_model + tokenizer + label2id.json)
MODEL_DIR = os.environ.get("MODEL_DIR", str(Path(__file__).resolve().parent.parent / "model"))
MAX_LENGTH = 512  # Match Stage 2 XLM-R training

app = FastAPI(
    title="CMS Complaint Classification API",
    description="Classify PMC/customer complaints into categories using fine-tuned IndicBERT.",
    version="1.0.0",
)

# Load once at startup
model = None
tokenizer = None
id2label = None
ensemble = None


@app.on_event("startup")
def load_model():
    global model, tokenizer, id2label, ensemble
    model_path = Path(MODEL_DIR)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}. "
            "Set MODEL_DIR or place the trained model in ./model (see README)."
        )
    # Support both label2id.json (flat) and label_mappings.json (nested)
    label2id_path = model_path / "label2id.json"
    label_mappings_path = model_path / "label_mappings.json"
    if label2id_path.exists():
        with open(label2id_path) as f:
            label2id = json.load(f)
    elif label_mappings_path.exists():
        with open(label_mappings_path) as f:
            data = json.load(f)
        label2id = data.get("label2id") or data
        if not isinstance(label2id, dict) or not label2id:
            raise ValueError("label_mappings.json must contain a 'label2id' object.")
    else:
        raise FileNotFoundError(
            f"Neither label2id.json nor label_mappings.json found in {model_path}. "
            "Please add the label mapping file from your trained model."
        )
    id2label = {int(i): l for l, i in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print(f"Model loaded from {model_path}")

    # Hybrid ensemble: transformer + adaptive classifier (loads from model dir if present)
    adaptive_path = model_path / "adaptive_classifier.pkl"
    ensemble = AdaptiveEnsemble(
        transformer_model=model,
        transformer_tokenizer=tokenizer,
        id2label=id2label,
        adaptive_model_path=adaptive_path,
    )
    print("Ensemble ready (transformer + adaptive classifier)")


class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Complaint text to classify")
    return_probabilities: bool = Field(False, description="If true, return per-class probabilities")


class ClassifyResponse(BaseModel):
    label: str = Field(..., description="Predicted category")
    label_id: int = Field(..., description="Numeric label id")
    confidence: Optional[float] = Field(None, description="Ensemble confidence (0-1)")
    routing: Optional[str] = Field(None, description="accept (good confidence >80%) or human_feedback (low confidence)")
    probabilities: Optional[dict] = Field(None, description="Label -> probability (if requested)")
    # Ensemble metrics (for UI)
    transformer_label: Optional[str] = Field(None, description="RoBERTa predicted category")
    transformer_confidence: Optional[float] = Field(None, description="RoBERTa confidence (0-1)")
    adaptive_label: Optional[str] = Field(None, description="SGD classifier predicted category")
    adaptive_confidence: Optional[float] = Field(None, description="SGD classifier confidence (0-1)")
    agreement: Optional[bool] = Field(None, description="True if both models predicted same category")


class BatchClassifyRequest(BaseModel):
    texts: List[str] = Field(..., description="List of complaint texts to classify")
    return_probabilities: bool = Field(False, description="If true, return per-class probabilities for each")


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = True


class FeedbackRequest(BaseModel):
    complaint_text: str = Field(..., description="Original complaint text")
    correct_category: str = Field(..., description="Category selected by human")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.get("/ui", include_in_schema=False)
def serve_ui():
    """Serve the testing UI at /ui (works locally and when deployed)."""
    ui_path = Path(__file__).resolve().parent / "ui.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(ui_path, media_type="text/html")


@app.get("/labels")
def list_labels():
    """Return list of possible classification labels."""
    if id2label is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"labels": [id2label[i] for i in sorted(id2label)]}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    """Classify a single complaint. Returns label, confidence, and routing (accept | human_feedback)."""
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")
    result = ensemble.predict([text], return_probabilities=req.return_probabilities)
    pred = result["predictions"][0]
    out = ClassifyResponse(
        label=pred["label"],
        label_id=pred["label_id"],
        confidence=pred["confidence"],
        routing=pred["routing"],
        transformer_label=pred.get("transformer_label"),
        transformer_confidence=pred.get("transformer_confidence"),
        adaptive_label=pred.get("adaptive_label"),
        adaptive_confidence=pred.get("adaptive_confidence"),
        agreement=pred.get("agreement"),
    )
    if req.return_probabilities and "probabilities" in pred:
        out.probabilities = pred["probabilities"]
    return out


@app.post("/classify/batch")
def classify_batch(req: BatchClassifyRequest):
    """Classify multiple complaint texts. Returns predictions with confidence and routing."""
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    texts = [t.strip() or "" for t in (req.texts or []) if t.strip()]
    if not texts:
        return {"predictions": []}
    result = ensemble.predict(texts, return_probabilities=req.return_probabilities)
    predictions = []
    for pred in result["predictions"]:
        entry = {
            "label": pred["label"],
            "label_id": pred["label_id"],
            "confidence": pred["confidence"],
            "routing": pred["routing"],
        }
        if req.return_probabilities and "probabilities" in pred:
            entry["probabilities"] = pred["probabilities"]
        predictions.append(entry)
    return {"predictions": predictions}


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Submit human feedback: correct category for a complaint. Updates the adaptive classifier."""
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    text = (req.complaint_text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="complaint_text must be non-empty")
    category = (req.correct_category or "").strip()
    if not category:
        raise HTTPException(status_code=400, detail="correct_category must be non-empty")
    if category not in id2label.values():
        raise HTTPException(
            status_code=400,
            detail=f"correct_category must be one of the 78 labels (e.g. use GET /labels). Got: {category[:50]}...",
        )
    ensemble.learn([text], [category])
    adaptive_path = Path(MODEL_DIR) / "adaptive_classifier.pkl"
    ensemble.save_adaptive_model(adaptive_path)
    return {"status": "learned", "message": "Adaptive classifier updated with feedback"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5015)
