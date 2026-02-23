"""
CMS Complaint Classification API — FastAPI service for XLM-RoBERTa complaint classification
with hybrid ensemble (transformer + adaptive SGDClassifier). Two-case routing:
  - Good confidence > 80% -> Accept
  - Low confidence <= 80% -> Human feedback
Serves on port 5016. Loads model from MODEL_DIR (default: ../model).
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
from .feedback_store import (
    add_feedback,
    add_feedback_batch,
    get_category_feedback_counts,
    get_complaint_hash,
    get_feedback_category,
    has_feedback,
    init_db as init_feedback_db,
)

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
    # Weights can be overridden via env for stronger SGD influence after bulk feedback
    tw = os.environ.get("ENSEMBLE_TRANSFORMER_WEIGHT")
    aw = os.environ.get("ENSEMBLE_ADAPTIVE_WEIGHT")
    if tw is not None and aw is not None:
        transformer_weight = float(tw)
        adaptive_weight = float(aw)
    elif aw is not None:
        adaptive_weight = float(aw)
        transformer_weight = 1.0 - adaptive_weight
    elif tw is not None:
        transformer_weight = float(tw)
        adaptive_weight = 1.0 - transformer_weight
    else:
        transformer_weight = 0.7
        adaptive_weight = 0.3
    adaptive_path = model_path / "adaptive_classifier.pkl"
    ensemble = AdaptiveEnsemble(
        transformer_model=model,
        transformer_tokenizer=tokenizer,
        id2label=id2label,
        adaptive_model_path=adaptive_path,
        transformer_weight=transformer_weight,
        adaptive_weight=adaptive_weight,
    )
    print(f"Ensemble ready (transformer={transformer_weight}, adaptive={adaptive_weight})")
    init_feedback_db()


class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Complaint text to classify")
    return_probabilities: bool = Field(False, description="If true, return per-class probabilities")


class ClassifyResponse(BaseModel):
    label: str = Field(..., description="Predicted category")
    label_id: int = Field(..., description="Numeric label id")
    confidence: Optional[float] = Field(None, description="Ensemble confidence (0-1)")
    routing: Optional[str] = Field(None, description="accept (good confidence >80%) or human_feedback (low confidence)")
    probabilities: Optional[dict] = Field(None, description="Label -> probability (if requested)")
    # Feedback routing: don't ask twice for same complaint
    complaint_hash: Optional[str] = Field(None, description="Hash of complaint text; use when submitting feedback")
    needs_feedback: Optional[bool] = Field(None, description="True if UI should show feedback form (routing=human_feedback and not already learned)")
    already_learned: Optional[bool] = Field(None, description="True if we already have feedback for this exact complaint")
    previous_corrected_category: Optional[str] = Field(None, description="When already_learned=true, the category you previously submitted")
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


class FeedbackItem(BaseModel):
    complaint_text: str = Field(..., description="The complaint text")
    correct_category: str = Field(..., description="The correct category label")

class BatchFeedbackRequest(BaseModel):
    feedbacks: List[FeedbackItem] = Field(..., description="List of feedback objects with complaint_text and correct_category")


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
    complaint_hash = get_complaint_hash(text)
    already_learned = has_feedback(complaint_hash)
    previous_corrected_category = get_feedback_category(complaint_hash) if already_learned else None
    category_counts = get_category_feedback_counts()
    try:
        result = ensemble.predict(
            [text],
            return_probabilities=req.return_probabilities,
            category_feedback_counts=category_counts,
        )
    except Exception as e:
        import traceback
        print(f"Classify error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
    pred = result["predictions"][0]
    routing = pred["routing"]
    needs_feedback = routing == "human_feedback" and not already_learned
    # When we already have feedback, show the human-corrected category as the result
    if already_learned and previous_corrected_category:
        label = previous_corrected_category
        label_id = ensemble.label2id.get(previous_corrected_category, pred["label_id"])
    else:
        label = pred["label"]
        label_id = pred["label_id"]
    out = ClassifyResponse(
        label=label,
        label_id=label_id,
        confidence=pred["confidence"],
        routing=routing,
        complaint_hash=complaint_hash,
        needs_feedback=needs_feedback,
        already_learned=already_learned,
        previous_corrected_category=previous_corrected_category,
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
    """Classify multiple complaint texts. Returns predictions with confidence, routing, and feedback flags."""
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    texts = [t.strip() or "" for t in (req.texts or []) if t.strip()]
    if not texts:
        return {"predictions": []}
    category_counts = get_category_feedback_counts()
    result = ensemble.predict(
        texts,
        return_probabilities=req.return_probabilities,
        category_feedback_counts=category_counts,
    )
    predictions = []
    for i, pred in enumerate(result["predictions"]):
        text = texts[i] if i < len(texts) else ""
        complaint_hash = get_complaint_hash(text) if text else ""
        already_learned = has_feedback(complaint_hash) if complaint_hash else False
        previous_corrected_category = get_feedback_category(complaint_hash) if already_learned else None
        routing = pred["routing"]
        needs_feedback = routing == "human_feedback" and not already_learned
        entry = {
            "label": pred["label"],
            "label_id": pred["label_id"],
            "confidence": pred["confidence"],
            "routing": routing,
            "complaint_hash": complaint_hash,
            "needs_feedback": needs_feedback,
            "already_learned": already_learned,
            "previous_corrected_category": previous_corrected_category,
        }
        if req.return_probabilities and "probabilities" in pred:
            entry["probabilities"] = pred["probabilities"]
        predictions.append(entry)
    return {"predictions": predictions}


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Submit human feedback: correct category for a complaint. Updates the adaptive classifier."""
    try:
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
        add_feedback(complaint_hash=get_complaint_hash(text), complaint_text=text, corrected_category=category)
        adaptive_path = Path(MODEL_DIR) / "adaptive_classifier.pkl"
        adaptive_path.parent.mkdir(parents=True, exist_ok=True)
        ensemble.save_adaptive_model(adaptive_path)
        return {"status": "learned", "message": "Adaptive classifier updated with feedback"}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error updating adaptive classifier: {str(e)}\n{traceback.format_exc()}"
        print(f"Feedback error: {error_detail}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/feedback/batch")
def feedback_batch(req: BatchFeedbackRequest):
    """Submit multiple feedback samples at once. More efficient than individual /feedback calls."""
    try:
        if ensemble is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        if not req.feedbacks:
            raise HTTPException(status_code=400, detail="feedbacks list must be non-empty")
        
        texts = []
        categories = []
        for i, fb in enumerate(req.feedbacks):
            text = (fb.complaint_text or "").strip()
            category = (fb.correct_category or "").strip()
            if not text:
                raise HTTPException(status_code=400, detail=f"Feedback {i}: complaint_text must be non-empty")
            if not category:
                raise HTTPException(status_code=400, detail=f"Feedback {i}: correct_category must be non-empty")
            if category not in id2label.values():
                raise HTTPException(
                    status_code=400,
                    detail=f"Feedback {i}: correct_category must be one of the 78 labels. Got: {category[:50]}...",
                )
            texts.append(text)
            categories.append(category)
        
        ensemble.learn(texts, categories)
        add_feedback_batch([
            (get_complaint_hash(t), t, c) for t, c in zip(texts, categories)
        ])
        adaptive_path = Path(MODEL_DIR) / "adaptive_classifier.pkl"
        adaptive_path.parent.mkdir(parents=True, exist_ok=True)
        ensemble.save_adaptive_model(adaptive_path)
        return {
            "status": "learned",
            "message": f"Adaptive classifier updated with {len(texts)} feedback samples",
            "count": len(texts)
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error updating adaptive classifier: {str(e)}\n{traceback.format_exc()}"
        print(f"Batch feedback error: {error_detail}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5016)
