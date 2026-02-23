"""
Hybrid ensemble: XLM-RoBERTa (static) + AdaptiveClassifier (SGDClassifier).
Combines predictions and computes confidence for two-case routing:
  - Good confidence > 80% -> Accept
  - Low confidence <= 80% -> Human feedback
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .adaptive_classifier import AdaptiveClassifier


# Confidence threshold: only two cases
CONFIDENCE_ACCEPT = 0.80  # > this -> Accept; <= this -> Human feedback


class AdaptiveEnsemble:
    """
    Combines transformer model and adaptive classifier.
    Returns prediction, confidence, and routing (accept | human_feedback).
    """

    def __init__(
        self,
        transformer_model: Any,
        transformer_tokenizer: Any,
        id2label: Dict[int, str],
        adaptive_model_path: Optional[Path] = None,
        transformer_weight: float = 0.7,
        adaptive_weight: float = 0.3,
    ):
        self.transformer_model = transformer_model
        self.transformer_tokenizer = transformer_tokenizer
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.transformer_weight = transformer_weight
        self.adaptive_weight = adaptive_weight
        self.adaptive_classifier = AdaptiveClassifier(adaptive_model_path)
        if adaptive_model_path and Path(adaptive_model_path).exists():
            self.adaptive_classifier.load(Path(adaptive_model_path))

    def _transformer_predict(self, texts: List[str]) -> tuple:
        """Run transformer and return (pred_labels, confidences, probas)."""
        inputs = self.transformer_tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.transformer_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = probs.argmax(axis=-1)
        pred_labels = [self.id2label[int(pid)] for pid in pred_ids]
        confidences = [float(probs[i, pred_ids[i]]) for i in range(len(texts))]
        return pred_labels, confidences, probs

    def _adaptive_weights_for_category(self, category: str, category_feedback_counts: Optional[Dict[str, int]] = None) -> tuple:
        """Return (transformer_weight, adaptive_weight). Boost SGD when it has learned from feedback in this category."""
        if not category_feedback_counts:
            return self.transformer_weight, self.adaptive_weight
        count = category_feedback_counts.get(category, 0)
        if count >= 20:
            return 0.5, 0.5
        if count >= 10:
            return 0.6, 0.4
        return self.transformer_weight, self.adaptive_weight

    def predict(
        self,
        texts: List[str],
        return_probabilities: bool = False,
        category_feedback_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Run ensemble and return predictions with confidence and routing.
        Routing: "accept" if confidence > 0.80, else "human_feedback".
        category_feedback_counts: optional dict category -> count; boosts SGD weight for well-learned categories.
        """
        trans_labels, trans_confs, trans_probs = self._transformer_predict(texts)

        if self.adaptive_classifier.is_trained:
            try:
                adapt_labels = self.adaptive_classifier.predict(texts)
                adapt_probs = self.adaptive_classifier.predict_proba(texts)
                adapt_confs = [float(adapt_probs[i].max()) for i in range(len(texts))]
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Adaptive classifier failed, using transformer only: %s", e
                )
                adapt_labels = trans_labels
                adapt_confs = trans_confs
                adapt_probs = trans_probs
        else:
            adapt_labels = trans_labels
            adapt_confs = trans_confs
            adapt_probs = trans_probs

        predictions = []
        for i in range(len(texts)):
            tl, tc = trans_labels[i], trans_confs[i]
            al, ac = adapt_labels[i], adapt_confs[i]

            tw, aw = self._adaptive_weights_for_category(al, category_feedback_counts)

            if tl == al:
                final_conf = min(1.0, max(tc, ac) * 1.05)
                final_label = tl
            else:
                final_conf = tw * tc + aw * ac
                if tw * tc >= aw * ac:
                    final_label = tl
                else:
                    final_label = al

            # If RoBERTa is highly confident, don't send to human feedback just because SGD disagrees
            if tc >= CONFIDENCE_ACCEPT:
                final_label = tl
                final_conf = max(final_conf, tc)
                routing = "accept"
            else:
                routing = "accept" if final_conf > CONFIDENCE_ACCEPT else "human_feedback"

            label_id = self.label2id.get(final_label, 0)

            pred = {
                "label": final_label,
                "label_id": label_id,
                "confidence": round(final_conf, 4),
                "routing": routing,
                "transformer_label": tl,
                "transformer_confidence": round(tc, 4),
                "adaptive_label": al,
                "adaptive_confidence": round(ac, 4),
                "agreement": tl == al,
            }
            if return_probabilities:
                pred["probabilities"] = {
                    self.id2label[j]: round(float(trans_probs[i][j]), 4)
                    for j in range(len(self.id2label))
                }
            predictions.append(pred)

        return {
            "predictions": predictions,
            "weights": {
                "transformer": self.transformer_weight,
                "adaptive": self.adaptive_weight,
            },
        }

    def learn(self, texts: List[str], labels: List[str]) -> None:
        """Update adaptive classifier from human feedback."""
        classes = list(self.id2label.values())
        self.adaptive_classifier.partial_fit(texts, labels, classes=classes)

    def save_adaptive_model(self, path: Path) -> None:
        """Persist adaptive classifier to disk."""
        self.adaptive_classifier.save(Path(path))
