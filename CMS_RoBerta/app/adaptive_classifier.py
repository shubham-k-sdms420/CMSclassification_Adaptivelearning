"""
Adaptive (online) classifier for the hybrid ensemble.
Uses SGDClassifier + TF-IDF: learns from feedback via partial_fit()
and provides probability estimates for confidence-based routing.
"""
import logging
import pickle
from pathlib import Path
from typing import List, Optional


class _NumpyCompatUnpickler(pickle.Unpickler):
    """
    Remaps `numpy._core` (NumPy 2.x internal module path) to `numpy.core`
    so pkl files saved with NumPy 2.x can be loaded under NumPy 1.x.
    """

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

logger = logging.getLogger(__name__)


class AdaptiveClassifier:
    """
    Lightweight online classifier: TF-IDF + SGDClassifier (modified_huber).
    Used as the adaptive part of the ensemble; learns from human feedback.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            strip_accents="unicode",
            sublinear_tf=True,
            min_df=1,  # Allow single occurrence for online learning
            max_df=0.95,  # Will be adjusted dynamically for small initial batches
        )
        self.classifier = SGDClassifier(
            loss="modified_huber",
            learning_rate="adaptive",
            eta0=0.01,
            alpha=1e-4,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            # Note: class_weight="balanced" is not supported with partial_fit
            # Using None (uniform weights) for online learning compatibility
            class_weight=None,
            warm_start=True,
        )
        self.is_trained = False
        self.classes_: Optional[List[str]] = None
        self.model_path = model_path

    def fit(self, texts: List[str], labels: List[str]) -> None:
        """Initial training on a batch of (text, label) pairs."""
        logger.info("Training adaptive classifier on %d examples", len(texts))
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.classes_ = list(self.classifier.classes_)
        self.is_trained = True
        logger.info("Adaptive classifier trained with %d classes", len(self.classes_))

    def partial_fit(
        self,
        texts: List[str],
        labels: List[str],
        classes: Optional[List[str]] = None,
    ) -> None:
        """Update the classifier with new examples (online learning)."""
        if classes is not None:
            self.classes_ = list(classes)
        if not self.is_trained:
            # For initial training, adjust vectorizer params to handle single-sample training
            # max_df and min_df conflict when sample count is very low
            n_samples = len(texts)
            # When training with few samples (especially 1), we need relaxed parameters
            if n_samples < 10:
                # Create a new vectorizer with relaxed params for small sample training
                self.vectorizer = TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                    sublinear_tf=True,
                    min_df=1,  # Allow single occurrence
                    max_df=1.0,  # Don't filter by max_df for small samples (1.0 = 100%)
                )
            # Now fit_transform with the (possibly adjusted) vectorizer
            X = self.vectorizer.fit_transform(texts)
            if self.classes_ is None:
                self.classes_ = list(sorted(set(labels)))
            self.classifier.partial_fit(X, labels, classes=self.classes_)
            self.is_trained = True
        else:
            X = self.vectorizer.transform(texts)
            # Always pass classes parameter to avoid errors with unseen labels
            if self.classes_ is None:
                raise ValueError("classes_ must be set when classifier is trained")
            self.classifier.partial_fit(X, labels, classes=self.classes_)
        logger.debug("Updated adaptive classifier with %d new examples", len(texts))

    def predict(self, texts: List[str]) -> List[str]:
        """Predict category for each text."""
        if not self.is_trained:
            raise ValueError("Adaptive classifier not trained yet")
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X).tolist()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Return probability estimates per class (for confidence).
        
        Handles both classifiers with predict_proba (modified_huber) and
        those without (hinge) by using decision_function + softmax.
        """
        if not self.is_trained:
            raise ValueError("Adaptive classifier not trained yet")
        X = self.vectorizer.transform(texts)
        
        # Try predict_proba first (works for modified_huber, log_loss, etc.)
        try:
            return self.classifier.predict_proba(X)
        except AttributeError:
            # Fallback for hinge loss: use decision_function + softmax
            logger.debug("predict_proba not available, using decision_function + softmax")
            decision_scores = self.classifier.decision_function(X)
            
            # Handle both binary and multiclass cases
            if decision_scores.ndim == 1:
                # Binary classification: convert to 2D array
                decision_scores = np.column_stack([-decision_scores, decision_scores])
            
            # Apply softmax to convert scores to probabilities
            # Subtract max for numerical stability
            exp_scores = np.exp(decision_scores - decision_scores.max(axis=1, keepdims=True))
            probas = exp_scores / exp_scores.sum(axis=1, keepdims=True)
            
            return probas

    def save(self, path: Path) -> None:
        """Persist classifier and vectorizer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "classifier": self.classifier,
                    "vectorizer": self.vectorizer,
                    "is_trained": self.is_trained,
                    "classes_": self.classes_,
                },
                f,
            )
        logger.info("Saved adaptive classifier to %s", path)

    def load(self, path: Path) -> None:
        """Load classifier and vectorizer from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            data = _NumpyCompatUnpickler(f).load()
        self.classifier = data["classifier"]
        self.vectorizer = data["vectorizer"]
        self.is_trained = data["is_trained"]
        self.classes_ = data.get("classes_")
        logger.info("Loaded adaptive classifier from %s", path)
