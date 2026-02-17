# Hybrid Ensemble Adaptive Learning - Integration Guide

## Overview

This document details the integration of **confidence-based routing** with **Hybrid Ensemble** approach for adaptive complaint classification. Routing is based on **exactly two cases** (no other tiers).

---

## Two Cases for Implementation

The system has **only two routing outcomes** based on ensemble confidence:

| Case | Condition | Action |
|------|-----------|--------|
| **1. Good confidence** | Confidence **> 80%** | **Accept** — Use model prediction; return result to user; no human review. |
| **2. Low confidence** | Confidence **≤ 80%** (i.e. &lt; 80%) | **Human feedback** — Send to human review queue; human selects correct category; system learns from feedback. |

Implementation must treat every prediction as either **Accept** or **Human feedback**; there are no other branches (e.g. no "medium" band).

---

## System Architecture

### Core Components

1. **XLM-RoBERTa Model** (Static, Primary Classifier)

   - Fine-tuned transformer model
   - Provides baseline predictions with confidence scores
   - Never modified directly
2. **Online Adaptive Classifier** (Dynamic, Secondary Classifier)

   - **SGDClassifier** with `loss='modified_huber'` (provides probability estimates)
   - **TF-IDF Vectorizer** for feature extraction (max_features=5000, ngram_range=(1,2))
   - Learns from feedback in real-time via `partial_fit()`
   - Adapts to new patterns quickly (milliseconds update time)
   - Provides confidence scores via `predict_proba()` for routing decisions
3. **Ensemble Combiner**

   - Weighted voting mechanism
   - Confidence calculation
   - Decision routing logic
4. **Confidence-Based Router**

   - **Good confidence (>80%):** Accept — return prediction, no human review.
   - **Low confidence (≤80%):** Human feedback — queue for review, human selects category, then learning pipeline runs.
   - Only these two cases; no other routing branches.
5. **Feedback Learning System**

   - Stores human corrections
   - Updates online classifier immediately
   - Maintains learning buffer

---

## Online Classifier Selection: SGDClassifier

### **SGDClassifier with Modified Huber Loss**

For the ensemble's online adaptive classifier, we use **SGDClassifier** with specific configuration optimized for complaint classification. Here's why this is the perfect choice:

#### **Why SGDClassifier?**

**1. Probability Estimates (Critical for Confidence Calculation)**

- **SGDClassifier with `loss='modified_huber'`** provides probability estimates via `predict_proba()`
- This is **essential** for calculating confidence scores needed for routing decisions
- **Impact:** We need confidence scores to determine if prediction is >80% (auto-accept) or <80% (human review)

**2. Multi-Class Classification Support**

- SGDClassifier handles **78 categories** efficiently
- Supports `class_weight='balanced'` for imbalanced categories
- **Impact:** Your system has 78 complaint categories - SGDClassifier handles this well

**3. Learning Rate Control**

- SGDClassifier offers `learning_rate='adaptive'` which adjusts learning rate automatically
- Can fine-tune with `eta0` parameter for optimal convergence
- **Impact:** Gradual learning prevents overfitting to individual feedback examples

**4. Regularization for Stability**

- L2 regularization (`penalty='l2'`, `alpha=1e-4`) prevents overfitting
- Important when learning from small batches of feedback
- **Impact:** More stable learning, especially with limited feedback data

**5. Partial Fit Efficiency**

- SGDClassifier supports `partial_fit()` for online learning
- Fast incremental updates and good memory efficiency for large feature spaces (TF-IDF vectors)
- **Impact:** Real-time updates when human provides feedback

**6. Production Stability**

- SGDClassifier is stable with varying data distributions
- Less sensitive to outliers in feedback
- **Impact:** More reliable in production environment

#### **SGDClassifier Capabilities**

| Feature                               | SGDClassifier                         |
| ------------------------------------- | ------------------------------------- |
| **Probability Estimates**       | ✅ Yes (modified_huber)               |
| **Confidence Scores**           | ✅ Native support via predict_proba() |
| **Multi-Class (78 categories)** | ✅ Excellent                          |
| **Learning Rate Control**       | ✅ Adaptive/Constant                  |
| **Regularization**              | ✅ L1/L2/ElasticNet                   |
| **Stability**                   | ✅ High                               |
| **Update Speed**                | ✅ Fast (partial_fit)                 |
| **Memory Efficiency**           | ✅ High                               |
| **Handles Imbalance**           | ✅ class_weight='balanced'            |

#### **Recommended Configuration**

```python
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Online Adaptive Classifier
classifier = SGDClassifier(
    loss='modified_huber',      # Provides probability estimates
    learning_rate='adaptive',   # Auto-adjusts learning rate
    eta0=0.01,                  # Initial learning rate
    alpha=1e-4,                 # L2 regularization strength
    max_iter=1000,              # Max iterations per update
    tol=1e-3,                   # Convergence tolerance
    random_state=42,            # Reproducibility
    class_weight='balanced',    # Handle imbalanced categories
    warm_start=True             # Efficient partial_fit
)

# Feature Extraction
vectorizer = TfidfVectorizer(
    max_features=5000,         # Limit feature space
    ngram_range=(1, 2),        # Unigrams + bigrams
    strip_accents='unicode',    # Handle multilingual text
    sublinear_tf=True,          # Log scaling for TF
    min_df=2,                   # Ignore rare terms
    max_df=0.95                 # Ignore very common terms
)
```

#### **Final Recommendation**

**Use SGDClassifier** because:

- ✅ **Critical:** Provides probability estimates for confidence-based routing
- ✅ **Essential:** Better handles 78-category multi-class problem
- ✅ **Important:** More stable learning with regularization
- ✅ **Beneficial:** Adaptive learning rate for optimal convergence
- ✅ **Production-ready:** Proven stability in production systems

---

## Why This Approach is Best

### ✅ **Advantages**

1. **Zero Risk to Production Model**

   - XLM-RoBERTa remains unchanged
   - Online classifier can be reset if needed
   - Easy rollback capability
2. **Immediate Learning**

   - Online classifier updates instantly from feedback
   - No waiting for batch retraining
   - Real-time adaptation
3. **Two-Case Routing**

   - Good confidence (>80%): Accept — no human intervention
   - Low confidence (≤80%): Human feedback — review and learn
   - Only these two cases; simple to implement and reason about
4. **Gradual Adaptation**

   - Online classifier starts with low weight
   - Gradually gains trust as it learns
   - Smooth transition from transformer to ensemble
5. **Handles Edge Cases**

   - New complaint types: Low confidence triggers human review
   - Ambiguous complaints: Human decides, system learns
   - Pattern changes: Online classifier adapts quickly
6. **Production Ready**

   - Simple to implement and maintain
   - Easy to monitor and debug
   - Scalable architecture

---

## Complete Workflow

### Phase 1: Classification Request Flow

```
┌─────────────────────┐
│  New Complaint      │
│  Text Input         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Ensemble Predict   │
│  (XLM-RoBERTa +     │
│   Online Classifier)│
└──────────┬──────────┘
           │
           ├─────────────────┐
           │                 │
           ▼                 ▼
┌──────────────────┐  ┌──────────────────┐
│ XLM-RoBERTa      │  │ SGDClassifier    │
│ Prediction       │  │ (TF-IDF + Online) │
│ Confidence       │  │ Prediction        │
│                  │  │ Confidence        │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Ensemble Combiner    │
         │ - Weighted Voting    │
         │ - Confidence Calc    │
         │ - Agreement Check    │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Final Prediction     │
         │ - Category           │
         │ - Confidence Score   │
         │ - Agreement Status   │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Confidence Router    │
         │ Check: > 80%?        │
         └──────────┬───────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐      ┌───────────────┐
│ Good conf     │      │ Low conf      │
│ (> 80%)       │      │ (≤ 80%)       │
└───────┬───────┘      └───────┬───────┘
        │                     │
        ▼                     ▼
┌───────────────┐      ┌───────────────┐
│ Accept        │      │ Human Feedback│
│ Return result │      │ Queue         │
└───────────────┘      └───────┬───────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ Human Selects │
                        │ Correct Label │
                        └───────┬───────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ Learning      │
                        │ Pipeline      │
                        └───────────────┘
```

### Phase 2: Learning Pipeline Flow

```
┌─────────────────────┐
│ Human Feedback      │
│ - Complaint Text    │
│ - Correct Category  │
│ - Original Pred     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Store in Buffer     │
│ (Learning Queue)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Update SGDClassifier│
│ (partial_fit)       │
│ Immediate Learning   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Adjust Ensemble     │
│ Weights             │
│ (Gradual)           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Log Learning Event  │
│ - Metrics           │
│ - Performance       │
└─────────────────────┘
```

### Phase 3: Adaptive Learning Flow

```
┌─────────────────────┐
│ New Complaint       │
│ Similar to Learned  │
│ Pattern             │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ XLM-RoBERTa:       │
│ Low Confidence      │
│ (New pattern)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ SGDClassifier:     │
│ High Confidence     │
│ (Learned pattern)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Ensemble:           │
│ - Online weight ↑   │
│ - Better prediction │
│ - Higher confidence │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Result:             │
│ Auto-accept         │
│ (Learned!)          │
└─────────────────────┘
```

---

## Detailed Implementation Flow

### Step 1: Complaint Classification Request

**Input:** Complaint text**Process:**

1. Tokenize and encode complaint
2. Get XLM-RoBERTa prediction + confidence
3. Get SGDClassifier prediction + confidence (if trained)
   - TF-IDF vectorization of complaint text
   - SGDClassifier.predict_proba() for confidence scores
4. Combine predictions using ensemble weights
5. Calculate final confidence score
6. Route using the two cases only: >80% → Accept; ≤80% → Human feedback.

**Output:**

- Predicted category
- Confidence score
- Routing decision: **Accept** (good confidence >80%) or **Human feedback** (low confidence ≤80%)

### Step 2: Good Confidence — Accept (>80%)

**When:** Final ensemble confidence > 80%.

**Action:**
1. **Accept** — Return prediction to user immediately.
2. Log classification event.
3. No human review.

**Why:** Model is confident; use prediction as-is.

### Step 3: Low Confidence — Human Feedback (≤80%)

**When:** Final ensemble confidence ≤ 80%.

**Action:**
1. **Human feedback** — Queue complaint for human review.
2. Show prediction and top alternatives.
3. Human selects correct category.
4. Trigger learning pipeline (update SGDClassifier, then continue).

**Why:** Model uncertain; human provides correct label and system learns.

### Step 4: Learning from Feedback

**When:** Human provides correct category**Action:**

1. Store (complaint_text, correct_category, original_prediction)
2. Update SGDClassifier immediately:
   - Transform complaint text using TF-IDF vectorizer
   - Call `SGDClassifier.partial_fit(X_tfidf, [correct_category])`
   - Model updates weights instantly (milliseconds)
3. Adjust ensemble weights:
   - Increase SGDClassifier weight slightly (e.g., 0.30 → 0.31)
   - Decrease transformer weight proportionally (0.70 → 0.69)
4. Log learning event with metrics

**Result:** SGDClassifier learns new pattern instantly via partial_fit()

### Step 5: Future Similar Complaints

**When:** New complaint similar to learned pattern**Process:**

1. XLM-RoBERTa: May still have low confidence (hasn't seen this pattern)
2. SGDClassifier: High confidence (learned from feedback via partial_fit)
   - TF-IDF features match learned pattern
   - predict_proba() returns high confidence for learned category
3. Ensemble: Combines both predictions, SGDClassifier's weight has increased
4. Final confidence: Higher due to ensemble agreement (>80%)
5. Result: Auto-accept (no human needed)

**Why:** SGDClassifier adapted to new pattern through online learning

---

## Confidence Calculation Strategy

### Ensemble Confidence Formula

```python
# Option 1: Weighted Average
final_confidence = (
    transformer_weight * transformer_confidence +
    online_weight * online_confidence
)

# Option 2: Agreement-Based (Recommended)
if transformer_pred == online_pred:
    # Both agree - boost confidence
    final_confidence = max(transformer_confidence, online_confidence) * 1.1
    final_confidence = min(final_confidence, 1.0)  # Cap at 1.0
else:
    # Disagreement - use weighted average
    final_confidence = (
        transformer_weight * transformer_confidence +
        online_weight * online_confidence
    )
```

### Confidence Thresholds (Two Cases Only)

- **Good confidence:** > 80% → **Accept** (use prediction; no human review).
- **Low confidence:** ≤ 80% → **Human feedback** (queue for review; human selects category; system learns).

No other thresholds or bands (e.g. no "medium" tier).

---

## Ensemble Weight Adjustment Strategy

### Initial Weights

- **Transformer Weight:** 0.7 (70%)
- **Online Weight:** 0.3 (30%)

### Dynamic Adjustment

```python
# After each learning event
if online_classifier_accuracy > transformer_accuracy:
    # Online classifier performing better
    online_weight = min(0.5, online_weight + 0.01)
    transformer_weight = 1 - online_weight
elif online_classifier_accuracy < transformer_accuracy - 0.05:
    # Online classifier underperforming
    online_weight = max(0.2, online_weight - 0.01)
    transformer_weight = 1 - online_weight
```

### Weight Limits

- **Online Weight:** 0.2 (min) to 0.5 (max)
- **Transformer Weight:** 0.5 (min) to 0.8 (max)

**Why:** Always trust transformer more, but allow online classifier to contribute

---

## Learning Scenarios

### Scenario 1: New Complaint Type

**Initial State:**

- Complaint: "New type of issue never seen before"
- XLM-RoBERTa: Predicts closest category, confidence 45%
- SGDClassifier: Not trained on this pattern (returns transformer prediction)
- Ensemble: Low confidence → Human review

**After Human Feedback:**

- Human selects correct category
- SGDClassifier learns: `partial_fit(tfidf_vector, "correct_category")`
- SGDClassifier updates weights instantly (milliseconds)
- SGDClassifier weight increases slightly (0.30 → 0.31)

**Next Similar Complaint:**

- XLM-RoBERTa: Still low confidence (hasn't been retrained)
- SGDClassifier: High confidence (learned from feedback via partial_fit)
  - TF-IDF features match learned pattern
  - predict_proba() returns high confidence (>85%)
- Ensemble: Higher confidence (SGDClassifier contributes more)
- Result: Auto-accept if ensemble confidence > 80%

### Scenario 2: Ambiguous Complaint

**Initial State:**

- Complaint: Could belong to multiple categories
- XLM-RoBERTa: Predicts category A, confidence 65%
- SGDClassifier: Predicts category B, confidence 70%
- Ensemble: Disagreement → Lower confidence → Human review

**After Human Feedback:**

- Human selects correct category (e.g., category C)
- SGDClassifier learns correct pattern via partial_fit()
- TF-IDF features capture ambiguity indicators
- System understands ambiguity better

**Future Ambiguous Complaints:**

- SGDClassifier recognizes ambiguity pattern from learned features
- Provides better predictions with higher confidence
- Reduces human review needed

### Scenario 3: Pattern Drift

**Initial State:**

- Old pattern: Complaints about X → Category A
- New pattern: Complaints about X → Category B (changed)
- XLM-RoBERTa: Still predicts Category A (trained on old data)
- SGDClassifier: Learns new pattern from feedback via partial_fit()

**After Multiple Feedback:**

- SGDClassifier adapts to new pattern (TF-IDF captures pattern shift)
- Ensemble confidence increases for new pattern
- System adapts without retraining transformer
- SGDClassifier weight increases as it proves more accurate

---

## Implementation Architecture

### Component Structure

```
CMS_RoBerta/
├── app/
│   ├── main.py                 # FastAPI app (existing)
│   ├── ensemble.py             # NEW: Ensemble combiner
│   ├── adaptive_classifier.py  # NEW: Online classifier
│   ├── confidence_router.py    # NEW: Confidence-based routing
│   └── learning_pipeline.py    # NEW: Learning orchestration
├── model/
│   └── [existing model files]
└── storage/
    ├── adaptive_classifier.pkl # Online classifier state
    ├── learning_buffer.db      # Feedback storage
    └── learning_logs.jsonl     # Learning events log
```

### Data Flow

```
API Request
    │
    ├─→ main.py (FastAPI endpoint)
    │       │
    │       ├─→ ensemble.py
    │       │       │
    │       │       ├─→ XLM-RoBERTa (static)
    │       │       └─→ adaptive_classifier.py (SGDClassifier + TF-IDF)
    │       │
    │       └─→ confidence_router.py
    │               │
    │               ├─→ Good conf (>80%) → Accept (return result)
    │               └─→ Low conf (≤80%) → Human feedback (queue)
    │
    └─→ Feedback endpoint
            │
            └─→ learning_pipeline.py
                    │
                    ├─→ Store in buffer
                    ├─→ Update SGDClassifier (partial_fit)
                    └─→ Adjust ensemble weights
```

---

## API Endpoints Design

### 1. Classify Endpoint (Modified)

```python
POST /classify
{
    "text": "complaint text",
    "return_probabilities": true
}

Response:
{
    "label": "predicted_category",
    "label_id": 59,
    "confidence": 0.85,
    "routing": "accept",  # or "human_feedback" (only these two values)
    "ensemble_details": {
        "transformer_prediction": "...",
        "transformer_confidence": 0.82,
        "sgdclassifier_prediction": "...",
        "sgdclassifier_confidence": 0.88,
        "agreement": true,
        "weights": {
            "transformer": 0.7,
            "sgdclassifier": 0.3
        }
    },
    "probabilities": {...}
}
```

### 2. Feedback Endpoint (New)

```python
POST /feedback
{
    "complaint_text": "original complaint",
    "predicted_category": "what model predicted",
    "correct_category": "what human selected",
    "prediction_id": "unique_id_from_classify"
}

Response:
{
    "status": "learned",
    "message": "Model updated with feedback",
    "learning_details": {
        "sgdclassifier_updated": true,
        "partial_fit_applied": true,
        "ensemble_weights_adjusted": true,
        "new_sgdclassifier_weight": 0.31,
        "new_transformer_weight": 0.69
    }
}
```

### 3. Human Review Queue Endpoint (New)

```python
GET /review/queue
Response:
{
    "pending_reviews": [
        {
            "prediction_id": "id1",
            "complaint_text": "...",
            "predicted_category": "...",
            "confidence": 0.65,
            "top_alternatives": [...],
            "timestamp": "..."
        }
    ]
}

POST /review/submit
{
    "prediction_id": "id1",
    "correct_category": "selected_category"
}
```

---

## Learning Metrics & Monitoring

### Key Metrics to Track

1. **Classification Metrics**

   - Overall accuracy
   - Good-confidence (Accept) path accuracy
   - Low-confidence (Human feedback) path accuracy
   - Confidence distribution (two cases only)
2. **Learning Metrics**

   - Number of feedback events
   - Online classifier accuracy
   - Ensemble weight changes
   - Learning rate (feedback per day)
3. **Efficiency Metrics**

   - Accept rate (good confidence >80%)
   - Human feedback rate (low confidence ≤80%)
   - Average confidence score
   - Time saved (auto-accepts)
4. **Adaptation Metrics**

   - New patterns learned
   - Pattern drift detection
   - Online classifier improvement over time

### Monitoring Dashboard

```
┌─────────────────────────────────────┐
│ Adaptive Learning Dashboard        │
├─────────────────────────────────────┤
│ Today's Stats:                      │
│ - Total Classifications: 1,234      │
│ - Auto-Accepted: 987 (80%)         │
│ - Human Reviewed: 247 (20%)         │
│ - Feedback Received: 189           │
│                                     │
│ Learning Progress:                  │
│ - Online Classifier Accuracy: 87%  │
│ - Ensemble Weight (Online): 0.32   │
│ - Patterns Learned: 45             │
│                                     │
│ Confidence (two cases only):       │
│ - Good (>80%): Accept — 987        │
│ - Low (≤80%): Human feedback — 247│
└─────────────────────────────────────┘
```

---

## Why This Integration is Optimal

### 1. **Leverages Your Existing Logic**

- Exactly two cases: Good confidence (>80%) → Accept; Low confidence (≤80%) → Human feedback
- No other routing branches (e.g. no "medium" tier)

### 2. **Adds Adaptive Learning**

- Online classifier learns from human feedback
- System improves over time
- Reduces need for human review

### 3. **Best of Both Worlds**

- Transformer: Strong baseline, handles known patterns
- Online Classifier: Quick adaptation, learns new patterns
- Ensemble: Combines strengths, provides confidence

### 4. **Production Safe**

- No risk to transformer model
- Can rollback online classifier if needed
- Easy to monitor and debug

### 5. **Efficient Resource Usage**

- Low confidence (≤80%): human feedback path
- Good confidence (>80%): accept path (no human review)
- Learning happens in background

### 6. **Scalable**

- Online classifier updates are fast (milliseconds)
- No GPU needed for learning
- Can handle high volume

---

## Implementation Priority

### Phase 1: Core Ensemble (Week 1)

1. Implement `adaptive_classifier.py`
2. Implement `ensemble.py`
3. Integrate with existing `/classify` endpoint
4. Test basic ensemble predictions

### Phase 2: Confidence Routing (Week 2)

1. Implement `confidence_router.py`
2. Add routing logic to `/classify`
3. Create human review queue
4. Test both cases: Accept (>80%) and Human feedback (≤80%)

### Phase 3: Learning Pipeline (Week 3)

1. Implement `learning_pipeline.py`
2. Add `/feedback` endpoint
3. Implement immediate learning updates
4. Add ensemble weight adjustment

### Phase 4: Monitoring (Week 4)

1. Add metrics collection
2. Create monitoring dashboard
3. Set up logging
4. Performance optimization

---

## Success Criteria

### Short-term (1 month)

- ✅ Ensemble predictions working
- ✅ Confidence routing functional
- ✅ Learning from feedback operational
- ✅ Two-case routing working: Accept (>80%) and Human feedback (≤80%)

### Medium-term (3 months)

- ✅ Online classifier accuracy > 85%
- ✅ Human review rate decreasing
- ✅ New patterns learned automatically
- ✅ System adapting to changes

### Long-term (6 months)

- ✅ Significant reduction in human review
- ✅ High accuracy on learned patterns
- ✅ Stable ensemble performance
- ✅ Production-ready adaptive system

---

## Conclusion

This integrated approach combines:

- **Two-case routing** — Good confidence (>80%) Accept; Low confidence (≤80%) Human feedback
- **Hybrid ensemble** (best of both models)
- **Real-time learning** (immediate adaptation)
- **Production safety** (zero risk to transformer)

The result is a robust, adaptive system that learns from feedback while maintaining high accuracy and efficiency.
