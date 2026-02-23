# How the Adaptive Learning System Works

This document explains in simple terms how the complaint classification system works with the RoBERTa model and when it **accepts** a prediction or asks for **human feedback**.

---

## What the System Does

When a citizen submits a complaint (in English or Marathi), the system:

1. **Predicts** which of the 78 PMC categories the complaint belongs to (e.g. Street Lights, Garbage, Drainage).
2. **Decides** whether to use that prediction as-is (**Accept**) or send it to a human to choose the correct category (**Human feedback**).
3. **Learns** from human corrections so that similar complaints can be handled better next time.

---

## The Two Models (Working Together)

The system uses **two models** that both look at the **same complaint text**. They do not run one after the other; they run **in parallel**, and then their answers are combined.

### 1. RoBERTa Model (Main, Fixed)

- **What it is:** A large AI model (XLM-RoBERTa) trained on complaint data. It understands the **meaning** of the text.
- **What it does:** Reads the complaint and outputs (i) a category and (ii) how confident it is (0–100%).
- **Important:** This model is **never changed** after deployment. It always gives the same kind of answer for the same kind of input.

### 2. SGD Classifier (Adaptive, Learns Over Time)

- **What it is:** A lighter model that uses word/phrase patterns (TF-IDF) instead of deep meaning.
- **What it does:** Also reads the same complaint and outputs a category and a confidence score.
- **Important:** This model **can learn**. When a human corrects a prediction, this model is updated so that next time a similar complaint comes, it can do better.

### How They Are Combined (Ensemble)

- The system takes the **same complaint** and gets:
  - From RoBERTa: category A, confidence X%
  - From SGD: category B, confidence Y%
- It then **combines** these (with more weight on RoBERTa, e.g. 70% RoBERTa, 30% SGD) to get:
  - One **final category**
  - One **final confidence** (a single number between 0 and 100%)

That single final confidence is what the system uses to decide: **Accept** or **Human feedback**.

---

## The Two Outcomes (Accept vs Human Feedback)

There are **only two** possible outcomes for each complaint:

| Outcome           | When it happens              | What the system does                                      |
|-------------------|------------------------------|-----------------------------------------------------------|
| **Accept**        | Final confidence **greater than 80%** | Uses the predicted category. No human review.             |
| **Human feedback**| Final confidence **80% or less**      | Sends to human; human picks the correct category; system learns. |

So:

- **Above 80%** → system is confident → **Accept** (use prediction).
- **80% or below** → system is unsure → **Human feedback** (human decides, system learns).

---

## Step-by-Step: What Happens When a Complaint Arrives

```
Citizen submits complaint text
         │
         ▼
┌─────────────────────────────────────┐
│  Same text is sent to BOTH models   │
│  • RoBERTa (fixed)                  │
│  • SGD classifier (adaptive)        │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Combine their answers               │
│  → One final category               │
│  → One final confidence (0–100%)    │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Is confidence > 80%?               │
└─────────────────────────────────────┘
         │
    ┌────┴────┐
    │         │
   YES       NO
    │         │
    ▼         ▼
 Accept   Human feedback
 (done)   (human picks category
           → system learns)
```

---

## Scenario 1: Clear Complaint → Accept

**Example complaint:**  
*"Street light not working near my house for the past one week."*

- **RoBERTa:** Predicts something like "Street Lights" with high confidence (e.g. 92%).
- **SGD:** Also predicts "Street Lights" with high confidence (e.g. 88%).
- **Combined:** Final category = "Street Lights", final confidence = high (e.g. 90%).
- **Decision:** 90% > 80% → **Accept**. The system returns "Street Lights" and no human review is needed.

**In short:** When the complaint is clear and both models agree, confidence is high and the system **accepts** the prediction.

---

## Scenario 2: Unclear Complaint → Human Feedback

**Example complaint:**  
*"Problem at ward 12 near the temple. Please look into it."*

- **RoBERTa:** Might predict one category (e.g. "General Complaints") with low confidence (e.g. 55%).
- **SGD:** Might predict another category or the same with low confidence (e.g. 50%).
- **Combined:** Final confidence stays low (e.g. 52%).
- **Decision:** 52% ≤ 80% → **Human feedback**. The complaint is sent to a human. The human selects the correct category (e.g. "Road, pavement, divider, pits..." or "Drainage", etc.).

**In short:** When the complaint is vague or could fit many categories, confidence is low and the system asks for **human feedback**.

---

## Scenario 3: Human Gives Correction → System Learns

**What happens:**

1. A complaint got **Human feedback** (low confidence).
2. A human looked at it and selected the **correct category** (e.g. "Garbage Depot Complaint").
3. The user (or UI) calls the **feedback** API with:
   - the same complaint text, and  
   - the correct category.

**What the system does:**

- It **updates only the SGD classifier** with this (complaint text, correct category) pair.
- The RoBERTa model is **not** changed.
- The correction is **stored** (e.g. in `CMS_RoBerta/model/feedback.db`) so the system knows this complaint was already learned.
- The updated SGD classifier is **saved** to `CMS_RoBerta/model/adaptive_classifier.pkl` so the learning is kept for the next time the server runs.

**In short:** Human correction is used to **teach the SGD model** so that similar complaints can be handled better in the future. If the **same complaint text** is submitted again, the system will **not** ask for feedback again; it will show the previously corrected category and `already_learned: true`.

---

## Scenario 4: Similar Complaint Later → Better Result (After Learning)

**Example:**

- Earlier: *"Uncleared garbage in front of our society gate for one week"* → low confidence → human said correct category is "Garbage Depot Complaint" → feedback was submitted.
- Later: A **similar** complaint comes, e.g. *"Garbage not collected from our society gate. Bin is full."*

**What happens:**

- **RoBERTa:** May still give a different or uncertain answer (it was not updated).
- **SGD:** Has **learned** from the earlier feedback. It may now predict "Garbage Depot Complaint" with higher confidence.
- **Combined:** Because SGD is more confident and agrees (or adds weight) toward the right category, the **final confidence can go up** (e.g. above 80%).
- **Decision:** System may now **Accept** this similar complaint instead of sending it to human.

**In short:** After human feedback, the **system improves on similar complaints** without changing the RoBERTa model.

---

## Scenario 5: RoBERTa and SGD Disagree

**Example:**  
RoBERTa says "Road, pavement, divider, pits..." with 65% confidence. SGD says "Drainage" with 75% confidence.

- The system **combines** the two (with weights, e.g. 70% RoBERTa, 30% SGD).
- The **final category** is chosen by which side (after weighting) has the stronger support.
- The **final confidence** is a mix of both; it might still be below 80%.
- **Decision:** If final confidence ≤ 80% → **Human feedback**. The human’s choice can then be sent as feedback so that the SGD model learns for next time.

**In short:** Disagreement often keeps confidence lower, so the system tends to ask for **human feedback** and then learn from it.

---

## Summary Table

| Situation                    | Typical result        | What you see                    |
|-----------------------------|------------------------|----------------------------------|
| Clear, common complaint     | High confidence        | **Accept** – use prediction     |
| Vague or ambiguous complaint| Low confidence         | **Human feedback** – human picks category |
| Human submits correct category | SGD model updated   | Next similar complaint can get **Accept** |
| Both models agree           | Confidence often high  | **Accept**                       |
| Both models disagree        | Confidence often low   | **Human feedback**              |

---

## Important Points

1. **RoBERTa is never retrained** in this system. It stays fixed. Only the **SGD classifier** is updated when you submit feedback.
2. **Routing is decided by confidence.** If **RoBERTa confidence is ≥ 80%**, the outcome is **Accept** (RoBERTa’s prediction is used). Otherwise, the **combined** (ensemble) confidence is used: above 80% → **Accept**; 80% or below → **Human feedback**.
3. **Same complaint not asked twice.** If the system already has feedback for the exact same complaint text, it returns `already_learned: true` and the previously corrected category; it does **not** show the feedback form again.
4. **Learning is from human feedback.** Every time a human selects the correct category for a low-confidence complaint and that is sent to the feedback API, the system stores it (e.g. in `feedback.db`) and updates the SGD model; it gets better on similar complaints in the future.
5. **This system uses a Human-in-the-Loop (HITL) adaptive approach:**
   - **Correction and validation:** When the system is unsure (low confidence), a human reviews the complaint and provides the **correct category**. That correction is used to update the adaptive (SGD) model immediately.
   - **Hybrid intelligence:** The machine (RoBERTa + SGD) does the heavy lifting and handles clear cases automatically; humans step in for **edge cases** (ambiguous or rare complaints) where contextual judgment is needed.
   - The system combines the **data-processing power** of the two models with **human judgment** when the models are not confident, and it **adapts** from those human corrections over time.

This is how the adaptive learning system works with the RoBERTa model in simple terms.

---

## Related

- **Bulk retraining and accuracy:** The **Testing** folder contains a feedback simulator and scripts to run automated feedback over a CSV, apply category mapping, and train on weak categories. See the project root **readme.md** and **Testing/README.md** for current accuracy (~90.6%) and retraining steps.
- **API details:** See **docs/API_DOCUMENTATION.md** for full request/response fields (including `complaint_hash`, `needs_feedback`, `already_learned`, `previous_corrected_category`).
