# CMS Complaint Classification API Documentation

**Version:** 1.0.0  
**Base URL:** `http://localhost:5015` (development) or your deployed server URL  
**Content-Type:** `application/json`

## Overview

The CMS Complaint Classification API is a RESTful service that classifies municipal complaints (PMC - Pune Municipal Corporation) into 78 predefined categories using a hybrid ensemble of XLM-RoBERTa (transformer) and SGDClassifier (adaptive). The system provides two-case routing based on confidence: **Accept** (>80%) or **Human feedback** (≤80%).

### Key Features

- **Hybrid Ensemble**: Combines transformer (XLM-RoBERTa) and adaptive classifier (SGDClassifier)
- **Confidence-based Routing**: Automatically routes predictions based on confidence threshold
- **Adaptive Learning**: Learns from human feedback via `/feedback` endpoint
- **Batch Processing**: Supports single and batch classification
- **78 Categories**: Supports bilingual (English/Marathi) complaint classification

---

## Authentication

Currently, the API does not require authentication. For production deployments, consider adding API keys or OAuth2.

---

## Complete Endpoint List

All endpoints are available at the base URL: `http://localhost:5015` (replace with your server URL in production)

| Method | Complete Endpoint URL | Description |
|--------|----------------------|-------------|
| `GET` | `http://localhost:5015/health` | Health check and model status |
| `GET` | `http://localhost:5015/labels` | List all 78 available category labels |
| `POST` | `http://localhost:5015/classify` | Classify a single complaint |
| `POST` | `http://localhost:5015/classify/batch` | Classify multiple complaints in batch |
| `POST` | `http://localhost:5015/feedback` | Submit human feedback to update adaptive classifier |
| `GET` | `http://localhost:5015/ui` | Access interactive web UI |
| `GET` | `http://localhost:5015/docs` | Swagger UI interactive API documentation |
| `GET` | `http://localhost:5015/redoc` | ReDoc API documentation |

### Quick Reference - Complete Endpoints

```
GET    http://localhost:5015/health
GET    http://localhost:5015/labels
POST   http://localhost:5015/classify
POST   http://localhost:5015/classify/batch
POST   http://localhost:5015/feedback
GET    http://localhost:5015/ui
GET    http://localhost:5015/docs
GET    http://localhost:5015/redoc
```

**Note:** Replace `localhost:5015` with your actual server URL in production (e.g., `https://api.yourdomain.com`).

---

## Endpoints

### 1. Health Check

Check if the API is running and the model is loaded.

**Complete Endpoint:** `GET http://localhost:5015/health`  
**Endpoint Path:** `/health`

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

**Example:**
```bash
curl http://localhost:5015/health
```

**Response Codes:**
- `200 OK`: API is healthy and model is loaded
- `503 Service Unavailable`: Model not loaded (check server logs)

---

### 2. List Available Labels

Get the list of all 78 possible classification categories.

**Complete Endpoint:** `GET http://localhost:5015/labels`  
**Endpoint Path:** `/labels`

**Response:**
```json
{
  "labels": [
    "Street Lights",
    "Drainage",
    "Road, pavement, divider, pits, ...",
    ...
  ]
}
```

**Example:**
```bash
curl http://localhost:5015/labels
```

**Response Codes:**
- `200 OK`: Success
- `503 Service Unavailable`: Model not loaded

---

### 3. Classify Single Complaint

Classify a single complaint text and get prediction with confidence and routing decision.

**Complete Endpoint:** `POST http://localhost:5015/classify`  
**Endpoint Path:** `/classify`

**Request Body:**
```json
{
  "text": "Street light not working near my house",
  "return_probabilities": false
}
```

**Request Fields:**
- `text` (string, required): The complaint text to classify
- `return_probabilities` (boolean, optional): If `true`, returns probability distribution across all 78 categories. Default: `false`

**Response:**
```json
{
  "label": "Street Lights",
  "label_id": 42,
  "confidence": 0.92,
  "routing": "accept",
  "complaint_hash": "a1b2c3...",
  "needs_feedback": false,
  "already_learned": false,
  "previous_corrected_category": null,
  "transformer_label": "Street Lights",
  "transformer_confidence": 0.94,
  "adaptive_label": "Street Lights",
  "adaptive_confidence": 0.88,
  "agreement": true
}
```

**Response Fields:**
- `label` (string): Final predicted category (or previous corrected category when `already_learned` is true)
- `label_id` (integer): Numeric ID of the predicted category
- `confidence` (float, 0-1): Final ensemble confidence score
- `routing` (string): `"accept"` if confidence > 80% or RoBERTa confidence ≥ 80%, `"human_feedback"` if ≤ 80%
- `complaint_hash` (string, optional): Hash of complaint text (for use with feedback)
- `needs_feedback` (boolean): True if UI should show feedback form (routing is human_feedback and not already learned)
- `already_learned` (boolean): True if we already have feedback for this exact complaint
- `previous_corrected_category` (string, optional): When already_learned is true, the category previously submitted
- `transformer_label` (string): XLM-RoBERTa predicted category
- `transformer_confidence` (float, 0-1): XLM-RoBERTa confidence
- `adaptive_label` (string): SGD classifier predicted category
- `adaptive_confidence` (float, 0-1): SGD classifier confidence
- `agreement` (boolean): `true` if both models predicted the same category
- `probabilities` (object, optional): Probability distribution if `return_probabilities=true`

**Routing:** If RoBERTa confidence ≥ 80%, the system returns routing "accept" and uses RoBERTa's prediction.

**Example with cURL:**
```bash
curl -X POST http://localhost:5015/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Street light not working near my house",
    "return_probabilities": false
  }'
```

**Example with probabilities:**
```bash
curl -X POST http://localhost:5015/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Drainage water overflowing on the road",
    "return_probabilities": true
  }'
```

**Response with probabilities:**
```json
{
  "label": "Drainage",
  "label_id": 15,
  "confidence": 0.87,
  "routing": "accept",
  "probabilities": {
    "Drainage": 0.87,
    "Road, pavement, divider, pits, ...": 0.08,
    "Water Supply": 0.03,
    ...
  },
  ...
}
```

**Response Codes:**
- `200 OK`: Success
- `400 Bad Request`: Invalid request (empty text, invalid JSON)
- `503 Service Unavailable`: Model not loaded

---

### 4. Classify Batch Complaints

Classify multiple complaints in a single request.

**Complete Endpoint:** `POST http://localhost:5015/classify/batch`  
**Endpoint Path:** `/classify/batch`

**Request Body:**
```json
{
  "texts": [
    "Street light not working",
    "Drainage water overflowing",
    "Road has potholes"
  ],
  "return_probabilities": false
}
```

**Request Fields:**
- `texts` (array of strings, required): List of complaint texts to classify
- `return_probabilities` (boolean, optional): If `true`, returns probabilities for each complaint. Default: `false`

**Response:**
```json
{
  "predictions": [
    {
      "label": "Street Lights",
      "label_id": 42,
      "confidence": 0.92,
      "routing": "accept"
    },
    {
      "label": "Drainage",
      "label_id": 15,
      "confidence": 0.75,
      "routing": "human_feedback"
    },
    {
      "label": "Road, pavement, divider, pits, ...",
      "label_id": 38,
      "confidence": 0.88,
      "routing": "accept"
    }
  ]
}
```

**Example:**
```bash
curl -X POST http://localhost:5015/classify/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Street light not working",
      "Drainage water overflowing"
    ],
    "return_probabilities": false
  }'
```

**Response Codes:**
- `200 OK`: Success (returns empty predictions array if no valid texts)
- `400 Bad Request`: Invalid request (invalid JSON)
- `503 Service Unavailable`: Model not loaded

**Note:** Empty strings in the `texts` array are automatically filtered out.

---

### 5. Submit Feedback

Submit human correction for a low-confidence prediction. This updates the adaptive classifier and improves future predictions.

**Complete Endpoint:** `POST http://localhost:5015/feedback`  
**Endpoint Path:** `/feedback`

**Request Body:**
```json
{
  "complaint_text": "Street light not working near my house",
  "correct_category": "Street Lights"
}
```

**Request Fields:**
- `complaint_text` (string, required): The original complaint text
- `correct_category` (string, required): The correct category label (must be one of the 78 labels from `/labels`)

**Response:**
```json
{
  "status": "learned",
  "message": "Adaptive classifier updated with feedback"
}
```

**Example:**
```bash
curl -X POST http://localhost:5015/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "complaint_text": "Street light not working",
    "correct_category": "Street Lights"
  }'
```

**Response Codes:**
- `200 OK`: Feedback accepted and classifier updated
- `400 Bad Request`: Invalid request (empty text, invalid category, category not in label list)
- `503 Service Unavailable`: Model not loaded

**Important Notes:**
- The `correct_category` must exactly match one of the labels returned by `GET /labels`
- Only the adaptive (SGD) classifier is updated; the transformer model remains unchanged
- The updated classifier is automatically saved to `CMS_RoBerta/model/adaptive_classifier.pkl`
- This improves future predictions for similar complaints

---

### 6. Web UI

Access the interactive web UI for testing and visualization.

**Complete Endpoint:** `GET http://localhost:5015/ui`  
**Endpoint Path:** `/ui`

**Response:** HTML page with interactive UI

**Example:**
Open in browser: `http://localhost:5015/ui`

**Features:**
- Single complaint classification
- Batch classification
- Feedback submission
- Real-time confidence calculation display
- Ensemble metrics visualization

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Codes

| Status Code | Description | Common Causes |
|-------------|-------------|---------------|
| `400 Bad Request` | Invalid request format or missing required fields | Empty text, invalid JSON, invalid category |
| `404 Not Found` | Endpoint not found | Wrong URL path |
| `503 Service Unavailable` | Model not loaded | Server startup issue, model files missing |
| `500 Internal Server Error` | Server error | Unexpected error (check server logs) |

**Example Error Response:**
```json
{
  "detail": "text must be non-empty"
}
```

---

## Confidence Calculation

The final confidence is calculated differently based on model agreement:

### Agreement Case (both models predict same category)
```
final_confidence = min(1.0, max(transformer_confidence, adaptive_confidence) × 1.05)
```
- Takes the maximum of the two confidences
- Applies a 5% boost
- Caps at 1.0 (100%)

### Disagreement Case (models predict different categories)
```
final_confidence = (0.7 × transformer_confidence) + (0.3 × adaptive_confidence)
```
- Weighted average: 70% transformer, 30% adaptive
- Uses the confidence for each model's predicted label

### Routing Decision
- **confidence > 0.80** → `routing: "accept"` (use prediction automatically)
- **confidence ≤ 0.80** → `routing: "human_feedback"` (requires human review)

---

## Rate Limits

Currently, there are no rate limits enforced. For production deployments, consider implementing:
- Rate limiting per IP address
- API key-based rate limits
- Request throttling

---

## Best Practices

1. **Always check `/health`** before making classification requests
2. **Use batch endpoint** for multiple complaints to reduce API calls
3. **Submit feedback** for low-confidence predictions to improve the system
4. **Validate categories** using `/labels` before submitting feedback
5. **Handle errors gracefully** - check status codes and error messages
6. **Cache label list** - `/labels` doesn't change frequently

---

## Example Integration (Python)

```python
import requests

BASE_URL = "http://localhost:5015"

# Check health
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Get available labels
response = requests.get(f"{BASE_URL}/labels")
labels = response.json()["labels"]
print(f"Available categories: {len(labels)}")

# Classify a complaint
response = requests.post(
    f"{BASE_URL}/classify",
    json={
        "text": "Street light not working near my house",
        "return_probabilities": False
    }
)
result = response.json()
print(f"Predicted: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Routing: {result['routing']}")

# Submit feedback if needed
if result["routing"] == "human_feedback":
    feedback_response = requests.post(
        f"{BASE_URL}/feedback",
        json={
            "complaint_text": "Street light not working near my house",
            "correct_category": "Street Lights"
        }
    )
    print(feedback_response.json())
```

---

## Example Integration (JavaScript/Node.js)

```javascript
const axios = require('axios');

const BASE_URL = 'http://localhost:5015';

async function classifyComplaint(text) {
  try {
    // Check health
    const health = await axios.get(`${BASE_URL}/health`);
    console.log('API Status:', health.data);

    // Classify
    const response = await axios.post(`${BASE_URL}/classify`, {
      text: text,
      return_probabilities: false
    });

    const result = response.data;
    console.log(`Predicted: ${result.label}`);
    console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
    console.log(`Routing: ${result.routing}`);

    // Submit feedback if needed
    if (result.routing === 'human_feedback') {
      await axios.post(`${BASE_URL}/feedback`, {
        complaint_text: text,
        correct_category: result.label // or user-selected category
      });
      console.log('Feedback submitted');
    }

    return result;
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
    throw error;
  }
}

// Usage
classifyComplaint('Street light not working near my house');
```

---

## Example Integration (cURL)

```bash
#!/bin/bash

BASE_URL="http://localhost:5015"

# Health check
curl -X GET "$BASE_URL/health"

# Get labels
curl -X GET "$BASE_URL/labels"

# Classify
curl -X POST "$BASE_URL/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Street light not working",
    "return_probabilities": false
  }'

# Batch classify
curl -X POST "$BASE_URL/classify/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Complaint 1", "Complaint 2"]
  }'

# Submit feedback
curl -X POST "$BASE_URL/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "complaint_text": "Street light not working",
    "correct_category": "Street Lights"
  }'
```

---

### 7. Interactive API Documentation (Swagger UI)

FastAPI automatically generates interactive API documentation with Swagger UI.

**Complete Endpoint:** `GET http://localhost:5015/docs`  
**Endpoint Path:** `/docs`

**Features:**
- Interactive API testing
- Request/response schemas
- Try-it-out functionality
- Automatic validation
- Example requests

### 8. Interactive API Documentation (ReDoc)

Alternative API documentation interface.

**Complete Endpoint:** `GET http://localhost:5015/redoc`  
**Endpoint Path:** `/redoc`

**Features:**
- Clean, readable documentation format
- Request/response schemas
- Better for reading and understanding API structure

---

## Support

For issues or questions:
1. Check server logs for detailed error messages
2. Verify model files are present in `CMS_RoBerta/model/`
3. Ensure all dependencies are installed (`pip install -r requirements.txt`)
4. Check that port 5015 is not already in use

---

## Changelog

### Version 1.0.0
- Initial API release
- Single and batch classification
- Human feedback endpoint
- Hybrid ensemble (XLM-RoBERTa + SGDClassifier)
- Confidence-based routing
- Adaptive learning from feedback
