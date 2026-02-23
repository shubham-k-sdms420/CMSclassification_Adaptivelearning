# CMS Complaint Classification API — Integration Guide

**Version:** 1.0.0
**Date:** 12 February 2026
**Prepared for:** PMC CMS Development Team
**Prepared by:** AI/ML Team

---

## Table of Contents

1. [Overview](#1-overview)
2. [Base URL & Connectivity](#2-base-url--connectivity)
3. [General Information](#3-general-information)
4. [API Endpoints](#4-api-endpoints)
   - 4.1 [Health Check](#41-health-check---get-health)
   - 4.2 [List Labels](#42-list-all-labels---get-labels)
   - 4.3 [Classify Single Complaint](#43-classify-single-complaint---post-classify)
   - 4.4 [Classify Batch of Complaints](#44-classify-batch-of-complaints---post-classifybatch)
5. [Classification Categories](#5-classification-categories)
6. [Error Handling](#6-error-handling)
7. [Integration Examples](#7-integration-examples)
   - 7.1 [cURL](#71-curl)
   - 7.2 [Python](#72-python)
   - 7.3 [JavaScript / Node.js](#73-javascript--nodejs)
   - 7.4 [C# / .NET](#74-c--net)
   - 7.5 [Java](#75-java)
8. [Integration Workflow & Best Practices](#8-integration-workflow--best-practices)
9. [FAQ](#9-faq)
10. [Support & Contact](#10-support--contact)

---

## 1. Overview

The **CMS Complaint Classification API** is an AI-powered service that automatically classifies citizen/customer complaint text into one of **78 predefined PMC department categories**. It uses a fine-tuned **XLM-RoBERTa-Large** deep learning model and supports both English and Marathi complaint text.

### What It Does

- Accepts complaint text (single or batch)
- Returns the predicted **department/category label** and its numeric ID
- Optionally returns **confidence probabilities** for all 78 categories

### Key Capabilities

| Feature | Details |
|---------|---------|
| Model | XLM-RoBERTa-Large (fine-tuned, Stage 2) |
| Categories | 78 PMC department categories |
| Languages | English, Marathi (multilingual support) |
| Max Input | 512 tokens (~300-400 words) |
| Batch Support | Yes (multiple complaints in one request) |
| Authentication | None required (internal network) |
| Response Format | JSON |

---

## 2. Base URL & Connectivity

| Environment | Base URL |
|-------------|----------|
| **Production** | `http://34.227.36.59:5016` |

All endpoint paths documented below are relative to this base URL.

**Example:** To call the health endpoint, the full URL is:
```
http://34.227.36.59:5016/health
```

### Interactive API Documentation (Swagger UI)

A built-in interactive API explorer is available at:
```
http://34.227.36.59:5016/docs
```

You can test all endpoints directly from the browser using this interface.

---

## 3. General Information

### Protocol & Content Type

| Parameter | Value |
|-----------|-------|
| Protocol | HTTP |
| Content-Type (Request) | `application/json` |
| Content-Type (Response) | `application/json` |
| Character Encoding | UTF-8 |
| Authentication | **None** (no API key or token required) |

### HTTP Methods Used

| Method | Used For |
|--------|----------|
| `GET` | Health check, retrieving labels |
| `POST` | Classifying complaint text |

### Rate Limits

There are no rate limits configured at this time. However, for batch operations, we recommend keeping batch sizes reasonable (under 100 texts per request) for optimal response time.

---

## 4. API Endpoints

---

### 4.1 Health Check — `GET /health`

Check whether the API service and the ML model are running and ready to accept requests.

**Use this endpoint for:** Monitoring, service health dashboards, integration readiness checks.

#### Request

```
GET http://34.227.36.59:5016/health
```

No request body or parameters required.

#### Response

**Status Code:** `200 OK`

```json
{
  "status": "ok",
  "model_loaded": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"ok"` when the server is running |
| `model_loaded` | boolean | `true` if the ML model is loaded and ready; `false` if still loading |

> **Important:** Only send classification requests when `model_loaded` is `true`.

---

### 4.2 List All Labels — `GET /labels`

Retrieve the complete list of all 78 classification category labels that the model can predict.

**Use this endpoint for:** Populating dropdown menus, validating model output, syncing category lists.

#### Request

```
GET http://34.227.36.59:5016/labels
```

No request body or parameters required.

#### Response

**Status Code:** `200 OK`

```json
{
  "labels": [
    "Abhyagat Kaksha",
    "Aids Control",
    "AutoDCR",
    "BSUP project",
    "Bhavan",
    "..."
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `labels` | array of strings | All 78 category labels, sorted by their internal numeric ID (0–77) |

#### Error Response

**Status Code:** `503 Service Unavailable` — Model not yet loaded.

---

### 4.3 Classify Single Complaint — `POST /classify`

Classify a single complaint text and get the predicted department/category.

**This is the primary endpoint for real-time classification of individual complaints.**

#### Request

```
POST http://34.227.36.59:5016/classify
Content-Type: application/json
```

**Request Body:**

```json
{
  "text": "Road potholes near Shivaji Nagar main market causing accidents",
  "return_probabilities": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | **Yes** | — | The complaint text to classify. Can be in English or Marathi. |
| `return_probabilities` | boolean | No | `false` | If `true`, returns confidence scores for all 78 categories. |

#### Response (without probabilities)

**Status Code:** `200 OK`

```json
{
  "label": "Road, pavement, divider, pits, repair / new speed breaker / zebra crossing",
  "label_id": 60,
  "probabilities": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | The predicted category name (department) |
| `label_id` | integer | The numeric ID of the predicted category (0–77) |
| `probabilities` | object or null | `null` when `return_probabilities` is `false` |

#### Response (with probabilities)

When `return_probabilities` is set to `true`:

```json
{
  "label": "Road, pavement, divider, pits, repair / new speed breaker / zebra crossing",
  "label_id": 60,
  "probabilities": {
    "Abhyagat Kaksha": 0.0001,
    "Aids Control": 0.0002,
    "AutoDCR": 0.0001,
    "...": "...",
    "Road, pavement, divider, pits, repair / new speed breaker / zebra crossing": 0.9523,
    "...": "...",
    "Water Supply": 0.0003
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `probabilities` | object | Key-value pairs of `"category_name": probability_score`. Probabilities are floats between 0 and 1, rounded to 4 decimal places. All 78 categories are included. The sum of all probabilities is approximately 1.0. |

#### Error Responses

| Status Code | Condition | Response Body |
|-------------|-----------|---------------|
| `400 Bad Request` | Empty or missing `text` field | `{"detail": "text must be non-empty"}` |
| `422 Unprocessable Entity` | Invalid JSON / missing required fields | Validation error details |
| `503 Service Unavailable` | Model not loaded | `{"detail": "Model not loaded"}` |

---

### 4.4 Classify Batch of Complaints — `POST /classify/batch`

Classify multiple complaint texts in a single request. Useful for bulk processing.

**Use this endpoint for:** Batch imports, bulk re-classification, periodic processing jobs.

#### Request

```
POST http://34.227.36.59:5016/classify/batch
Content-Type: application/json
```

**Request Body:**

```json
{
  "texts": [
    "Water supply is not available since 3 days in Kothrud area",
    "Stray dogs are creating nuisance in my colony at Hadapsar",
    "Street light not working at Sinhagad road near PMC school"
  ],
  "return_probabilities": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `texts` | array of strings | **Yes** | — | List of complaint texts to classify. Each can be in English or Marathi. |
| `return_probabilities` | boolean | No | `false` | If `true`, returns confidence scores for all categories for each prediction. |

#### Response

**Status Code:** `200 OK`

```json
{
  "predictions": [
    {
      "label": "Water Supply",
      "label_id": 77,
      "probabilities": null
    },
    {
      "label": "Stray Dogs",
      "label_id": 66,
      "probabilities": null
    },
    {
      "label": "Street Lights",
      "label_id": 67,
      "probabilities": null
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `predictions` | array of objects | One prediction object per input text, in the same order as the input `texts` array |
| `predictions[].label` | string | Predicted category name |
| `predictions[].label_id` | integer | Numeric ID of the predicted category (0–77) |
| `predictions[].probabilities` | object or null | Per-category confidence scores (only when `return_probabilities` is `true`) |

> **Note:** If an empty `texts` array is sent, the response will be `{"predictions": []}`.

#### Error Responses

| Status Code | Condition | Response Body |
|-------------|-----------|---------------|
| `422 Unprocessable Entity` | Invalid JSON / missing `texts` field | Validation error details |
| `503 Service Unavailable` | Model not loaded | `{"detail": "Model not loaded"}` |

---

## 5. Classification Categories

The model classifies complaints into the following **78 categories**. The `label_id` corresponds to the numeric ID returned in responses.

| label_id | Category Label |
|----------|---------------|
| 0 | Abhyagat Kaksha |
| 1 | Aids Control |
| 2 | AutoDCR |
| 3 | BSUP project |
| 4 | Bhavan |
| 5 | Bicycle Plan |
| 6 | Birth And Death |
| 7 | Bogus doctor / pregnancy diagnosis / unauthorized sonography center |
| 8 | Bridge / Overbridge / Subway / pedestrian |
| 9 | Building Permission |
| 10 | Chawl Complaints |
| 11 | City Development Plan |
| 12 | City Family Welfare |
| 13 | Communicable Disease |
| 14 | Contributed Health Scheme |
| 15 | Crematorium electrical work(H.O) |
| 16 | Cultural Department |
| 17 | Drainage |
| 18 | Electrical(H.O) |
| 19 | Employee Transfer / Promotion / Service Record/Others (GAD/Est.)/Regarding Officer/Women Harassment |
| 20 | Encroachments on public premises / roads |
| 21 | Environment (H.O) |
| 22 | Fogging / Mosquito nuisance / Dengue malaria disease |
| 23 | Garbage Depot Complaint |
| 24 | Garden Civil maintenance work |
| 25 | Garden Cleaning & maintenance |
| 26 | Garden Electric maintenance work |
| 27 | General Administration |
| 28 | General Complaints |
| 29 | Heritage Cell (Bhavan HO) |
| 30 | Immunization |
| 31 | Information Technology |
| 32 | Information Technology-Employee |
| 33 | JICA Project related Complaint |
| 34 | Land Acquisition |
| 35 | Lashkar Water Supply |
| 36 | License (Parwana) |
| 37 | Lifting of bird flu deceased birds |
| 38 | Mandai |
| 39 | Marriage Registration |
| 40 | Medicine Supplies |
| 41 | National Programms |
| 42 | PMC Hospitals treatment |
| 43 | PMC Properties |
| 44 | PMC Security Complaints |
| 45 | Pension Complaints |
| 46 | Pradhan Mantri Awas Yojana (PMAY) |
| 47 | Primary Education |
| 48 | Property Tax Assessment / Payment |
| 49 | Property Tax-COMPUTER SECTION |
| 50 | Property Tax-MOBILE TOWER |
| 51 | Property Tax-SOLAR VERMI & RAINWATER HARVESTING (SVR) |
| 52 | Property Tax-TAT(Title Transfer) |
| 53 | Providant Fund / CPS |
| 54 | Public Health Related |
| 55 | Pune Smart City (PSCDCL) |
| 56 | RCH / NUHM |
| 57 | RTI Related Complaint |
| 58 | River Front development Project |
| 59 | Road Sweeping / Toilet Cleaning / Garbage disposal |
| 60 | Road, pavement, divider, pits, repair / new speed breaker / zebra crossing |
| 61 | Secondary and Technical Education |
| 62 | Slum |
| 63 | Social Development Schemes |
| 64 | Sports |
| 65 | Stray Animals |
| 66 | Stray Dogs |
| 67 | Street Lights |
| 68 | TDR |
| 69 | Traffic Signal |
| 70 | Traffic-Planning |
| 71 | Tree Authority |
| 72 | Unauthorized banners / advertisements / Permit / license in Pune City |
| 73 | Unauthorized hoardigs banners / advertisements on roads / footpath / buildings / directions panels |
| 74 | Unauthorized slaughterhouse / crude meat |
| 75 | Urban Poor Scheme |
| 76 | Vehicle department |
| 77 | Water Supply |

---

## 6. Error Handling

### Standard Error Response Format

All errors are returned as JSON with an HTTP error status code:

```json
{
  "detail": "Human-readable error message"
}
```

### Error Codes Summary

| HTTP Status | Meaning | When It Occurs | Recommended Action |
|-------------|---------|----------------|-------------------|
| `400` | Bad Request | Empty `text` field in `/classify` | Validate input before sending |
| `422` | Unprocessable Entity | Malformed JSON or missing required fields | Check request body format |
| `500` | Internal Server Error | Unexpected server-side error | Retry after a short delay; contact support if persistent |
| `503` | Service Unavailable | Model is still loading or not loaded | Wait and retry; check `/health` endpoint |

### Recommended Retry Strategy

For `503` and `500` errors, implement a simple retry with exponential backoff:

1. Wait 2 seconds, then retry
2. Wait 4 seconds, then retry
3. Wait 8 seconds, then retry
4. After 3 retries, log the error and alert

---

## 7. Integration Examples

### 7.1 cURL

**Single complaint classification:**

```bash
curl -X POST http://34.227.36.59:5016/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Water supply not available since 2 days in Kothrud area",
    "return_probabilities": false
  }'
```

**Batch classification:**

```bash
curl -X POST http://34.227.36.59:5016/classify/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Garbage not collected from our society since 1 week",
      "Street light is not working near Shivaji Park",
      "Stray dogs attacking people in Hadapsar area"
    ],
    "return_probabilities": false
  }'
```

**Health check:**

```bash
curl http://34.227.36.59:5016/health
```

**Get all labels:**

```bash
curl http://34.227.36.59:5016/labels
```

---

### 7.2 Python

```python
import requests

BASE_URL = "http://34.227.36.59:5016"

# --- Single Complaint Classification ---
def classify_complaint(text, return_probabilities=False):
    """Classify a single complaint text."""
    response = requests.post(
        f"{BASE_URL}/classify",
        json={
            "text": text,
            "return_probabilities": return_probabilities
        }
    )
    response.raise_for_status()  # Raises exception for HTTP errors
    return response.json()

# Example usage
result = classify_complaint("Water supply not available since 2 days in Kothrud area")
print(f"Category: {result['label']}")
print(f"Label ID: {result['label_id']}")


# --- Batch Classification ---
def classify_batch(texts, return_probabilities=False):
    """Classify multiple complaints in a single request."""
    response = requests.post(
        f"{BASE_URL}/classify/batch",
        json={
            "texts": texts,
            "return_probabilities": return_probabilities
        }
    )
    response.raise_for_status()
    return response.json()

# Example usage
complaints = [
    "Garbage not collected from our society since 1 week",
    "Street light not working near Shivaji Park",
    "Pothole on main road near market area"
]
results = classify_batch(complaints)
for i, prediction in enumerate(results["predictions"]):
    print(f"Complaint {i+1}: {prediction['label']} (ID: {prediction['label_id']})")


# --- Health Check ---
def check_health():
    """Check if the API and model are ready."""
    response = requests.get(f"{BASE_URL}/health")
    return response.json()

health = check_health()
print(f"Service status: {health['status']}, Model loaded: {health['model_loaded']}")


# --- Get All Labels ---
def get_labels():
    """Retrieve all possible classification labels."""
    response = requests.get(f"{BASE_URL}/labels")
    response.raise_for_status()
    return response.json()["labels"]

labels = get_labels()
print(f"Total categories: {len(labels)}")
```

---

### 7.3 JavaScript / Node.js

```javascript
const BASE_URL = "http://34.227.36.59:5016";

// --- Single Complaint Classification ---
async function classifyComplaint(text, returnProbabilities = false) {
  const response = await fetch(`${BASE_URL}/classify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: text,
      return_probabilities: returnProbabilities,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`API Error ${response.status}: ${error.detail}`);
  }

  return await response.json();
}

// Example usage
classifyComplaint("Water supply not available since 2 days in Kothrud area")
  .then((result) => {
    console.log(`Category: ${result.label}`);
    console.log(`Label ID: ${result.label_id}`);
  })
  .catch(console.error);


// --- Batch Classification ---
async function classifyBatch(texts, returnProbabilities = false) {
  const response = await fetch(`${BASE_URL}/classify/batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      texts: texts,
      return_probabilities: returnProbabilities,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`API Error ${response.status}: ${error.detail}`);
  }

  return await response.json();
}

// Example usage
const complaints = [
  "Garbage not collected from our society since 1 week",
  "Street light not working near Shivaji Park",
  "Pothole on main road near market area",
];

classifyBatch(complaints).then((results) => {
  results.predictions.forEach((pred, i) => {
    console.log(`Complaint ${i + 1}: ${pred.label} (ID: ${pred.label_id})`);
  });
});


// --- Health Check ---
async function checkHealth() {
  const response = await fetch(`${BASE_URL}/health`);
  return await response.json();
}
```

---

### 7.4 C# / .NET

```csharp
using System.Net.Http;
using System.Text;
using System.Text.Json;

public class CmsClassificationClient
{
    private readonly HttpClient _client;
    private readonly string _baseUrl;

    public CmsClassificationClient(string baseUrl = "http://34.227.36.59:5016")
    {
        _client = new HttpClient();
        _baseUrl = baseUrl;
    }

    // --- Single Complaint Classification ---
    public async Task<JsonElement> ClassifyAsync(string text, bool returnProbabilities = false)
    {
        var payload = new
        {
            text = text,
            return_probabilities = returnProbabilities
        };

        var content = new StringContent(
            JsonSerializer.Serialize(payload),
            Encoding.UTF8,
            "application/json"
        );

        var response = await _client.PostAsync($"{_baseUrl}/classify", content);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<JsonElement>(json);
    }

    // --- Batch Classification ---
    public async Task<JsonElement> ClassifyBatchAsync(List<string> texts, bool returnProbabilities = false)
    {
        var payload = new
        {
            texts = texts,
            return_probabilities = returnProbabilities
        };

        var content = new StringContent(
            JsonSerializer.Serialize(payload),
            Encoding.UTF8,
            "application/json"
        );

        var response = await _client.PostAsync($"{_baseUrl}/classify/batch", content);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<JsonElement>(json);
    }

    // --- Health Check ---
    public async Task<JsonElement> CheckHealthAsync()
    {
        var response = await _client.GetAsync($"{_baseUrl}/health");
        var json = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<JsonElement>(json);
    }
}

// Usage:
// var client = new CmsClassificationClient();
// var result = await client.ClassifyAsync("Water supply problem in Kothrud");
// Console.WriteLine(result.GetProperty("label").GetString());
```

---

### 7.5 Java

```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class CmsClassificationClient {

    private final HttpClient client;
    private final String baseUrl;

    public CmsClassificationClient() {
        this("http://34.227.36.59:5016");
    }

    public CmsClassificationClient(String baseUrl) {
        this.client = HttpClient.newHttpClient();
        this.baseUrl = baseUrl;
    }

    // --- Single Complaint Classification ---
    public String classify(String text, boolean returnProbabilities) throws Exception {
        String jsonBody = String.format(
            "{\"text\": \"%s\", \"return_probabilities\": %s}",
            text.replace("\"", "\\\""),
            returnProbabilities
        );

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(baseUrl + "/classify"))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
            .build();

        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new RuntimeException("API Error: " + response.statusCode() + " - " + response.body());
        }

        return response.body();
    }

    // --- Health Check ---
    public String checkHealth() throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(baseUrl + "/health"))
            .GET()
            .build();

        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        return response.body();
    }

    // Usage:
    // CmsClassificationClient client = new CmsClassificationClient();
    // String result = client.classify("Water supply problem in Kothrud", false);
    // System.out.println(result);
}
```

---

## 8. Integration Workflow & Best Practices

### Recommended Integration Flow

```
┌─────────────────────────────────────────────────────────┐
│                   CMS Application                        │
│                                                          │
│  1. Citizen submits complaint text                       │
│              │                                           │
│              ▼                                           │
│  2. CMS sends POST /classify with complaint text         │
│              │                                           │
│              ▼                                           │
│  3. API returns predicted category (label + label_id)    │
│              │                                           │
│              ▼                                           │
│  4. CMS auto-assigns complaint to the predicted          │
│     department OR suggests it for manual review           │
│              │                                           │
│              ▼                                           │
│  5. (Optional) Officer confirms/overrides category       │
└─────────────────────────────────────────────────────────┘
```

### Best Practices

1. **Health Check on Startup:**
   Call `GET /health` when your application starts and verify `model_loaded` is `true` before routing complaints to the API.

2. **Use `label_id` for Database Mapping:**
   Store the `label_id` (integer) in your database for internal mapping. Use the `label` (string) for display to end users.

3. **Use Batch Endpoint for Bulk Operations:**
   When processing multiple complaints (e.g., importing historical data or batch jobs), use `POST /classify/batch` instead of calling `/classify` in a loop. It is significantly faster.

4. **Handle Timeouts Gracefully:**
   Set a reasonable HTTP timeout (e.g., 30 seconds for single, 60 seconds for batch). If a timeout occurs, retry once, then queue the complaint for later classification.

5. **Confidence-Based Routing (Optional):**
   Use `return_probabilities: true` to implement confidence thresholds. For example:
   - If the top prediction probability > 0.80 → auto-assign to department
   - If the top prediction probability is 0.50–0.80 → suggest category but require manual confirmation
   - If the top prediction probability < 0.50 → flag for manual classification

6. **Keep Labels in Sync:**
   Periodically call `GET /labels` to ensure your system's category list is in sync with the model's output categories.

7. **Input Quality:**
   - Send the complaint text as-is; the model handles preprocessing internally
   - Longer, more descriptive text generally yields more accurate predictions
   - The model handles both English and Marathi text

---

## 9. FAQ

**Q: Is authentication required?**
A: No. The API does not require any authentication, API keys, or tokens. It is intended for use within the internal network.

**Q: What is the maximum text length?**
A: The model processes up to 512 tokens (~300-400 words). Longer text is automatically truncated. For best results, send the most relevant portion of the complaint.

**Q: Can I send Marathi text?**
A: Yes. The model (XLM-RoBERTa) is multilingual and supports both English and Marathi complaint text natively.

**Q: How fast is the API?**
A: Single complaint classification typically responds in under 1 second. Batch classification time depends on the batch size (roughly proportional).

**Q: What if the API returns a 503 error?**
A: This means the model is still loading (typically happens just after a server restart). Wait 30-60 seconds and retry. Use the `/health` endpoint to check when the model is ready.

**Q: Can I get the top-N predictions instead of just one?**
A: Set `return_probabilities: true` and sort the `probabilities` object by value in descending order on your side to get top-N predictions.

**Q: What happens if I send an empty text?**
A: The `/classify` endpoint returns a `400 Bad Request` error with `{"detail": "text must be non-empty"}`.

**Q: Is there a limit on batch size?**
A: There is no hard limit, but we recommend keeping batches under 100 texts for optimal response time and memory usage.

---

## 10. Support & Contact

For any issues or questions regarding this API, please contact the AI/ML team:

| Contact | Details |
|---------|---------|
| Team | AI/ML Development Team |
| Email | *(to be updated)* |
| Swagger Docs | http://34.227.36.59:5016/docs |

---

*This document is confidential and intended for PMC CMS Development Team internal use only.*
