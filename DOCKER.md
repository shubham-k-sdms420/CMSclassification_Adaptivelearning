# Docker Setup for CMS Complaint Classification API

## Prerequisites

- Docker (version 20.10+)
- Docker Compose (v2.0+)

## Quick Start

### Build and Run with Docker Compose

```bash
# Build and start the container
docker compose up --build

# Run in detached mode (background)
docker compose up -d --build

# View logs
docker compose logs -f

# Stop the container
docker compose down
```

### Build and Run with Docker

```bash
# Build the image
docker build -t cms-roberta-api:latest .

# Run the container
docker run -d \
  --name cms-complaint-classifier \
  -p 5015:5015 \
  -v $(pwd)/CMS_RoBerta/model:/app/CMS_RoBerta/model \
  -e MODEL_DIR=/app/CMS_RoBerta/model \
  cms-roberta-api:latest

# View logs
docker logs -f cms-complaint-classifier

# Stop the container
docker stop cms-complaint-classifier
docker rm cms-complaint-classifier
```

## Access the API

Once the container is running:

- **API**: http://localhost:5015
- **UI**: http://localhost:5015/ui
- **Health Check**: http://localhost:5015/health
- **API Docs**: http://localhost:5015/docs

## Test the API

```bash
# Health check
curl http://localhost:5015/health

# Classify a complaint
curl -X POST http://localhost:5015/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Street light not working near my house"}'
```

## Notes

- The model directory is mounted as a volume, so model files and the adaptive classifier (`adaptive_classifier.pkl`) persist between container restarts.
- First startup may take 30-60 seconds to load the transformer model.
- The container includes a health check that verifies the API is responding.
