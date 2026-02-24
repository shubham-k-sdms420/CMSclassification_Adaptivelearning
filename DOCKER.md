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
  -p 5016:5016 \
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

- **API**: http://localhost:5016
- **UI**: http://localhost:5016/ui
- **Health Check**: http://localhost:5016/health
- **API Docs**: http://localhost:5016/docs

## Test the API

```bash
# Health check
curl http://localhost:5016/health

# Classify a complaint
curl -X POST http://localhost:5016/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Street light not working near my house"}'
```

## Notes

- The model directory is mounted as a volume, so model files and the adaptive classifier (`adaptive_classifier.pkl`) persist between container restarts.
- First startup may take 30-60 seconds to load the transformer model.
- The container includes a health check that verifies the API is responding.

## Deployment: Model files required

**The Docker image does not include the transformer model weights.** The error  
`OSError: Error no file named model.safetensors, or pytorch_model.bin, found in directory /app/CMS_RoBerta/model`  
means the directory mounted at `/app/CMS_RoBerta/model` is missing the weight file.

**Before starting the container (or before the first deploy):**

1. On the **host** (or in the volume you mount), ensure **`CMS_RoBerta/model/`** contains:
   - **`model.safetensors`** (or **`pytorch_model.bin`**) — transformer weights (~2 GB), from the project [Google Drive folder](https://drive.google.com/drive/folders/1qA-Sg2pVO9TXpux_LRNsJ-gVYbksNnGS?usp=sharing)
   - **`config.json`**, **`tokenizer_config.json`**, **`tokenizer.json`**, **`label2id.json`** (and any other files from that folder)
   - Optionally **`adaptive_classifier.pkl`** from [this Drive link](https://drive.google.com/file/d/10bwLoCl8lQgqHnxbkzuLtXJe3rdJ74L0/view?usp=sharing) for the adaptive classifier

2. **Docker Compose:** The compose file mounts `./CMS_RoBerta/model` from the **current working directory**. So on the deployment server, either:
   - Clone the repo, then download/copy the model files into `CMS_RoBerta/model/`, then run `docker compose up --build`, or
   - Mount a different host path that already contains the model files, e.g.  
     `- /opt/cms-model:/app/CMS_RoBerta/model`

3. **Kubernetes / other orchestrators:** Use a volume (e.g. PVC, NFS, or init container that downloads the model) that contains the above files and mount it at `/app/CMS_RoBerta/model`, and set `MODEL_DIR=/app/CMS_RoBerta/model`.
