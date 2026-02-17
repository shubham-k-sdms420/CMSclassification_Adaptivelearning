# CMS Complaint Classification API - Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY CMS_RoBerta/ ./CMS_RoBerta/
COPY requirements.txt .

# Expose port
EXPOSE 5015

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/CMS_RoBerta/model

# Run the application
CMD ["python3", "-m", "uvicorn", "CMS_RoBerta.app.main:app", "--host", "0.0.0.0", "--port", "5015"]
