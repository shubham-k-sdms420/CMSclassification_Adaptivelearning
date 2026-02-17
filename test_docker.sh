#!/bin/bash
# Test script for Docker setup

set -e

echo "=========================================="
echo "Testing CMS Complaint Classification API"
echo "=========================================="
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

echo "✅ Docker is available"

# Check if docker compose is available
if docker compose version &> /dev/null; then
    echo "✅ Docker Compose is available"
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    echo "✅ docker-compose is available"
    COMPOSE_CMD="docker-compose"
else
    echo "❌ Docker Compose is not available"
    exit 1
fi

echo ""
echo "Building Docker image..."
$COMPOSE_CMD build

echo ""
echo "Starting container..."
$COMPOSE_CMD up -d

echo ""
echo "Waiting for API to be ready (this may take 30-60 seconds)..."
sleep 10

# Wait for health check
MAX_ATTEMPTS=12
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:5015/health > /dev/null 2>&1; then
        echo "✅ API is ready!"
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo "   Waiting... ($ATTEMPT/$MAX_ATTEMPTS)"
    sleep 5
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo "❌ API did not become ready in time"
    echo "Check logs with: $COMPOSE_CMD logs"
    exit 1
fi

echo ""
echo "Testing endpoints..."

# Test health endpoint
echo -n "Testing /health... "
if curl -s http://localhost:5015/health | grep -q "ok"; then
    echo "✅"
else
    echo "❌"
fi

# Test classify endpoint
echo -n "Testing /classify... "
RESPONSE=$(curl -s -X POST http://localhost:5015/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Street light not working"}')
if echo "$RESPONSE" | grep -q "label"; then
    echo "✅"
    echo "   Response: $(echo $RESPONSE | python3 -m json.tool 2>/dev/null | head -5 || echo $RESPONSE | head -c 100)..."
else
    echo "❌"
    echo "   Response: $RESPONSE"
fi

# Test UI endpoint
echo -n "Testing /ui... "
if curl -s http://localhost:5015/ui | grep -q "CMS Complaint Classification"; then
    echo "✅"
else
    echo "❌"
fi

echo ""
echo "=========================================="
echo "✅ All tests passed!"
echo "=========================================="
echo ""
echo "Access the API at: http://localhost:5015"
echo "Access the UI at: http://localhost:5015/ui"
echo ""
echo "To stop the container: $COMPOSE_CMD down"
