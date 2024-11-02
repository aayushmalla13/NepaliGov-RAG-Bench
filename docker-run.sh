#!/bin/bash

# Docker run script for NepaliGov-RAG-Bench
set -e

echo "🐳 Starting NepaliGov-RAG-Bench with Docker..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Create necessary directories if they don't exist
mkdir -p data cache config logs

# Start services
echo "Starting services..."
docker compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Check health
echo "Checking service health..."
if docker compose ps | grep -q "Up"; then
    echo "✅ Services are running!"
    echo ""
    echo "🌐 Application: http://localhost:8093"
    echo "📚 Documentation: http://localhost:8080"
    echo ""
    echo "📋 Service status:"
    docker compose ps
    echo ""
    echo "📝 To view logs:"
    echo "   docker compose logs -f"
    echo ""
    echo "🛑 To stop services:"
    echo "   docker compose down"
else
    echo "❌ Services failed to start. Check logs:"
    docker compose logs
    exit 1
fi
