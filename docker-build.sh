#!/bin/bash

# Docker build script for NepaliGov-RAG-Bench
set -e

echo "🐳 Building NepaliGov-RAG-Bench Docker image..."

# Build the main application image
echo "Building main application image..."
docker build -t nepali-gov-rag-bench:latest .

# Build the documentation image
echo "Building documentation image..."
docker build -f Dockerfile.docs -t nepali-gov-docs:latest .

echo "✅ Docker images built successfully!"
echo ""
echo "📋 Available images:"
docker images | grep nepali-gov

echo ""
echo "🚀 To run the application:"
echo "   docker-compose up -d"
echo ""
echo "📚 To run documentation only:"
echo "   docker run -p 8080:8080 -v \$(pwd)/docs:/app/docs nepali-gov-docs:latest"
echo ""
echo "🔍 To run main application only:"
echo "   docker run -p 8093:8093 -v \$(pwd)/data:/app/data nepali-gov-rag-bench:latest"
